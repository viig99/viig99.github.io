+++
title = "Building a Chess Bot with HSTU: From Lichess Pretraining to Value Search"
description = "A running build log on training a compact chess policy from Lichess games with an HSTU sequence model, adding a WDL value head, and turning it into ~2250-strength play with a sequential-halving value search. Plus the current bet: distilling that search back into the policy."
tags = [
  "chess",
  "HSTU",
  "next-token prediction",
  "value head",
  "sequential halving",
  "MCTS",
  "policy distillation",
  "on-policy distillation",
  "Lichess",
  "Stockfish",
  "sequential modeling",
]
date = "2026-07-16"
categories = [
  "Machine Learning",
  "Chess",
  "Systems"
]
menu = "main"
draft = false
+++

## Build Log Snapshot

This is my running build log for [`imba-chess`](https://github.com/viig99/imba-chess). I update it as things change, so it reads out of order in places. That is on purpose.

Where it stands today:

- I trained a small HSTU sequence model to imitate high-Elo Lichess games (next-move prediction). It reaches `hr@10` around `0.92` and `top-1` around `0.43` on held-out games.
- On its own, that policy is weak at actual chess. Greedy play scores `0.21` against Stockfish limited to 1400 Elo. Good imitation, bad chess.
- Adding a WDL value head and a value-guided search at move-selection time changes the story. The current best system scores `0.595` against Stockfish at 2200 Elo, which puts the whole thing at roughly **2250 Elo**. That is about an 800 to 1000 Elo jump over the raw policy, with no reinforcement learning and no self-play yet.
- The search that got me there is `value_search_halving`: sequential halving to decide which root move to spend compute on, plus a prior-ordered beam for the tree underneath it.
- The current bet is distilling that search back into the policy head, so the policy itself gets stronger without paying for search at inference. This is grounded in the Gumbel MuZero policy-improvement result. It is designed and being built, not yet a proven win.

Two model sizes show up below: `v3` (512-dim, 6 layers, about 10M params) and `v4` (768-dim, 8 layers, about 27M params). The bigger `v4` trunk finally hit my original 20-30M target and it mattered. Most of the dev loop runs on modest hardware: an 8GB laptop GPU (RTX 3070 class), some runs on a 4090, and a rented 5090 for a while to speed up rollout generation.

My core framing has not changed: **a chess game is a structured event sequence**, and the recsys sequence toolkit should transfer to it.

---

## What Triggered This Project

Before this, I finished an STU + FlexAttention rewrite and tested it on two large recommender datasets, Zvuk-200M and Yambda-500M. That work beat strong SASRec baselines and scaled cleanly. After a point the recsys gains felt marginal, so I wanted a harder sequential-reasoning problem to point the same tools at.

Chess fit:

- two-player sequential decisions,
- long-horizon credit assignment,
- explicit win/draw/loss outcomes,
- cheap large-scale supervised data from Lichess.

So this became a side project with one question behind it: can the same sequence toolkit transfer from recommendation to strategy?

---

## Why Chess and Why HSTU

Most strong chess engines lean on self-play plus heavy search. That works, but it is expensive to build and run. I started from the cheaper end: imitate strong human games first, then bolt value and search on top.

HSTU was built for jagged, structured sequences in recsys. Chess has the same shape: variable-length event sequences, structured per-step state, and a known next-action target. The bet was simple. If it can predict the next item well, it can predict the next move well.

---

## Data Pipeline

### Source

I use the [Lichess open game database](https://database.lichess.org/) through `Lichess/standard-chess-games` on Hugging Face. The `year/month` partitioning makes clean temporal splits easy.

Main filter: average Elo `(WhiteElo + BlackElo) / 2 >= 2000`. I also drop very fast time controls, since bullet games are full of tactical noise that I do not want the policy imitating.

### Temporal Splits

Chronological splits, not random, to avoid leaking the future into training.

| Split | Window |
|---|---|
| train | 2021-01 through 2025-06 |
| val | 2025-07 (single month) |
| test | 2025-08 through 2025-09 |

No model is ever selected on the test split. An earlier run used a longer train window back to 2018 and a stricter 2400-Elo test filter. From those notes: roughly 422M games in the high-Elo pool, ingest around 1.5M games/hour, average game length around 70 to 80 moves.

### PGN Parsing and Board State

Each game is replayed with `python-chess`. At every ply the board becomes a structured token, not a FEN string and not text.

| Field | Values | Notes |
|---|---|---|
| `piece_ids` | `[64]`, 0-12 | 0=empty, 1-6=white, 7-12=black |
| `turn_id` | 0/1 | side to move |
| `castle_id` | 0-15 | KQkq bitmask |
| `ep_file_id` | 0-8 | en-passant file + 1, 0=none |
| `halfmove_bucket_id` | >=0 | bucketed clock |
| `fullmove_bucket_id` | >=0 | bucketed move number |

Targets are UCI move ids from a static vocabulary of every geometrically reachable from-to pair plus promotions (1,970 tokens including specials). That vocabulary provably covers every legal standard-chess move, so the policy never has to invent a token for a move it has not seen.

One encoding detail that bit me: the board embedding is a joint `(piece, square)` table, mean-pooled over the 64 squares. An additive `piece + square` scheme collapses to a bag of material under pooling (it throws away where the pieces are), so the joint table is load-bearing, not a nicety.

### Jagged Batching

Multiple games are packed into one flat token buffer with `seq_offsets` marking boundaries, the same trick from [FlexAttention HSTU at 500M Events](./hstu-for-yambda.md): no cross-game attention, no padding waste.

```text
[BOS | ply1 ply2 ... plyN | BOS | ply1 ... plyM | ...]
       game 1                    game 2
```

No per-token padding mask is materialized. A block mask is derived from `seq_offsets` at runtime.

---

## Model Architecture

### Overview

{{< mermaid >}}
flowchart LR
    A[piece_ids 64 tokens] --> B[E_piece + E_square]
    B --> C[Mean pool to board_emb]
    D[prev_move_id] --> E[E_move]
    F[turn/castle/ep/clk] --> G[E_meta]
    C --> H[Additive event composition]
    E --> H
    G --> H
    H --> I[HSTU backbone, causal layers]
    I --> J[Policy head to move logits]
    I --> K[Value head to loss/draw/win]
{{< /mermaid >}}

Each ply becomes a single event vector by embedding the structured fields and summing them: mean-pooled board embedding, previous-move embedding, and metadata (turn, castling, en passant, clock buckets). The backbone is causal with a relative position bias over plies.

Two sizes are in play: `v3` at about 10M params (512-dim, 6 layers) and `v4` at about 27M (768-dim, 8 layers, with the value loss weighted higher). The `v4` trunk is where the value head got good enough for search to pay off at the higher rungs.

---

## Training Phase 1: Supervised Policy Pretraining

### Objective

Cross-entropy on the next UCI move over the full static vocabulary. BOS positions are excluded with `ignore_index = -100`. This is pure imitation: no reward, no self-play.

### Elo-Weighted Loss

Not all supervision is equally good, so I weight each move by the Elo of the player who made it. Stronger players' moves pull the gradient harder.

```text
norm_i = clamp((elo_i - min_elo) / (max_elo - min_elo), 0, 1)
w_i    = 1 + strength * (norm_i ^ alpha)
loss   = sum_i(w_i * ce_i) / sum_i(w_i)
```

I normalize the weights so gradients stay stable, and use label smoothing because strong positions often have several reasonable moves and I do not want to punish the model for picking a different good one.

### Where Imitation Topped Out

The policy trains well by its own metrics. Held-out `hr@10` crossed `0.9`, `top-1` sat around `0.45`. Then I played it against Stockfish and it lost badly, even to Stockfish limited to 1400. That gap was the most useful thing that happened early on.

Good imitation metrics are necessary but not sufficient for winning play. The objective learns "what humans played", not "what maximizes winning chances". Greedy decoding overcommits to narrow policy modes. Sampling recovers the occasional win but not stable strength. The win/loss signal is only weakly coupled to move-level cross-entropy unless something at inference explicitly optimizes for it.

Pretraining gave me a good prior. It did not give me reliable chess.

---

## Phase 2: The Value Head

### Why

The policy answers "what move do humans pick here". It does not answer "does this move lead to a winning position". Those are different questions, and the second one is what you need to not hang a piece. So the trunk gets a second head.

### WDL Classification

A 3-class head predicts win/draw/loss from the side-to-move perspective. I use 3 classes on purpose rather than a scalar. A scalar `0.0` cannot tell "certain draw" apart from "unclear, 50/50 win or lose", and those are genuinely different situations. A scalar is recovered at inference when I need one:

```text
V(s) = p(win) - p(loss)  in [-1, 1]
```

### The Labels Are Noisy, and What Helps

The label for every position in a game is that game's final result. That is a high-variance Monte-Carlo label. A winning position that the player later threw away gets labeled "loss". Two levers help:

- **Progress weighting.** Each position's value loss is weighted by `progress ^ value_weight_alpha`, where progress is how far into the game the position is. Final outcomes are noisy labels for early positions and clean labels for late ones, so early plies contribute little gradient. I moved `alpha` from `0.9` (near a full linear discount) down to `0.1` after a decile-bucketed held-out study showed `0.1` won in 7 of 10 game-progress buckets, mostly in the early and middle game.
- **Elo-weighting the value loss too.** Stronger players convert winning positions more reliably, so their outcomes are lower-noise value labels. Weighting the value loss by player Elo (the same weighting the policy loss uses) turned out to be the single biggest lever I found at the top rung. More on that below.

### Combined Loss

```text
total_loss = policy_loss + value_loss_weight * value_loss
```

The value head gets its own small MLP capacity (`Linear -> SiLU -> Linear`) so the policy objective does not crowd it out of the shared trunk. `value_loss_weight` is a knob I am still tuning; `v4` runs it higher than `v3` did.

---

## Phase 3: Using the Value Head at Inference

Training a value head barely moves playing strength on its own. The gain comes from using the value during **move selection**. I built these up in order, each one measured against the last on the same checkpoint and the same Stockfish settings.

### Greedy (baseline)

Play the highest-logit legal move. One forward pass, nothing checks consequences, so a natural-looking move that hangs a piece to a two-move tactic gets played anyway. This is the baseline everything else has to beat.

### Value Rerank (1-ply lookahead)

Take the top-K policy candidates, evaluate the position after each with the value head, and play the best:

```text
score(move) = V(position after move) + lambda * log_prob(move)
```

Value does the choosing; the policy log-prob is a near-tie breaker. One thing this taught me is durable: setting `lambda = 0` measurably collapses. With no policy term the search over-exploits value-head noise. It finds the move the value head is most wrong about in the optimistic direction. This is Goodhart, plain and simple, and the prior term is a real regularizer, not decoration.

### Value Search d2 (adversarial lookahead)

Value rerank plus one level of "if I play this, what is the worst reply". The important detail is which opponent replies get evaluated: the policy top-K **plus every capture, check, and promotion**. The refutation of a bad move is often a move the human-imitation policy ranks low, so pruning by probability alone would hide exactly the move that disproves the candidate. Each candidate is scored by its worst reply (`grade = min over responses of V`). This cleared my pre-registered bar for building a real search: `+0.13` over greedy on the same checkpoint.

### Value Search Halving (the one that worked)

d2 spends its budget uniformly. The obviously losing move gets as much attention as the two moves the decision actually hinges on. That is wasteful. Halving fixes the **allocation** by treating "which root move do I play" as a best-arm identification problem, using sequential halving (the root allocation from Gumbel MuZero).

Per turn:

1. **Arms.** The top `search_top_m` moves by prior, plus any capture, check, or promotion outside that set.
2. **Rounds.** The value budget is split across rounds. After each round the worst-scoring half of the arms is eliminated, and their unspent budget flows to the survivors. Obvious losers die after a handful of evaluations. The final two candidates get deep trees.
3. **Tree growth by plausibility.** Each surviving arm owns a priority queue of positions to expand, ordered by the cumulative policy log-prob of the line (both sides). Forced replies (captures, checks, promotions) inherit their parent's priority, so a refutation competes at the plausibility of the line it refutes instead of getting pruned. Forced lines go deep, quiet wide positions stay shallow.
4. **Scoring.** Negamax backup over the realized tree (terminals scored exactly, frontier leaves on their value estimate), plus the `lambda * log_prob` root-move term.

The rule that keeps this honest: **value never chooses what to expand.** The queue is ordered by the prior alone. Value enters only at backup and at arm comparison. If you let value pick what to expand, you keep exactly the lines where the opponent conveniently blunders, which is the `lambda = 0` failure again in tree form.

{{< mermaid >}}
flowchart TD
    A[Root position] --> B[Arms: top-m by prior + forcing moves]
    B --> C[Round: split budget across surviving arms]
    C --> D[Grow each arm's tree by prior-ordered beam]
    D --> E[Negamax backup, score each arm]
    E --> F[Eliminate worst half, budget flows to survivors]
    F --> G{More than one arm?}
    G -- yes --> C
    G -- no --> H[Play surviving root move]
{{< /mermaid >}}

### Prefix-Cache Decode (what made bigger budgets affordable)

Search is only useful if you can afford enough of it. The once-per-turn root forward pass doubles as a prefill, and its per-layer K/V become a shared cache. Each searched position is then a single new token attending to that cache, which is O(1) new work per evaluation instead of re-encoding the whole game history. That is roughly 3.7x faster and is what made the 1024 and 2048 budgets practical.

---

## The Strength Journey: From Losing to SF1400 to Around 2250

Here is the arc that matters, each row measured over 100 games, colors alternating, Stockfish at 0.05s per move with its Elo cap set per rung. Score is `(wins + 0.5 * draws) / games`, with about `±0.05` standard error at 100 games. The eval script does not emit calibrated Elo yet, so the Elo numbers below are derived from score rate against a known opponent, not measured directly.

| Opponent | Move selection | Model | W / D / L | Score |
|---|---|---|---|---|
| SF1400 | greedy | v3 | 7 / 28 / 65 | 0.21 |
| SF1400 | value_search_d2 | v3 | 22 / 16 / 47 | 0.34 |
| SF1400 | halving 256/d4 | v3 | 88 / 7 / 5 | **0.915** |
| SF1800 | halving 1024/d6 | v4 | 56 / 19 / 25 | 0.655 |
| SF2000 | halving 1024/d6 | v4 | 41 / 22 / 37 | 0.520 |
| SF2200 | halving 2048/d8 | v4 (Elo-weighted value loss) | 44 / 31 / 25 | **0.595** |

What each step actually bought, in order:

1. **Greedy** set the imitation floor: `0.21` vs SF1400, which is roughly 1170-Elo play. The policy loses to a 1400.
2. **Value rerank** established two facts I kept relying on: value-dominant scoring beats policy-dominant, and `lambda = 0` collapses.
3. **Value search d2** added adversarial lookahead with forcing-move refutations, `+0.13` over greedy.
4. **Halving** replaced uniform allocation with sequential halving and a prior-ordered tree. On SF1400 that took `0.34` to `0.915`, roughly `+330` Elo, and saturated the rung.
5. **Prefix-cache decode** made the bigger budgets affordable, which opened up the 1800 and 2000 rungs.
6. **The v4 trunk** (768-dim, 8 layers, higher value weight) lifted the value oracle itself. At a matched search budget it took SF1800 from `0.465` to `0.600`, and it revived a budget curve that had flattened under `v3`. This is the part I want to flag for other applied folks: search compute only converts into strength as well as your value estimate lets it. A better oracle is what unlocked more search paying off.
7. **Elo-weighting the value loss** was the biggest single lever at SF2200. Resuming training with it took the pure-head system from `0.510` to `0.595` at 2048/d8. A `0.595` score against a 2200 opponent is worth about `+65` Elo, which is where the "around 2250" claim comes from.

One dead end worth recording. I tried blending in a separate Stockfish-distilled value net as a second opinion. It helped at the 1800 and 2000 rungs, where the trunk's own value head was still weak. But as the head improved (v4, then Elo-weighted value loss) the external net stopped earning its place and eventually became a small drag. I removed it. When I audited the code I also found it had quietly never been wired into the shipped config for a stretch, which is its own lesson about checking that your feature is actually on before trusting its ablation.

All of this is one-or-two-datapoint territory at 100 games each. Treat it as directions that held up, not laws.

---

## The Current Bet: Distilling Search Back Into the Policy

The system above pays for search at inference, every move, forever. The obvious question is whether I can push the search's improvement back into the policy itself, so the raw policy gets stronger and I need less search (or none) at play time. This is expert iteration, and it is where I am spending time now.

### First attempt distilled search into the value head, and it did not work

The first idea was to take the search's backed-up value at each position and blend it into the value-head target against the real game outcome. I ran it at production scale and it did not beat the pure-outcome baseline at any blend weight. Held-out value loss got consistently worse the more I leaned on the searched target.

Reading around fixed why. Every example I could find where search-to-**value** distillation actually works (Gumbel MuZero, EfficientZero V2, the older Meep chess program) runs inside a **continual self-play loop**, and where the motivation is stated it is about correcting staleness in a replay buffer. My setup had neither property. I generated rollouts once, from one frozen checkpoint, against a static set of human games, and only labeled the root of each sampled position. The mechanism that makes this pay off elsewhere was mostly absent from how I ran it.

### The better-grounded idea: distill search into the policy

Gumbel MuZero (Danihelka et al., ICLR 2022) has a result that the value approach does not: distilling the search's improved root distribution into the **policy** gives a single-round improvement guarantee that holds even from a frozen checkpoint. That is exactly my situation. So Phase 1b distills `value_search_halving`'s root-arm outcome into the policy head, reusing the same rollout data.

The mechanism, in applied terms:

- For each searched position, build an improved target over moves. Searched arms get scored by their backed value `q̂`. Every other move keeps whatever probability mass the current policy already gives it. The paper calls this `completedQ`.
- Turn that into a target distribution, `π' = softmax(logits + alpha * completedQ)`, and pull the policy toward it with a KL term, mixed with the usual human-move cross-entropy.
- `alpha` is a single fixed constant, swept on a small grid. I deliberately did not make it visit-adaptive. I tried that (a `c_visit` scheme borrowed from the same paper) and it blew up: a value tuned at a small search budget meant something completely different at the production budget and collapsed search quality to `0.10`. Same Goodhart failure as `lambda = 0`, reached from a different direction. The simplest version the guarantee actually supports is the one to build.

The root sampling matters here too. The rollouts sample root moves with a Gumbel-top-k trick, which is an unbiased sample from the policy rather than a deterministic top-k. Deterministic top-k can miss the only good move and score worse than the raw prior, which the paper shows with a clean counterexample. So the rollouts are genuinely on-policy samples, which is what the improvement guarantee needs.

### Why I call it on-policy self-distillation

There is no external teacher. The "teacher" is the same model's own search built from its policy and value heads. The rollouts come from the model's own on-policy sampling. The improved target gets distilled back into the same model's policy head. That is self-distillation, and because the rollouts are generated by the policy being trained, it is on-policy distillation.

That framing comes with a known hazard, which the recent "Lightning OPD" work (Wu, Han and Cai, NVIDIA) makes precise: on-policy distillation needs **teacher consistency**. The model that generated the training targets and the model being trained should be the same, or a gradient bias creeps in that grows as the student drifts away from the checkpoint that produced the rollouts. Their exact bound is for a different setup (trajectory-level, advantage-weighted distillation over the model's own autoregressive rollouts), and mine is a per-position soft-label KL over human game positions, so the precise theorem does not transfer. But the practical concern does, by analogy. A rollout's backed value reflects whichever checkpoint's search produced it, and training against it while drifting away from that checkpoint is the same staleness axis that sank the value-distillation attempt. So Phase 1b scores the target against the **live** current model rather than the frozen rollout, to track that drift instead of pretending it is not there, with a checkpoint-consistency guard to keep me honest.

### Status: designed, being built, blocked on throughput

This is not a result yet. The loss is wired in and tested. The thing in the way is boring and real: generating enough search rollouts to cover the KL target over 10,000 to 20,000 training steps is slow. I tried scaling to a rented 5090 and it did not help the single-shard rate. Profiling pointed at a slow legal-move / `gives_check` path as the actual bottleneck, and a faster version is designed but not built. So the honest status is: the idea is grounded, the plumbing exists, and I am currently fighting the data-generation cost before I can say whether it works.

---

## What I Got Wrong (a running list)

- **Trusting offline metrics.** `hr@10 > 0.9` and a policy that loses to SF1400. Good imitation is not good chess.
- **`lambda = 0` in value scoring.** Drop the policy prior and the search happily optimizes value-head noise. The prior is load-bearing.
- **A hyperparameter screen at the wrong budget.** The `c_visit` idea looked good at budget 256 and collapsed at 2048, because its effect scales with the budget. A cheap screen is not a substitute for testing at the deployment budget, even when the scaling is mathematically obvious in hindsight.
- **Distilling search into the value head, offline and one-shot.** The literature that makes it work runs a continual loop. I did not, and it did not.
- **Shipping an ablation for a feature that was not actually on.** The external value net sat in configs as a silent no-op for a stretch. Check that the thing is wired in before you trust its numbers.

---

## Current Limitations

- Training is single-process (no DDP launcher yet).
- No legal-move masking in the training loss. The policy is trained as full-vocab classification and projected to legal moves only at play time.
- Prefix K/V caching is per-turn only; the cache is rebuilt each move and games are played one at a time (no cross-game batching).
- Value labels are still raw game outcomes, which are noisy. Progress weighting and Elo weighting help but do not fully fix it.
- The policy-distillation loop is throughput-bound on rollout generation, as above.

---

## Planned Next Steps

- Land the faster `gives_check` path and actually run the Phase 1b policy-distillation sweep to a conclusion.
- If policy distillation improves the raw policy, revisit value distillation inside the resulting loop, where the staleness argument finally applies.
- Slice metrics by game phase and Elo. Global averages hide regressions.
- Scale the trunk past `v4` and see whether Elo keeps climbing smoothly, and whether search helps more or less as the policy gets stronger.
- Only then, RL. Self-play with a KL-regularized PPO objective is still the likely path to the next jump, but I want to exhaust the cheaper imitation-plus-search-plus-distillation route first.

---

## References

- [imba-chess on GitHub](https://github.com/viig99/imba-chess)
- [Lichess Open Database](https://database.lichess.org/)
- [python-chess](https://python-chess.readthedocs.io/)
- [Searchless Chess (DeepMind)](https://github.com/google-deepmind/searchless_chess)
- [grpo_chess](https://github.com/noamdwc/grpo_chess)
- Danihelka et al., "Policy improvement by planning with Gumbel" (Gumbel MuZero), ICLR 2022
- Spigler, "Proximal Policy Distillation", 2024
- Wu, Han and Cai, "Lightning OPD" (on-policy distillation, teacher consistency), NVIDIA
- "Transcendence: Generative Models Can Outperform The Experts That Train Them"
- "Amortized Planning with Large-Scale Transformers: A Case Study on Chess"

## Related Posts

- [FlexAttention HSTU at 500M Events](./hstu-for-yambda.md)
- [Machine Learning Engineer Roadmap](./ml_engineer_guidelines.md)

## Closing

The bet has not changed: structured event modeling from recsys transfers to chess. What I have learned since the last update is where the strength actually comes from. Imitation gives a good prior and bad chess. A value head plus a search that allocates its compute well (sequential halving, prior-ordered trees, value never picking what to expand) is what turned that prior into roughly 2250-Elo play, with no RL yet. The open question now is whether I can distill that search back into the policy so it stops being a tax I pay every move. That part is still being built.
