+++
title = "Building a Chess Bot with HSTU: From Lichess Pretraining to Value Search"
description = "A learning journal on training a compact chess policy model using HSTU-style next-token prediction on Lichess games, adding a value head for WDL prediction, and using policy-pruned minimax search to improve playing strength."
tags = [
  "chess",
  "HSTU",
  "reinforcement learning",
  "next-token prediction",
  "value head",
  "minimax search",
  "Lichess",
  "Stockfish",
  "sequential modeling",
  "large-scale ML"
]
date = "2026-02-27"
categories = [
  "Machine Learning",
  "Chess",
  "Systems"
]
menu = "main"
draft = false
+++

## Build Log Snapshot

This post is my running build log for `imba-chess`.

Right now, the checked-in model is about **12.6M parameters**.  
My original target in the notes was 20–30M, but I am still iterating toward that.

The project is intentionally built to run on constrained hardware.  
Most of this training/dev loop has been on my **8GB RTX 3070 laptop**.

What I have done so far:

- Adapted HSTU-style causal sequence modeling from recommender systems to chess.
- Trained next-move prediction (policy head) on high-Elo Lichess game data.
- Evaluated offline metrics: top-1, hr@10, MRR.
- Tested against Stockfish. Results: humbling.
- Added a value head (WDL), wired it into move selection.
- Implemented depth-2 policy-pruned minimax search on top.
- Defined a Stockfish ladder evaluation pipeline for reproducible strength measurement.

My core framing is: **a chess game is a structured event sequence**.

---

## What Triggered This Project

Before this, I finished an STU + FlexAttention rewrite and tested it on:

- Zvuk-200M
- Yambda-500M

That work beat strong SASRec baselines and showed clean scaling.  
After a point, additional recsys gains felt marginal, so I wanted a harder sequential reasoning problem.

Chess was the natural next step:

- two-player sequential decisions
- long horizon credit assignment
- explicit win/draw/loss outcomes
- cheap large-scale supervised data from Lichess

So this became my side learning project: can the same sequence toolkit transfer from recommendation to strategy?

---

## Why Chess and Why HSTU

Most strong chess systems lean heavily on self-play + search. That works, but it is expensive.  
I started with a cheaper path: imitate strong human games first, then add value/search.

HSTU was built for jagged structured sequences in recsys.  
Chess has the same shape: variable-length event sequences, structured state, known next-action target.

Simple bet: **if it can predict next-item well, it can predict next-move well**.

---

## Original Project Brief (Early Notes)

My initial brief was:

- Use the full Lichess stream (~2.3TB scale over many years).
- Encode board state + move history + metadata.
- Add player embeddings (QR), Elo buckets, time-control features, and clock signals.
- Pretrain on next legal UCI move prediction.
- Move into RL post-training (PPO/GRPO), self-play via pufferlib, and/or engine matches.
- Evaluate against Stockfish/Leela by win-rate and Elo trends.
- Keep the whole system practical on a single 4090 in roughly 20–30M parameters.

That direction still holds, but the implementation has changed based on actual results.

---

## Data Pipeline

### Source

I use the [Lichess open game database](https://database.lichess.org/) via `Lichess/standard-chess-games` on Hugging Face.
The hive `year/month` partitioning makes temporal splits straightforward.

Main filter: average Elo `(WhiteElo + BlackElo) / 2 >= 2000`.

### Temporal Splits

I use chronological splits (not random) to avoid future leakage.

| Split | Window |
|---|---|
| train | 2021-01 through 2025-06 |
| val | 2025-07 (single month) |
| test | 2025-08 through 2025-09 |

No model is selected using the test split.

### Historical Run Profile (From Earlier Training Notes)

One earlier run (from notes) used:

- Train window: `2018-01 -> 2025-06`
- Train filter: average Elo >= 2000
- Test filter: average Elo >= 2400 on `2025-08 -> 2025-09`

From those notes:

- estimated high-Elo train pool around 422M games
- observed ingest/training throughput around ~1.5M games/hour in that run
- average game length observed around 70-80 moves

These are run-log observations, not the current default config.

### PGN Parsing and Board State

Each game is replayed with `python-chess`. At each ply, the board is converted to a structured token representation — not a FEN string, not a text sequence.

**Board state per ply:**

| Field | Values | Notes |
|---|---|---|
| `piece_ids` | `[64]`, 0–12 | 0=empty, 1–6=white, 7–12=black |
| `turn_id` | 0/1 | side to move |
| `castle_id` | 0–15 | KQkq bitmask |
| `ep_file_id` | 0–8 | en-passant file + 1, 0=none |
| `halfmove_bucket_id` | ≥0 | bucketed clock |
| `fullmove_bucket_id` | ≥0 | bucketed move number |

Targets are UCI move IDs from a static vocabulary of all legal UCI moves (from→to + promotions).

### Incremental Board Updates (Benchmarked, Not Yet Main Path)

I benchmarked the `board.piece_map()` rebuild path and explored incremental `bytearray(64)` updates.

In theory, a chess move touches at most 4 squares (castling: king + rook):

```text
Normal move:    2 squares
Capture:        2 squares
En passant:     3 squares
Castling:       4 squares
Promotion:      2 squares
```

The estimated improvement in notes was ~14–16 µs/ply down to ~2–5 µs/ply.  
But to be clear: the **current main encoder still rebuilds from `board.piece_map()`**. Incremental updates are planned, not merged.

### Jagged Batching

Multiple games are packed into a single flat token buffer, with `seq_offsets` marking boundaries. This is the same trick from [FlexAttention HSTU at 500M Events](./hstu-for-yambda.md): no cross-game attention, no padding waste.

```text
[BOS | ply1 ply2 ... plyN | BOS | ply1 ... plyM | ...]
       game 1                    game 2
```

Batch shape: `[total_tokens]` for most fields, `[total_tokens, 64]` for `piece_ids`.  
No per-token padding mask is materialized; a block mask is derived from `seq_offsets` at runtime.

---

## Model Architecture

### Overview

{{< mermaid >}}
flowchart LR
    A[piece_ids 64 tokens] --> B[E_piece + E_square]
    B --> C[Mean pool → board_emb]
    D[prev_move_id] --> E[E_move]
    F[turn/castle/ep/clk] --> G[E_meta]
    C --> H[Additive event composition]
    E --> H
    G --> H
    H --> I[HSTU Backbone causal layers]
    I --> J[Policy Head → move logits]
    I --> K[Value Head → loss/draw/win]
{{< /mermaid >}}

### Embedding Layers

Each ply is converted to a single event vector by embedding structured fields and **summing** them (current implementation):

```text
event_t = board_emb
        + seq_token_emb
        + turn_emb
        + castle_emb
        + ep_emb
        + halfmove_emb
        + fullmove_emb
        + prev_move_emb
```

Where:
- `board_emb` = mean over (E_piece(piece_ids) + E_square(index)) for all 64 squares
- `move_emb_{t-1}` = embedding of the previous move (or START token at ply 1)
- `meta_emb` = embeddings of turn, castling rights, en-passant file, clock buckets

### HSTU Backbone

The backbone is causal with relative position bias over plies.
Current footprint is ~12.6M params (with value head), and it trains fine on a single RTX 4090.

---

## Training Phase 1: Supervised Policy Pretraining

### Objective

Cross-entropy on next UCI move over the full static move vocabulary:

```text
loss = CE(logits, target_move_id)
```

BOS positions are excluded via `ignore_index = -100`.

### Elo-Weighted Loss

Not all supervision is equally useful, so I weight by played-by Elo:

```text
norm_i = clamp((elo_i - min_elo) / (max_elo - min_elo), 0, 1)
w_i    = 1 + strength × (norm_i ^ alpha)
loss   = Σ(w_i × ce_i) / Σ(w_i)
```

Weight normalization keeps gradients stable.  
I also use label smoothing because strong positions often have multiple reasonable moves.

### Training Infrastructure

- Optimizer: StableAdamW with OneCycleLR scheduler
- Precision: bfloat16 mixed precision
- Checkpointing: best by `hr@10` on full val, plus periodic last checkpoints
- Logging: TensorBoard + periodic fast val/test checks

---

## Evaluation Metrics

### Offline Metrics (Phase 1)

Evaluated on held-out val/test splits every N steps:

| Metric | What it measures |
|---|---|
| `loss_ce` | Cross-entropy on target move |
| `ppl` | Perplexity (exp of loss_ce) |
| `top1_acc` | Argmax move matches human move |
| `top3_acc` / `top5_acc` | Move in top-3/5 |
| `hr@10` | Hit rate at 10 (top-10 accuracy) |
| `mrr` | Mean reciprocal rank of ground-truth move |

Model selection uses `hr@10` from full val as the primary signal.

### Slice Metrics (Planned Expansion)

Global averages hide regressions.  
Phase/Elo slice reporting is in the eval spec, but current evaluator outputs are mostly global (`loss_ce`, `ppl`, `top-k`, `mrr`).

### Engine Evaluation (Phase 2)

After offline metrics stabilize, the model plays against Stockfish:

- Alternating colors, fixed time controls
- Current default ladder config: `1320, 1600, 1800, 2000` (plus optional full-strength segment)
- Reports currently include wins/draws/losses, score rate, color split, legal-move coverage, and run config
- Elo estimate + confidence intervals are part of the evaluation plan, but not yet emitted by the script

### Current Snapshot from Stored Artifacts

From current `artifacts/checkpoints/tb` and `artifacts/eval`:

| Area | Result |
|---|---|
| Offline full-val `hr@10` | improved to **0.9208** |
| Offline full-val `top1_acc` | **0.4341** |
| Offline full-val `mrr` | **0.6029** |
| Stockfish ladder @1320 (sample policy) | `2/22/76`, score `0.13` |
| Stockfish ladder @1600 (sample policy) | `0/5/95`, score `0.025` |
| Stockfish ladder @1800 (sample policy) | `0/10/90`, score `0.05` |
| Stockfish ladder @2000 (sample policy) | `1/7/92`, score `0.045` |
| Stockfish full strength (sample policy) | `0/0/100`, score `0.00` |

### Early Milestone Notes (Before Full Value/Search Integration)

From an earlier checkpointing phase:

- `hr@10 = 0.830764` was reached while training was still early.
- Later policy-only runs crossed `hr@10 > 0.9` and `top1 ~ 0.45`.
- Despite that, initial engine strength was poor: policy-only behavior could still underperform badly versus low-Elo Stockfish settings.

This was the key project inflection point for me: good imitation metrics were necessary, but not enough for winning play.

---

## Phase 2: Adding a Value Head

### Why a Value Head

A policy head mainly learns "what move humans pick".  
It does not directly optimize for outcome quality.  
The value head adds explicit WDL outcome prediction from the position.

Without value at inference time, the model cannot cleanly separate:
- "This move is popular in human games" (policy says yes)
- "This move leads to a winning position" (requires value)

### WDL Classification

The value head is a 3-class classifier from the side-to-move perspective:

```text
value_logits = Linear(d, 3)  # [loss, draw, win]
```

Labels are derived from the per-game result (`game_result_white ∈ {+1, 0, -1}`) and per-token `turn_id` (to flip perspective for black).

A scalar value is extracted as:

```text
V(s) = p(win) - p(loss) ∈ [-1, 1]
```

### Progress Weighting

Value labels derived from final game results are noisy — early positions have a weak causal link to who ultimately wins. We downweight early plies and emphasize later ones:

```text
progress_weight = (ply_index / total_plies) ^ alpha
```

Current config uses `value_weight_alpha = 0.9` (mild late-game emphasis). I still treat this as a tuning knob.

### Combined Loss

```text
total_loss = policy_loss + λ × value_loss
```

Early notes suggested starting near `λ = 0.15` for safety.  
Current checked-in config uses `value_loss_weight = 0.5`, and I am still tuning this.

{{< mermaid >}}
flowchart TD
    A[HSTU hidden state] --> B[Policy Head]
    A --> C[Value Head]
    B --> D[CE loss on next move]
    C --> E[CE loss on WDL outcome]
    D --> F[total_loss = policy_loss + λ × value_loss]
    E --> F
{{< /mermaid >}}

### Training Schedule

1. **Warm start** (optional): freeze backbone for 1k–3k steps, train only heads.
2. **Joint training**: unfreeze all, keep `value_loss_weight` low initially.
3. **Monitor**: if policy metrics drop, reduce value weight.

---

## Phase 3: Using Value at Inference

Adding a value head to training only modestly improves playing strength. The real gain comes from using the value during **move selection**.

### Mode 1: Greedy (Baseline)

Pick the highest-logit legal move. Fast, deterministic, no value used.

### Mode 2: Sampled Decoding

Sample from top-k / top-p legal moves with temperature. Adds variety, occasionally finds surprising moves, but can also pick blunders.

### Mode 3: Value Rerank (1-Ply Lookahead)

Take top-K policy candidates, evaluate each resulting position with the value head, pick the best:

```text
score(move) = log π(move | s) - λ × V(next_state)
```

The minus sign: after we move, it is the opponent's turn at `next_state`, so high opponent value is bad for us.

{{< mermaid >}}
flowchart LR
    A[Current state s] --> B[Policy: top-K legal moves]
    B --> C[Apply each move → s']
    C --> D[Value head on each s']
    D --> E[Score = log π - λ V_opp]
    E --> F[Pick best scoring move]
{{< /mermaid >}}

Default settings: `K = 8`, `λ = 0.35`.

### Mode 4: Depth-2 Policy-Pruned Minimax

One step deeper: after my move, simulate opponent reply, then choose move with best worst-case reply value.

```text
Q(a) = min_{b ∈ top-K} V(apply(apply(s, a), b))
a*   = argmax_a Q(a)
```

The branching factor is controlled by keeping only top-K policy candidates at each ply: K1 candidate moves for us, K2 opponent responses each.

{{< mermaid >}}
flowchart TD
    A[Root state s] --> B[Our top-K1 moves]
    B --> C[For each candidate a → s']
    C --> D[Opponent top-K2 moves]
    D --> E[For each response b → s'']
    E --> F[Value V at s'']
    F --> G[Opponent picks b that minimizes V for us]
    G --> H[We pick a with best worst-case V]
{{< /mermaid >}}

Why this helped me early: many losses were immediate tactical misses.  
Depth-2 explicitly checks "what is their best next reply?"

**Batch optimization**: instead of calling the transformer node-by-node, batch all K1 × K2 grandchild states into a single forward pass. This can be 10–100× faster on GPU.

---

## What Went Wrong (And Why)

This was the most useful learning phase so far.

A policy model can look strong offline (`hr@10`, `top1`) and still collapse in actual play:

- imitation objective learns "what humans played", not "what maximizes winning chances"
- greedy decoding can overcommit to narrow policy modes
- sampling can recover occasional wins, but not stable strength
- win/loss signal is weakly coupled to policy CE unless value/search is explicitly used

In short: pretraining gave me a good prior, but not reliable tactical behavior by itself.

---

## Ablation Matrix

To measure what actually moves the needle on Stockfish win rate, this is the comparison matrix to run under identical settings:

| Configuration | Description |
|---|---|
| Policy-only + greedy | Baseline |
| Policy+value training, greedy decode | Does value training help representations? |
| Policy+value training, value-rerank | Does 1-ply value improve play? |
| Policy+value training, depth-2 search | Does minimax help further? |

All comparisons use the same Stockfish time controls and opening protocols.

---

## Current Limitations and Known Issues

- No legal-move masking in the training loss yet. Policy is trained as full-vocab classification, then projected to legal moves during play/eval.
- Training is single-process (no DDP launcher yet).
- `value_rerank` is one-ply only; `value_search_d2` is depth-2 and substantially slower than greedy.
- Value labels are noisy for early plies — progress weighting helps but does not fully solve this.
- Value head may learn player-strength bias (higher Elo games have more draws): we still need stronger value-slice/calibration diagnostics in the evaluator.

---

## Planned Next Steps

**Self-play RL (Phase 4)**

After pretraining and value metrics stabilize, next step is RL fine-tuning via self-play:

- Environment: `gym-chess` with parallel rollouts (~1000 workers via pufferlib)
- Algorithm: PPO or KL-regularized PPO (GRPO-style)
- Reward: +1 win, 0 draw, −1 loss; optional shaping from engine eval delta
- League: self-play against current + past checkpoints; optional Stockfish/Leela matches

**Beam Search**

Another path is beam search over likely continuations, with policy priors and value-scored leaves.

**Scaling**

Current production-ish config is ~12.6M parameters on a single RTX 4090. Interesting questions:
- How far can this config go before scaling width/depth?
- If we scale toward the original 20–30M target, does Elo improve smoothly?
- Does Elo scale smoothly with model size?
- Does value search help more at smaller model sizes (where policy alone is weaker)?

---

## References

- [Lichess Open Database](https://database.lichess.org/)
- [python-chess](https://python-chess.readthedocs.io/)
- [Searchless Chess (DeepMind)](https://github.com/google-deepmind/searchless_chess)
- [grpo_chess](https://github.com/noamdwc/grpo_chess)
- [PyTorch FlexAttention](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)
- [Mermaid](https://mermaid.js.org/)
- Transcendence: Generative Models Can Outperform The Experts That Train Them
- Amortized Planning with Large-Scale Transformers: A Case Study on Chess

## Related Posts

- [FlexAttention HSTU at 500M Events](./hstu-for-yambda.md)
- [Machine Learning Engineer Roadmap](./ml_engineer_guidelines.md)

## Closing

My working bet is still the same: structured event modeling from recsys transfers to chess.

Value + search is the bridge from imitation to actual board strength.

Whether that is enough without heavy RL is still open.

The current working hypothesis is:

- better pretraining gives a better prior,
- value + shallow search converts more of that prior into practical strength,
- and RL is likely needed to unlock the next jump in actual board-level reasoning.
