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

## Executive Summary

This is a running journal of `imba-chess`, a personal research project to build a compact (20–30M parameter) chess bot entirely from supervised pretraining on Lichess games, without any game-tree search engine to begin with.

The journey so far:

- Adapted HSTU-style causal sequence modeling from recommender systems to chess.
- Trained next-move prediction (policy head) on high-Elo Lichess game data.
- Evaluated offline metrics: top-1, hr@10, MRR.
- Tested against Stockfish. Results: humbling.
- Added a value head (WDL), wired it into move selection.
- Implemented depth-2 policy-pruned minimax search on top.
- Defined a Stockfish ladder evaluation pipeline for reproducible strength measurement.

The throughline is: **treat each chess ply as a structured event in a sequence, the same way you would model user behavior**.

---

## Why Chess and Why HSTU

Most chess neural nets (Leela, AlphaZero) are trained purely by self-play, using MCTS to generate experience. That is powerful but expensive. A cheaper first step is imitation: train on human games, learn the distribution of plausible moves, and then layer search on top.

HSTU (Hierarchical Sequential Transduction Unit) was designed for large-scale sequential recommendation — jagged user histories, variable event types, structured side features. Chess plylines have almost the same structure: a variable-length sequence of structured events (board states + moves), with a known target (next move) at each step.

The idea was: **if HSTU can predict next song plays, it can predict next chess moves**.

---

## Data Pipeline

### Source

All data comes from [Lichess open game database](https://database.lichess.org/), accessed via the `Lichess/standard-chess-games` Hugging Face dataset. It is hive-partitioned by `year/month`, which makes temporal splits clean.

Filter: average Elo of `(WhiteElo + BlackElo) / 2 >= 2000`. High-Elo games have lower noise and better move quality.

### Temporal Splits

Splits are chronological, not random. This prevents future-move leakage and gives a more realistic out-of-sample test.

| Split | Window |
|---|---|
| train | 2018-01 through 2025-07 |
| val | 2025-08 (single month) |
| test | 2025-09 (single month) |

No model is selected using the test split.

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

### Incremental Board Updates

Rebuilding `piece_ids` from `board.piece_map()` every ply takes ~14–16 µs per ply. The optimization: update only the changed squares.

A chess move touches at most 4 squares (castling: king + rook). So we maintain a `bytearray(64)` and apply incremental updates:

```text
Normal move:    2 squares
Capture:        2 squares
En passant:     3 squares
Castling:       4 squares
Promotion:      2 squares
```

This drops board encoding from ~16 µs to ~2–5 µs per ply. At Lichess dataset scale (hundreds of millions of plies) this is a meaningful saving.

### Jagged Batching

Multiple games are packed into a single flat token buffer, with `seq_offsets` marking boundaries. This is the same trick from [FlexAttention HSTU at 500M Events](./hstu-for-yambda.md): no cross-game attention, no padding waste.

```text
[BOS | ply1 ply2 ... plyN | BOS | ply1 ... plyM | ...]
       game 1                    game 2
```

Batch shape: `[total_tokens]` for most fields, `[total_tokens, 64]` for `piece_ids`. No attention mask needed when `seq_offsets` are passed directly to FlexAttention.

---

## Model Architecture

### Overview

{{< mermaid >}}
flowchart LR
    A[piece_ids 64 tokens] --> B[E_piece + E_square]
    B --> C[Mean pool → board_emb]
    D[prev_move_id] --> E[E_move]
    F[turn/castle/ep/clk] --> G[E_meta]
    C --> H[Concat + Project → event_t]
    E --> H
    G --> H
    H --> I[HSTU Backbone causal N layers]
    I --> J[Policy Head → move logits]
    I --> K[Value Head → loss/draw/win]
{{< /mermaid >}}

### Embedding Layers

Each ply is converted to a single event vector by embedding structured fields and concatenating them:

```text
event_t = LN(W · concat(board_emb, move_emb_{t-1}, meta_emb))
```

Where:
- `board_emb` = mean over (E_piece(piece_ids) + E_square(index)) for all 64 squares
- `move_emb_{t-1}` = embedding of the previous move (or START token at ply 1)
- `meta_emb` = embeddings of turn, castling rights, en-passant file, clock buckets

### HSTU Backbone

The backbone is a causal transformer with relative positional bias indexed by ply number. It runs over the event sequence `e_1 … e_T` and produces hidden states per ply.

Target model size: 20–30M parameters. This fits in a single RTX 4090 training run.

---

## Training Phase 1: Supervised Policy Pretraining

### Objective

Cross-entropy on next legal UCI move:

```text
loss = CE(logits_masked, target_move_id)
```

BOS positions are excluded via `ignore_index = -100`.

### Elo-Weighted Loss

Not all moves are equally informative. Moves by 2000 Elo players carry less signal than moves by 2600 Elo players. We apply per-token Elo weighting:

```text
norm_i = clamp((elo_i - min_elo) / (max_elo - min_elo), 0, 1)
w_i    = 1 + strength × (norm_i ^ alpha)
loss   = Σ(w_i × ce_i) / Σ(w_i)
```

Weight normalization keeps gradient scale stable when weighting is on.

Label smoothing is also applied to account for move non-uniqueness: strong positions often have multiple valid moves.

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

### Slice Metrics

Global averages hide regressions. We also report by:

- **Game phase**: opening (ply 1–20), middlegame (ply 21–60), endgame (ply 61+)
- **Elo bucket**: 2000–2199, 2200–2399, 2400+

### Engine Evaluation (Phase 2)

After offline metrics stabilize, the model plays against Stockfish:

- Alternating colors, fixed time controls
- Ladder evaluation across Elo settings: 1600, 1800, 2000, 2200, 2400, 2600, 2800
- Reports: wins/draws/losses, score rate, color split, Elo estimate with confidence intervals

---

## Phase 2: Adding a Value Head

### Why a Value Head

A policy head alone picks moves based on how likely a human would play them. It does not reason about **outcomes**. The value head adds a separate prediction: given the current board position, what is the probability of winning, drawing, or losing?

Without value at inference time, the model cannot distinguish between:
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

With `alpha = 1.5`, the first few plies contribute little; the endgame contributes most.

### Combined Loss

```text
total_loss = policy_loss + λ × value_loss
```

We start with `λ = 0.15` to protect policy quality while the value head bootstraps. If `top1/hr@10` drops sharply, reduce `λ` further.

{{< mermaid >}}
flowchart TD
    A[HSTU hidden state] --> B[Policy Head]
    A --> C[Value Head]
    B --> D[CE loss on next move]
    C --> E[CE loss on WDL outcome]
    D --> F[total_loss = policy_loss + 0.15 × value_loss]
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

Go one level deeper: after our move, simulate the opponent's best response, then pick our move that leads to the best position after that response. This is one-step minimax.

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

Why this matters more than RL early on: most amateur-level losses come from hanging pieces, missing forks, and stepping into mate threats. Depth-2 catches a large fraction of these because it explicitly asks "what is my opponent's best immediate reply?"

**Batch optimization**: instead of calling the transformer node-by-node, batch all K1 × K2 grandchild states into a single forward pass. This can be 10–100× faster on GPU.

---

## Ablation Matrix

To measure what actually moves the needle on Stockfish win rate, we run a controlled comparison:

| Configuration | Description |
|---|---|
| Policy-only + greedy | Baseline |
| Policy+value training, greedy decode | Does value training help representations? |
| Policy+value training, value-rerank | Does 1-ply value improve play? |
| Policy+value training, depth-2 search | Does minimax help further? |

All comparisons use the same Stockfish time controls and opening protocols.

---

## Current Limitations and Known Issues

- No legal-move masking in the prediction head yet. Full-vocab classification. The model can in principle output illegal move IDs (tracked separately as `legal_top1`).
- Training is single-process (no DDP launcher yet).
- `value_rerank` is one-ply only; `value_search_d2` is depth-2 and substantially slower than greedy.
- Value labels are noisy for early plies — progress weighting helps but does not fully solve this.
- Value head may learn player-strength bias (higher Elo games have more draws): tracked by Elo-diff slices during eval.

---

## Planned Next Steps

**Self-play RL (Phase 4)**

After pretraining stabilizes and value metrics look calibrated, the next stage is RL fine-tuning via self-play:

- Environment: `gym-chess` with parallel rollouts (~1000 workers via pufferlib)
- Algorithm: PPO or KL-regularized PPO (GRPO-style)
- Reward: +1 win, 0 draw, −1 loss; optional shaping from engine eval delta
- League: self-play against current + past checkpoints; optional Stockfish/Leela matches

**Beam Search**

Another direction: instead of depth-2 minimax, run beam search over likely continuations. Policy priors guide the beam; value head scores leaf nodes. More compute, potentially better tactical vision.

**Scaling**

Current target is 20–30M parameters on a single RTX 4090. Interesting questions:
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

## Related Posts

- [FlexAttention HSTU at 500M Events](./hstu-for-yambda.md)
- [Machine Learning Engineer Roadmap](./ml_engineer_guidelines.md)

## Closing

The bet here is that structured event modeling — the same pattern that works for sequential recommendation — transfers cleanly to chess. Board state is richer than a user's listening history, but the sequence modeling problem is the same: predict what comes next from what came before.

The value head and minimax search are the bridge from imitation to reasoning. Imitation learns the prior; search uses that prior to avoid mistakes.

Whether that's enough to reach a respectable Elo without full RL self-play is the open question.
