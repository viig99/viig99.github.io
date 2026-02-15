+++
title = "FlexAttention HSTU at 500M Events: RQ Tokens, QR Embeddings, and 1D Biases"
description = "How we scaled jagged sequential recommendation on Yambda-scale data with FlexAttention, residual quantization outputs, quotient-remainder embeddings, and on-the-fly 1D attention biases."
tags = [
  "recommender systems",
  "HSTU",
  "FlexAttention",
  "residual quantization",
  "quotient remainder embedding",
  "sequential recommendation",
  "Yambda",
  "large-scale ML"
]
date = "2026-02-15"
categories = [
  "Machine Learning",
  "Recommender Systems",
  "Systems"
]
menu = "main"
draft = false
+++

## Executive Summary

At Yambda scale (about 500M events and 9.4M items), models rarely break because one idea is bad. They break when many small, expensive defaults pile up.

This post is the story of the choices that kept an HSTU-style recommender trainable:

- Jagged, block-masked attention with [PyTorch FlexAttention](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)
- Residual quantization (RQ) token prediction instead of a giant item-ID softmax
- Quotient-remainder (QR) embeddings for large sparse categorical spaces
- On-the-fly 1D attention bias terms (time, duration, organic) instead of dense `[S, S]` bias tensors
- [ALiBi](https://arxiv.org/abs/2108.12409) positional bias instead of learned position embeddings

The throughline is simple: keep the inductive bias, cut the dense and quadratic costs.

## Why Yambda Forces Architectural Discipline

Four constraints shaped every design decision:

1. User histories are jagged, not fixed-length.
2. Item space is large enough that a full output projection is expensive.
3. Side metadata is sparse and partially missing.
4. Attention biasing must not materialize quadratic tensors.

At this scale, architecture is mostly cost control. So we optimized for memory and throughput first, then validated that quality still held.

## Architecture Overview

{{< mermaid >}}
flowchart LR
    A[RQ Codes 8x1024] --> B[Sum RQ Embeddings]
    C[QR Artist Embedding] --> D[Event Representation]
    E[Event Type Embedding] --> D
    B --> D
    D --> F[FlexAttention HSTU x N]
    F --> G[L2 Normalize]
    G --> H[Cascaded RQ Heads]
    H --> I[Logits S x 8 x 1024]
{{< /mermaid >}}

The pivotal move is at the output: we do not model a direct `item_id` distribution. We model codebooks.

## Decision 1: Replace Item-ID Softmax with RQ Outputs

With ~9.4M items, a direct head looks like:

```text
Linear(D -> 9,400,000)
```

That is expensive in parameters, optimizer state, and memory bandwidth. Instead, we train an 8-level residual quantizer over item embeddings and predict discrete code indices:

```text
8 x Linear(D -> 1024)
```

What this changes in practice:

- Smaller output parameterization and optimizer footprint
- Better fit for ANN retrieval over decoded embeddings
- Cleaner decomposition into coarse-to-fine prediction

Reference: [FAISS](https://github.com/facebookresearch/faiss) for vector retrieval.

## Decision 2: Keep Attention Priors, Drop O(S^2) Bias Tensors

A common failure mode is precomputing dense bias matrices for time, position, and feature priors. At longer sequence lengths, that burns memory for very little return.

In FlexAttention, we apply score modifiers lazily:

```python
def score_mod(score, b, h, q_idx, k_idx):
    score += alibi_bias(h, q_idx, k_idx)
    score += time_bias[time_bucket(q_idx, k_idx)]
    score += duration_bias[duration_bucket(k_idx)]
    score += organic_bias[is_organic(k_idx)]
    return score
```

The bias terms are 1D tables and scalar functions. Memory scales with bucket count, not pair count.

{{< mermaid >}}
flowchart TD
    A[Raw QK score] --> B[Add ALiBi]
    B --> C[Add time bucket bias]
    C --> D[Add duration bias]
    D --> E[Add organic bias]
    E --> F[Final attention score]
{{< /mermaid >}}

## Decision 3: Use ALiBi for Position Bias

Learned positional embeddings work, but ALiBi fits this setup better:

- No position-embedding table
- Relative bias available in every layer
- Fewer constraints when pushing sequence lengths

Reference: [Train Short, Test Long: ALiBi](https://arxiv.org/abs/2108.12409).

## Decision 4: Compress Large Categorical Spaces with QR Embeddings

For high-cardinality IDs (for example, artist IDs), we use quotient-remainder factorization:

```text
embed(id) = embed_q(id // R) + embed_r(id % R)
```

With `R = 1024`, this substantially reduces table size while preserving enough representational capacity for ranking.

This tradeoff is intentional: a modest representation loss is acceptable if it unlocks larger batches and faster iteration.

## Handling Sparse IDs and Missing Content Embeddings

What we observed:

- Item ID space can extend to ~9.4M
- Only a subset has precomputed content embeddings
- Missingness is non-trivial (around 18% in this run)

What we did:

- Keep a compact ID-to-RQ lookup for known items
- Route unknown/missing entries to a deterministic all-zero RQ code pattern
- Let the model learn a stable fallback behavior for unknown content

This avoids huge dense lookup tensors and keeps behavior deterministic.

## Jagged Batching with Block Masks

All sequence events are concatenated into one token buffer plus offsets. Attention is constrained to legal user-local ranges.

{{< mermaid >}}
flowchart LR
    A[User 1 tokens] --> D[Concatenated token buffer]
    B[User 2 tokens] --> D
    C[User 3 tokens] --> D
    D --> E[Block mask: no cross-user attention]
{{< /mermaid >}}

This improves accelerator utilization versus naive per-user padding while preserving exact user boundaries.

## Early Signal

| Step | Main metric |
| --- | --- |
| 32,432 | 0.0723 |
| 64,864 | 0.1431 |
| 97,296 | **0.1607** |

- Absolute gain: `+0.0884`
- Relative improvement from first logged point: about `2.2x`

This is still an intermediate training signal. Final offline validation should report `Hit@K`, `MRR`, and `NDCG` on retrieved candidates.
For boundary-case debugging during retrieval-stage evaluation, see [The Role of Negative Mining in Machine Learning](./hard_negatives.md).

## What Can Break

The main failure modes are straightforward:

1. Quantization bottleneck. If RQ codebooks are underfit, retrieval quality saturates early.
2. Bucket design brittleness. Bad time/duration buckets quietly cap model quality.
3. Missingness leakage. If unknown embeddings correlate with labels, the fallback path can become a shortcut feature.
4. Evaluation mismatch. Improvements in training metric may not transfer to retrieval-stage KPIs.

Recommended guardrails:

- Run ablations for each bias term (ALiBi/time/duration/organic)
- Track calibration and recall at retrieval depth
- Monitor unknown-code frequency by segment
- Keep one non-quantized baseline for regression detection
- For retrieval-quality diagnostics at scale, combine this with [Entity Resolution using Contrastive Learning](./entity_resolution.md)-style candidate analysis.

## Next Experiments

- QK normalization
- Per-layer residual scaling
- Logit soft-capping
- Alternative optimizers (including [Muon](https://github.com/KellerJordan/Muon))
- Explicit ablations of duration and organic priors

## References

- [PyTorch FlexAttention](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)
- [ALiBi paper](https://arxiv.org/abs/2108.12409)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Mermaid](https://mermaid.js.org/)

## Related Posts

- [The Role of Negative Mining in Machine Learning](./hard_negatives.md)
- [Entity Resolution using Contrastive Learning](./entity_resolution.md)
- [Machine Learning Engineer Roadmap](./ml_engineer_guidelines.md)

## Closing

At Yambda scale, disciplined defaults win:

- no dense attention bias matrices,
- no giant item softmax,
- no full-width categorical tables when compression is enough,
- no padding-heavy batching.

That is the full pattern: preserve signal, strip avoidable cost.
