---
title: Introduction
type: docs
---

## Applied Machine Learning Engineer

Hi, I'm **Arjun** — an Applied Machine Learning Engineer with 14+ years of turning fuzzy business problems into shipped ML systems. I like working end-to-end: data and sampling, modelling, distributed training, and low-latency `C++` / `Python` inference at scale.

I'm currently a **Senior Applied ML Engineer at [Shopify](https://www.shopify.com/)**, working on merchant foundation models.

## Machine Learning Proficiency

My work covers the full project lifecycle — and I keep it sharp by building things end-to-end:

- **Data & sampling** — unbiased datasets via feature-based sampling, hard-negative mining, and synthetic data that matches the real distribution.
- **Modelling** — `Convolutional`, `Recurrent`, and `Transformer` architectures; contrastive learning (`SimCLR`, `BYOL`, `SimSiam`); and sequential models.
- **Distributed training** — GPU clusters over `OpenMPI + RoCE`, `Torch RPC`, and `PyTorch Lightning`.
- **Calibration** — `Focal Loss`, label smoothing, isotonic regression, and Platt scaling.
- **Optimization** — quantization, pruning, distillation; graph fusing with `ONNX`, `TorchDynamo`, and `TVM`.
- **Inference** — scalable, low-latency `C++` serving (`ONNX`, `Drogon`, `Triton`) on `k8s`.

> [!TIP]
> The fastest way I learn a new technique is to build it end-to-end. The projects below are where that happens — and what they taught me tends to show up in production soon after.

## Recent Explorations

A few things I've been building lately, and what each one explores:

- **[imba-chess](https://github.com/viig99/imba-chess)** — a high-Elo chess bot that ports `HSTU` sequential transformers from recommender systems to chess: policy imitation from Lichess games, a `WDL` value head, and depth-2 policy-pruned minimax search. It probes how far sequence models transfer from recsys to strategic games. Full build log: [Building a Chess Bot with HSTU]({{< relref "/docs/posts/imba-chess" >}}).
- **[muvfde](https://github.com/viig99/muvfde)** — fixed-dimensional embeddings for multi-vector representations (Google Research's `MUVERA`), with a `C++` core exposed to Python via `nanobind` and shipped [on PyPI](https://pypi.org/project/muvfde/). An exercise in trading model size for retrieval latency using the right encoding.
- **[Comparing Online Hyperopts](https://github.com/viig99/Comparing-Online-Hyperopts)** — a benchmark of online hyper-parameter optimization methods across sampling efficiency, latency, and ease of implementation. A reminder that "best" depends entirely on your compute and latency budget.

## Domain Expertise

I've shipped machine learning across a broad range of problems:

- Search & Ranking · Cold-start Recommendations
- Constraint-based Optimization
- Speech Processing (Speech-to-Text, Text-to-Speech)
- Computer Vision (Segmentation, Classification, OCR)
- Natural Language Processing (Document QA, Classification, Entity Recognition)
- Contrastive Learning methods

## Get in Touch

I'm always up for a good ML problem — let's talk.

- [LinkedIn](https://www.linkedin.com/in/arjunvariar) {{< fab linkedin "#00a0dc" >}}
- [GitHub](https://github.com/viig99) {{< fab github "#211f1f" >}}
- [Twitter](https://twitter.com/vigi99/) {{< fab twitter "#7dbbe6" >}}
- [Email](mailto:accio.arjun@gmail.com) {{< far envelope "#EA4335" >}}
