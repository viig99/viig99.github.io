---
title: Introduction
type: docs
---

## Applied Machine Learning Engineer

Hi, I'm **Arjun**, an Applied Machine Learning Engineer with 14+ years of building and shipping ML systems. I work across the whole stack: data and sampling, modelling, distributed training, and low-latency `C++` / `Python` inference at scale.

I'm currently a **Senior Applied ML Engineer at [Shopify](https://www.shopify.com/)**, working on merchant foundation models.

## Machine Learning Proficiency

Areas I've worked in across the full project lifecycle:

- **Data & sampling**: unbiased datasets via feature-based sampling, hard-negative mining, and synthetic data that matches the real distribution.
- **Modelling**: `Convolutional`, `Recurrent`, and `Transformer` architectures; contrastive learning (`SimCLR`, `BYOL`, `SimSiam`); and sequential models.
- **Distributed training**: GPU clusters over `OpenMPI + RoCE`, `Torch RPC`, and `PyTorch Lightning`.
- **Calibration**: `Focal Loss`, label smoothing, isotonic regression, and Platt scaling.
- **Optimization**: quantization, pruning, distillation, and graph fusing with `ONNX`, `TorchDynamo`, and `TVM`.
- **Inference**: scalable, low-latency `C++` serving (`ONNX`, `Drogon`, `Triton`) on `k8s`.

## Recent Explorations

Some things I've built recently, and what each one looks at:

- **[imba-chess](https://github.com/viig99/imba-chess)**: a high-Elo chess bot that ports `HSTU` sequential transformers from recommender systems to chess, with policy imitation from Lichess games, a `WDL` value head, and depth-2 policy-pruned minimax search. It looks at how far sequence models transfer from recsys to strategic games. Build log: [Building a Chess Bot with HSTU]({{< relref "/docs/posts/imba-chess" >}}).
- **[muvfde](https://github.com/viig99/muvfde)**: fixed-dimensional embeddings for multi-vector representations, based on Google Research's `MUVERA`. The `C++` core is exposed to Python via `nanobind` and published [on PyPI](https://pypi.org/project/muvfde/). It trades model size for retrieval latency using a fixed-dimensional encoding.
- **[Comparing Online Hyperopts](https://github.com/viig99/Comparing-Online-Hyperopts)**: a benchmark of online hyper-parameter optimization methods on sampling efficiency, latency, and ease of implementation.

## Domain Expertise

Problems I've shipped machine learning for:

- Search & Ranking, Cold-start Recommendations
- Constraint-based Optimization
- Speech Processing (Speech-to-Text, Text-to-Speech)
- Computer Vision (Segmentation, Classification, OCR)
- Natural Language Processing (Document QA, Classification, Entity Recognition)
- Contrastive Learning methods

## Get in Touch

- [LinkedIn](https://www.linkedin.com/in/arjunvariar) {{< fab linkedin "#00a0dc" >}}
- [GitHub](https://github.com/viig99) {{< fab github "#211f1f" >}}
- [Twitter](https://twitter.com/vigi99/) {{< fab twitter "#7dbbe6" >}}
- [Email](mailto:accio.arjun@gmail.com) {{< far envelope "#EA4335" >}}
