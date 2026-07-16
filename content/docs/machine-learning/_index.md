---
weight: 3
bookFlatSection: false
title: "Machine Learning Toolkit"
---

## Machine Learning Toolkit

Tools and techniques I use across the ML lifecycle, from raw data to a monitored production model. A more detailed companion to my [Machine Learning Proficiency](/).

## Data Engineering

- Building unbiased datasets via feature-based sampling.
- Generating synthetic data that matches the real data distribution.
- Augmentation techniques for `vision`, `speech`, and `NLP`.

## Modelling & Feature Engineering

- `Convolutional`, `Recurrent`, and `Transformer`-based architectures.
- Contrastive learning: `SimCLR`, `BYOL`, `SimSiam`.
- Feature-importance analysis, model debugging, and profiling.
- Topic models via Probabilistic Graphical Models and embedding-based clustering.

## Training

- Distributed training with `OpenMPI + RoCE` and `Torch RPC`.
- `PyTorch Lightning` optimizations for throughput and memory.

## Calibration

- **Implicit**: `Focal Loss`, Maximum-Entropy Regularization, Label Smoothing, Random Dropout.
- **Explicit**: Isotonic Regression, Platt's scaling.

## Optimization

- Model compression: `Quantization`, `Pruning`, `Distillation`.
- Graph fusing and compilation via `ONNX`, `TorchDynamo`, `TVM`.

## Inference

- Low-latency `C++` serving with `ONNX` and `Drogon`.
- Serving frameworks: `Triton`, `Mosec`.
- Scaling on `k8s` via `OKD`.
- Monitoring & alerting with `Vector.io`, `Prometheus`, and `Grafana`.

## Online Monitoring

- Hard-negative mining around the calibrated threshold region.
- Sampling and persisting hard negatives for the next training round.
- Detecting and alerting on model and data drift.
