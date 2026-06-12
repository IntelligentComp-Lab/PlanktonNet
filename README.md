<h1 align="center">PlanktonNet</h1>
<h3 align="center">A Three-Stage Hybrid Backbone for Fine-Grained Plankton Recognition</h3>

<p align="center">
  <a href="Manuscript.pdf"><img src="https://img.shields.io/badge/Paper-PDF-red"></a>
  <a href="https://github.com/chenyanting1/PlanktonNet-A-Three-Stage-Hybrid-Backbone-for-Fine-Grained-Plankton-Recognition"><img src="https://img.shields.io/badge/GitHub-Repo-blue"></a>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green">
</p>

---

Official code repository for the paper:

> **PlanktonNet: A Three-Stage Hybrid Backbone for Fine-Grained Plankton Recognition**

<p align="center">
  <img src="assets/architecture.png" alt="PlanktonNet Architecture" width="800">
</p>

## Highlights

- A **three-stage hybrid architecture** that preserves high-resolution representations for fine-grained plankton features
- **UltraFormer block** integrating LESRA (Locally Enhanced Spatial Reduction Attention) and SwiGLU MLP gating
- **CAC (Class Anchor Clustering)** regularization for compact intra-class feature distributions
- State-of-the-art closed-set and open-set recognition across four plankton benchmarks

---

## Key Results

### Closed-Set Classification (Top-1 Accuracy)

| Benchmark | Accuracy |
|-----------|:--------:|
| ZooplanktonBench50 | **91.45%** |
| ZooCAMNet76 | **83.91%** |
| ZooScanNet101 | **91.30%** |
| HZWPlankton8 | **97.94%** |

### Open-Set Recognition (CAC + PlanktonNet, Table 8 in paper)

| Metric | Value |
|--------|:-----:|
| Known Acc (Closed) | **87.11%** |
| Known Acc (Th) | **85.20%** |
| Unknown Acc (Th) | **47.62%** |
| Open-Set Acc | **77.17%** |
| Open-Set F-score | **85.44%** |
| AUROC | **83.94%** |
| AUPR | **58.13%** |
| FPR95 | **58.47%** |

---

## File Structure

```
.
├── PlanktonNet.py              # PlanktonNet backbone
│   ├── PatchEmbedding           #   Stem
│   ├── MBConvDownSampling       #   Downsampling
│   ├── UltraFormerBlock         #   Core block
│   │   ├── SwiGLU MLP           #     Channel gating
│   │   ├── Global Mixer         #     LESRA token mixing
│   │   └── Group Normalization  #     Stable small-batch training
│   └── UltraFormer              #   Generic backbone class
│
├── cacUtils.py                 # CAC regularization
│   ├── CACModel                 #   Anchor-space projection
│   ├── CACLoss                  #   Anchor constraint loss
│   └── CACClassif               #   Inference-time scoring
│
├── train.py                    # Standard softmax training
├── train_cac.py                # Training with CAC regularization
├── validate_github.py          # Closed-set validation
├── test_cac_osr.py             # CAC OSR evaluation
├── osr_metrics.py              # OSR metric functions 
├── benchmark_jetson.py         # Jetson Orin NX edge deployment
│
└── checkpoint-261.pth.tar      # Pre-trained CAC + PlanktonNet
```

---

## Dependencies

```bash
pip install torch torchvision timm torchmetrics numpy tqdm
```

| Package | Purpose |
|---------|---------|
| PyTorch ≥ 1.12 | Deep learning framework |
| timm ≥ 0.9 | Data loading, optimizers, schedulers |
| torchmetrics | Pairwise distance computation |
| numpy, tqdm | Utilities |

---

## Datasets

### Available Benchmarks

| Dataset | Classes | Link |
|---------|:---:|------|
| ZooplanktonBench50 | 50 | [Google Drive](https://drive.google.com/file/d/1okK2YFM7SoFCH6xRhrk4bdoeOGrLhJHL/view?usp=drive_link) |
| ZooplanktonBench50 (Open-Set) | 50 + unknown | [Google Drive](https://drive.google.com/file/d/1mX-0wK0jmOaSuKmu-4uIuvT0WdneQOQM/view?usp=drive_link) |
| ZooCamNet76 | 76 | [Google Drive](https://drive.google.com/file/d/1UaPOsMGUktfrrUdexq6H6M6WgRibGJhN/view?usp=drive_link) |
| ZooScanNet101 | 101 | [Google Drive](https://drive.google.com/file/d/1s3qmZB9b_SUTyQhtZXNNY6O4UCXpqKgH/view?usp=drive_link) |
| HZWPlankton | — | Email: chenyanting@stu.hznu.edu.cn |

### Directory Convention

**Closed-set datasets:**
```
dataset/
├── train/           # Training set
├── validation/      # Validation set
└── test/            # Test set
```

**Open-set datasets** (names suffixed with `_openset`):
```
dataset_openset/
├── train/              # Gallery (K known classes)
├── validation/         # Known-class validation
├── validation_unknow/  # Unknown-class validation (threshold tuning)
├── test_know/          # Known-class test
└── test_unknow/        # Unknown-class test
```

---

## Quick Start

### 1. Train PlanktonNet (Softmax Baseline)

```bash
python train.py \
    --data-dir ZooplanktonBench50/ \
    --model PlanktonNet \
    --num-classes 50 \
    --batch-size 26 \
    --validation-batch-size 26 \
    --workers 4 \
    --epochs 300 \
    --opt adamw --lr 1e-4 \
    --warmup-epochs 10 --warmup-lr 1e-5 \
    --clip-grad 0.5 --weight-decay 0.2 \
    --cooldown-epochs 5 \
    --train-interpolation bicubic \
    --mixup 0.4 --cutmix 0.6 \
    --aa rand-m9-mstd0.5-inc1 \
    --img-size 224 \
    --device cuda:0
```

### 2. Train with CAC Regularization

```bash
python train_cac.py \
    --data-dir ./ \
    --dataset ZooplanktonBench50_padding \
    --num-classes 50 \
    --feature-dim 448 \
    --batch-size 64 \
    --epochs 400 \
    --opt adamw --lr 1e-4 \
    --cac-magnitude 10.0 \
    --cac-scale 0.1 \
    --device cuda:0
```

| Hyperparameter | Default | Description |
|:---|:---:|:---|
| `--cac-magnitude` | 10.0 | Scale of orthonormal anchor space |
| `--cac-scale` (λ) | 0.1 | Weight of anchor-distance regularization |
| `--feature-dim` | 448 | PlanktonNet output dimension |

### 3. Open-Set Evaluation

```bash
python test_cac_osr.py \
    --data-dir ./ \
    --dataset ZooplanktonBench50_padding \
    --checkpoint checkpoint-261.pth.tar \
    --num-classes 50 \
    --feature-dim 448 \
    --batch-size 64 \
    --device cuda:0
```

Threshold τ is selected on the validation set (`validation` + `validation_unknow`) and applied to the test set.

### 4. Closed-Set Validation

```bash
python validate_github.py \
    --data-dir ZooplanktonBench50/test \
    --model PlanktonNet \
    --checkpoint output/.../model_best.pth.tar \
    --img-size 224 --num-classes 50 \
    --batch-size 64 --metrics-avg macro \
    --device cuda:0
```

---

## Method Overview

### PlanktonNet Architecture

A three-stage hybrid architecture centered on the UltraFormer block:

| Stage | Downsampling | Dimension | Depth | Resolution |
|:-----:|-------------|:---------:|:-----:|:-----------:|
| Stem | PatchEmbedding (4×) | 200 | 1 | 56 × 56 |
| 1 | — | 200 | 2 | 56 × 56 |
| 2 | MBConvDownSampling (2×) | 376 | 3 | 28 × 28 |
| 3 | MBConvDownSampling (2×) | 448 | 2 | 14 × 14 |
| Head | GAP + LayerNorm | 448 | — | 1 × 1 |

By avoiding the 32× downsampling typical of four-stage backbones, PlanktonNet preserves the high-frequency spatial details critical for discriminating morphologically similar plankton species (e.g., copepod antennae).

### UltraFormer Block

Each block consists of four residual sub-blocks:

1. **DWConv 7×7** — Local spatial feature extraction
2. **SimpleFFN** — Lightweight channel mixing before attention
3. **Global Mixer (LESRA)** — Locally Enhanced Spatial Reduction Attention for long-range dependency modeling at reduced computational cost
4. **SwiGLU MLP** — Gated channel mixing that suppresses background water-column interference

All normalization layers use **Group Normalization** (G=1) to ensure stable convergence under small-batch training.

### CAC Regularization

The loss combines cross-entropy with a CAC anchor constraint ($\lambda = 0.1$):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{CAC}}$$

The CAC loss pulls features toward fixed orthonormal anchors $A = M \cdot I_K$ ($M = 10.0$):

$$\mathcal{L}_{\text{CAC}} = \frac{1}{B}\sum_{i=1}^{B} \| z_i - a_{y_i} \|_2^2$$

At inference, the projection matrix $W$ and training anchors $A$ are discarded. Class centers $c_j$ are recomputed as the per-class mean of backbone features $F(x)$ from the training set (gallery). A sample is rejected if its distance to the nearest center exceeds $\tau$:

$$c_j = \frac{1}{|D_j|} \sum_{(x,y) \in D_j} F(x), \qquad S_{\text{anomaly}}(x) = \min_j \| F(x) - c_j \|_2$$

---

## Evaluation Metrics

Eight OSR metrics are reported—5 threshold-dependent and 3 threshold-independent:

### Threshold-Dependent

| # | Metric | Description |
|---|--------|-------------|
| 1 | Known Acc (Closed) | Standard closed-set accuracy |
| 2 | Known Acc (Th) | Known samples correctly predicted and accepted |
| 3 | Unknown Acc (Th) | Unknown samples correctly rejected |
| 4 | Open-Set Acc | (TP + TN) / Total |
| 5 | Open-Set F-score | Harmonic mean of precision and recall |

### Threshold-Independent

| # | Metric | Description |
|---|--------|-------------|
| 6 | AUROC | Area under the ROC curve |
| 7 | AUPR | Area under the precision-recall curve |
| 8 | FPR95 | False positive rate at 95% true positive rate |

---

## Edge Deployment

The `benchmark_jetson.py` script measures deployment efficiency on the **NVIDIA Jetson Orin NX** (JetPack 6.2.1, PyTorch 2.5.0, CUDA 12.6), reporting params, FLOPs, model size, accuracy, FP32/FP16 latency, throughput, and GPU memory.

```bash
# Efficiency metrics only (no dataset needed)
python benchmark_jetson.py \
    --model PlanktonNet \
    --checkpoint checkpoint-261.pth.tar \
    --num-classes 50 \
    --img-size 224 \
    --output results.csv

# With accuracy measurement
python benchmark_jetson.py \
    --model PlanktonNet \
    --checkpoint checkpoint-261.pth.tar \
    --data-dir ZooplanktonBench50/test \
    --num-classes 50 \
    --img-size 224 \
    --output results.csv
```

---

## Citation

```bibtex
@article{chen2025planktonnet,
  title   = {PlanktonNet: A Three-Stage Hybrid Backbone for
             Fine-Grained Plankton Recognition},
  author  = {},
  journal = {Elsevier},
  year    = {2026}
}
```

---

## License

The PlanktonNet backbone and UltraFormer architecture are proposed in this paper. The codebase includes components adapted from timm (Apache 2.0).
