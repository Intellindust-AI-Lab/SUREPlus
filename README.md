<div align="center">

#  From Misclassifications to Outliers : Joint Reliability Assessment in Classification
##

[![arXiv](https://img.shields.io/badge/arXiv-2603.03903-b31b1b.svg)](https://arxiv.org/abs/2603.03903)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://your-project-page.github.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

</div>

Official PyTorch implementation of the paper **"From Misclassifications to Outliers: Joint Reliability Assessment in Classification"**.


---

## 📋 Table of Contents

- [Overview](#-overview)
- [Updates](#-updates)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Pretrained Models](#-pretrained-models)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

Existing approaches treat misclassification detection and OOD detection as separate problems. We propose a unified reliability assessment framework that:

- **Jointly optimizes** for both confidence calibration and OOD detection
- **Supports multiple backbones**: ResNet-18 and DINOv3 ViT-L/16
- **Integrates seamlessly** with OpenOOD for comprehensive evaluation
- **Achieves state-of-the-art** results on CIFAR-100 and ImageNet-1K


---

## 📢 Updates

- **[2026-03-05]** Paper released on arXiv
- **[2025-12-26]** Code and pretrained models released

---

## 🛠️ Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.4

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/SURE-plus.git
cd SURE-plus

# Create virtual environment
conda create -n sure_plus python=3.10
conda activate sure_plus

# Install dependencies
pip install -r requirements.txt

# For CUDA 12.4 (recommended)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
```

---

## 📁 Data Preparation



For instructions on downloading and preparing the dataset, please refer to the official guide from **[OpenOOD](https://github.com/Jingkang50/OpenOOD)**.

### PixMix Dataset

Download [PixMix](https://github.com/andyzoujm/pixmix) augmentation images.

### Training Dataset Structure

Organize your datasets in ImageFolder format:

```
/path/to/dataset/
├── train/
│   ├── class_001/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class_001/
    └── ...
```



---

## 🔥 Pretrained Models

We provide [pretrained checkpoints](https://drive.google.com/drive/folders/17GZPbCy9jeh6gClaztYPJzPSKEu5XZHn?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto) for SURE+


### DINOv3 Setup

For DINOv3, download the official pretrained weights from [Meta AI](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/):

```bash
# Set paths in your training scripts
--dinov3-path /path/to/dinov3_vitl16.pth \
--dinov3-repo /path/to/dinov3
```

---

## 🚀 Training

Training scripts are located in `run/train/`. We support both single-GPU and multi-GPU (DDP) training.

### Quick Start

#### ResNet-18 on CIFAR-100 (SURE+)

```bash
python main.py \
  --gpu 0 \
  --lr 0.05 \
  --batch-size 128 \
  --epochs 200 \
  --model-name resnet18 \
  --optim-name fsam \
  --pixmix-weight 1.0 \
  --regmixup-weight 1.0 \
  --rebn \
  --pixmix-path ./PixMixSet/fractals_and_fvis/first_layers_resized256_onevis/ \
  --save-dir ./checkpoints/ResNet18-Cifar100/SURE+ \
  Cifar100
```

Or use the provided script:
```bash
bash run/train/resnet18/SURE+.sh
```

#### DINOv3-L/16 on ImageNet-1K (SURE+)

```bash
python main.py \
  --gpu 0 1 2 3 4 5 6 \
  --lr 1e-5 \
  --weight-decay 5e-6 \
  --batch-size 64 \
  --epochs 20 \
  --model-name dinov3_l16 \
  --optim-name fsam \
  --pixmix-weight 1.0 \
  --mixup-weight 1.0 \
  --mixup-beta 10.0 \
  --rebn \
  --dinov3-repo ./dinov3 \
  --dinov3-path ./dinov3/dinov3_vitl16_pretrain.pth \
  --save-dir ./checkpoints/DinoV3_L16-ImageNet1k/SURE+ \
  ImageNet1k
```

Or use the provided script:
```bash
bash run/train/dinov3/SURE+.sh
```
---

## 📊 Evaluation

Testing scripts are in `run/test/` and are fully compatible with **OpenOOD**.

### Evaluation Workflow

1. **Baseline Evaluation**: Save raw logits
2. **Post-processing**: Apply various OOD detectors

### Supported Post-processors

- `msp` - Maximum Softmax Probability
- `odin` - ODIN
- `energy` - Energy Score
- `gradnorm` - Gradient Norm
- `knn` - k-Nearest Neighbors
- `vim` - Virtual-logit Matching
- And more...

### Quick Evaluation

```bash
# Evaluate ResNet-18 on CIFAR-100
bash run/test/resnet18/test.sh
```

### Custom Evaluation

```bash
export CUDA_VISIBLE_DEVICES=0

PYTHONPATH='.':$PYTHONPATH \
python openood/main.py \
  --config openood/configs/datasets/cifar100/cifar100.yml \
  openood/configs/datasets/cifar100/cifar100_ood.yml \
  openood/configs/networks/resnet18_32x32.yml \
  openood/configs/pipelines/test/test_ood.yml \
  openood/configs/preprocessors/base_preprocessor.yml \
  openood/configs/postprocessors/msp.yml \
  --network.checkpoint "./checkpoints/ResNet18-Cifar100/SURE+/best_1.pth" \
  --network.name resnet18_32x32 \
  --output_dir "./results/SURE+"
```

### Cosine Classifier Models

For CSC models, use the appropriate network config:

```bash
--network.name resnet18_32x32_csc  # For ResNet-18
--network.name dinov3_l_csc        # For DINOv3
```

---

## 🏆 Results

### ResNet-18 on CIFAR-100 and its Near/Far-OOD datasets

| Method | Acc ↑ | DS-F1 ↑ | DS-AURC ↓ |
|--------|-------|----------|---------|
| Cross-Entropy | 77.32 | 67.42 / 57.03 | 202.38 / 367.56 |
| SURE | 80.55 | 68.07 / 53.09 | 199.05 / 393.25 | 
| **SURE+ (Ours)** | **81.66** | **70.67 / 61.35** | **173.45 / 314.04** | **88.3** |

### DINOv3 ViT-L/16 on ImageNet-1K and its Near/Far-OOD datasets

| Method | Acc ↑ | DS-F1 ↑ | DS-AURC ↓ |
|--------|-------|----------|---------|
| Cross-Entropy | 86.89 | 75.42 / 84.38 | 168.33 / 48.07 |
| SURE | 87.94 | 76.62 / 85.44 | 157.07 / 42.13 | 
| **SURE+ (Ours)** | **88.49** | **77.10 / 86.15** | **156.00 / 38.07** |

*Full results available in the [paper](https://arxiv.org/abs/2501.xxxxx).*

---

## 📁 Project Structure

```
SURE-plus/
├── 📄 main.py                  # Main training entry point
├── 📄 train.py                 # Training loop implementation
├── 📁 model/                   # Model definitions
│   ├── resnet18.py            # ResNet-18 backbone
│   ├── classifier.py          # Cosine classifier
│   └── get_model.py           # Model factory
├── 📁 data/                    # Data loading utilities
│   ├── dataset.py             # Dataset and DataLoader
│   └── sampler.py             # Custom samplers
├── 📁 utils/                   # Utility functions
│   ├── option.py              # Argument parser
│   ├── optim.py               # Optimizers & schedulers
│   ├── ema.py                 # Exponential moving average
│   ├── sam.py / fsam.py       # SAM implementations
│   ├── valid.py               # Validation metrics
│   └── utils.py               # Helper functions
├── 📁 openood/                 # OpenOOD integration
│   ├── configs/               # Dataset & model configs
│   └── main.py                # OpenOOD evaluation
├── 📁 run/                     # Training & testing scripts
│   ├── train/                 # Training scripts
│   └── test/                  # Testing scripts
├── 📄 requirements.txt         # Python dependencies
└── 📄 README.md               # This file
```

---

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@article{li2026from,
  title={From Misclassifications to Outliers: Joint Reliability Assessment in Classification},
  author={Li, Yang and Sha, Youyang and Wang, Yinzhi and Timothy Hospedales and  Shell, Xu Hu and Shen, Xi and Yu, Xuanlong},
  journal={arXiv preprint arXiv:2603.xxxxx},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This work builds upon the following excellent open-source projects:

- **[OpenOOD](https://github.com/Jingkang50/OpenOOD)** - OOD detection benchmark
- **[DINOv3](https://github.com/facebookresearch/dinov3)** - Self-supervised vision transformer

We thank the authors for sharing their high-quality code and pretrained models.

---

## 📮 Contact

For questions or feedback, please open an [issue](https://github.com/yourusername/SURE-plus/issues) or contact the authors.

---

<div align="center">

⭐ Star us on GitHub — it motivates us a lot!

</div>
