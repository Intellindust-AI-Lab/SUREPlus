
---

# 🧠 From Misclassifications to Outliers — Joint Reliability Assessment in Classification

**Official repository** for the paper

> **“From Misclassifications to Outliers: Joint Reliability Assessment in Classification.”**

This repository provides the **training**, **evaluation**, and **OOD testing** framework for a unified reliability assessment approach that jointly performs **confidence estimation** and **out-of-distribution detection**.
The implementation supports **ResNet-18** and **DINOv3 (ViT-L/16 distilled)** backbones, and is fully compatible with **OpenOOD**.

---

## 🚀 Highlights

* Unified framework for **confidence estimation** & **OOD detection**
* Support for **PixMix**, **Mixup**, and **RegMixup** augmentations
* **OpenOOD**-compatible evaluation scripts
* Pretrained checkpoints for reproducible experiments

---

## 1️⃣ Data Preparation

Supported datasets: **CIFAR-100** and **ImageNet-1K**.

### Paths

Specify dataset directories via CLI:

```bash
--train-dir /path/to/imagenet/train
--val-dir   /path/to/imagenet/val
```

### PixMix 

Download [PixMix](https://github.com/andyzoujm/pixmix) and set its path:

```bash
--pixmix-path /path/to/pixmix
```

---

## 2️⃣ Pretrained Weights

We provide all pretrained and fine-tuned checkpoints via the following download link:

👉 **[Pretrained Models Download](https://your-link-here.com)**


These checkpoints can be directly used for evaluation by setting the `CHECKPOINTS` path in the testing script.

---

### 🔹 DINOv3 (ViT-L/16 distilled)

For DINOv3, please first set up the official pretrained weights following Meta’s release:
🔗 [Meta AI DINOv3 Downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)

Example:

```bash
--dinov3-path /path/to/dinov3_vitl16.pth
--dinov3-repo /path/to/dinov3
```


---


---

## 3️⃣ Training

Training scripts are under `run/train/`.
You may run shell wrappers or call `main.py` directly.

### Example: Train ResNet-18 on CIFAR-100

```bash
python3 main.py \
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
  --save-dir ./ResNet18-Cifar100/SURE+
```

### Example: Fine-tune DINOv3-L16 on ImageNet-1K

```bash
python3 main.py \
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
  --save-dir ./DinoV3_L16-ImageNet1k/SURE+
```

**Key Arguments**

* `--model-name`: backbone (`resnet18`, `dinov3_l16`, …)
* `--optim-name`: optimizer (`sam`, `fmfp`, `fsam`)
* `--pixmix-weight`, `--regmixup-weight`: data augmentation strength
* `--save-dir`: checkpoint directory

---

## 4️⃣ Testing & OOD Evaluation

Testing scripts are under `run/test/`, fully compatible with **OpenOOD**.

### Workflow

1. Run baseline (e.g., MSP) to save raw logits.
2. Apply post-processors (e.g., Energy, GradNorm) on saved results.

### Cosine Classifier Models

When using CSC models, specify:

```bash
--network.name resnet18_32x32_csc
--network.name dinov3_l_csc
```

### Example

```bash
bash run/test/resnet18/test.sh
```

### OOD Dataset Config

Configure dataset paths under:

```
openood/configs/datasets/cifar100/
openood/configs/datasets/imagenet/
```

Refer to [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main) for structure and usage.

---

## 📁 Project Layout

```
.
├── main.py                  # main entry for training
├── train.py                 # alternative training entry
├── run/
│   ├── train/               # training scripts
│   ├── test/                # testing & OOD evaluation
├── model/                   # backbone + classifier definitions
├── data/                    # dataset utilities
├── utils/                   # helper functions and options
├── openood/                 # OpenOOD integration & configs
└── requirements.txt         # environment dependencies
```

---

## ⚙️ Requirements

* Python ≥ 3.10
* PyTorch ≥ 2.0
* CUDA ≥ 11.4

For reproducibility:

```bash
pip install -r requirements.txt
```

or export a conda environment:

```bash
conda env export > environment.yml
```

---

## 📜 Citation

```bibtex
@article{li2025from,
  title={From Misclassifications to Outliers: Joint Reliability Assessment in Classification},
  author={Li, Yang and Sha, Youyang and Wang, Yinzhi and Hu, Shell Xu and Shen, Xi and Yu, Xuanlong},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}
```

---

## 🙏 Acknowledgements

This work builds upon **OpenOOD** and **DINOv3**.
We thank their authors for sharing high-quality open-source code and pretrained models.


