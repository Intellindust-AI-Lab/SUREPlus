# 🧠 FROM MISCLASSIFICATIONS TO OUTLIERS: JOINT RELIABILITY ASSESSMENT IN CLASSIFICATION

This repository provides the official **training framework** accompanying our paper:

> **“From Misclassifications to Outliers: Joint Reliability Assessment in Classification.”**

We present a unified framework for **joint confidence estimation and OOD detection**, enabling robust reliability modeling across diverse architectures.
This codebase supports **DINOv3 (ViT-L/16 distilled)** and **ResNet-18** backbones, trained on **ImageNet-1K**, with configurations fully compatible with the [**OpenOOD**](https://github.com/LIYangggggg/SURE-v2/tree/openood) benchmark.

---

## 🧩 Overview

Our framework provides a modular and reproducible implementation of the **training stage** for large-scale representation learning and reliability assessment.
It includes:

* 🔹 **DINOv3** training scripts (`run/dinov3/`)
* 🔹 **ResNet-18** training scripts (`run/r18/`)
* 🔹 Configuration alignment with **OpenOOD** for direct evaluation
* 🔹 Support for **PixMix**, **CutMix**, and other augmentation-based robustness studies

> ⚠️ **Note:** This repository focuses on **training**.
> For evaluation and OOD testing, please use the modified [OpenOOD](https://github.com/LIYangggggg/SURE-v2/tree/openood) framework.

---

## 📦 Environment Setup

A minimal environment setup is required for training.

**Requirements:**

* Python ≥ 3.10
* PyTorch ≥ 2.0
* CUDA ≥ 11.7

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Data Preparation

This framework requires only the **training** and **validation** splits of **ImageNet-1K**.

1. **Training Set:**
   Download from the [official ImageNet site](https://image-net.org/) and specify the path with:

   ```bash
   --train-dir /path/to/imagenet/train
   ```

2. **Validation Set:**
   Download from [Google Drive](https://drive.google.com/file/d/1rQvyeQNek1mGZxul9xar2A9skSgsfz7p/view?usp=sharing) and place it as:

   ```
   ./imagenet/val
   ```

   Specify via:

   ```bash
   --val-dir /path/to/imagenet/val
   ```

---

## 🔗 Pretrained Weights & External Resources

* **DINOv3 Pretrained Weights:**
  Download ViT-L/16 distilled weights from [Meta AI](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and specify:

  ```bash
  --dinov3-path /path/to/dinov3_vitl16.pth
  ```

* **DINOv3 Repository:**
  Clone [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) and specify:

  ```bash
  --dinov3-repo /path/to/dinov3
  ```

* **PixMix Dataset (Optional):**
  Download from [Google Drive](https://drive.google.com/file/d/1wnjYlCNArbjOXDlhCGZC53UxM-t-UvHV/view?usp=sharing) and define the path in `option.py`.

---

## 🚀 Training Usage

All training scripts are located under the `run/` directory.
Each subfolder corresponds to a backbone:

```
run/
 ├── dinov3/      # ViT-L/16 distilled training (DINOv3)
 └── r18/         # ResNet-18 baseline training
```

### Example 1: Train DINOv3 with Cross-Entropy

```bash
bash run/dinov3/ce.sh
```

### Example 2: Train DINOv3 with Our Proposed Method

```bash
bash run/dinov3/ours.sh
```

### Example 3: Train ResNet-18 Baselines

```bash
bash run/r18/ce.sh
bash run/r18/ours.sh
```


## 🧠 Research Context

This repository is designed for research on:

* **Misclassification reliability estimation**
* **OOD and uncertainty calibration**
* **Joint assessment of in- and out-distribution confidence**
* **Robust representation learning under distribution shifts**

---

## 📁 Directory Structure

```
.
├── run/
│   ├── dinov3/              # DINOv3 (ViT-L/16 distilled) training scripts
│   └── r18/                 # ResNet-18 training scripts
│
├── configs/                 # YAML configuration files
├── models/                  # Model definitions (backbone + classifier)
├── utils/                   # Training utilities and helper functions
├── option.py                # Argument definitions
├── requirements.txt         # Dependency list
└── README.md                # This document
```

---

## 📜 Citation

If you find this repository helpful, please consider citing our work:



