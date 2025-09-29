# 🧠 DINOv3 Training Framework on ImageNet-1K

This repository provides a **training framework for DINOv3 (ViT-L/16 distilled)** on **ImageNet-1K**, with configurations aligned to the [**OpenOOD**](https://github.com/LIYangggggg/SURE-v2/tree/openood) benchmark.  
It is designed for research on **OOD detection**, **robust classification**, and **representation learning**.

> ⚠️ **Note:** This repository focuses on **training only**. For evaluation and testing, please use the modified [OpenOOD](https://github.com/LIYangggggg/SURE-v2/tree/openood) framework.



## 📦 Environment Setup

A minimal environment is required for training. A pre-defined Conda/Pip environment file is provided for convenience.

- **Python:** 3.10+
- **PyTorch:** 2.0+

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## 📂 Data Preparation

This framework requires only the **training** and **validation** splits of ImageNet-1K.

1. **Training Set:**
   Download from the [official ImageNet site](https://image-net.org/) and specify the directory with `--train-dir`.

2. **Validation Set:**
   Download the prepared validation split from [Google Drive](https://drive.google.com/file/d/1rQvyeQNek1mGZxul9xar2A9skSgsfz7p/view?usp=sharing) and place it as:

```
./imagenet/val
```

Specify this path with the `--val-dir` argument.

---

## 🧪 Pretrained Weights & External Dependencies

* **PixMix Dataset (optional):** [Download here](https://drive.google.com/file/d/1wnjYlCNArbjOXDlhCGZC53UxM-t-UvHV/view?usp=sharing) and specify the path in `option.py`.
* **DINOv3 Pretrained Weights:** [Download ViT-L/16 distilled weights](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and pass `--dinov3-path`.
* **DINOv3 Repository:** Fork [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) and specify the repo path with `--dinov3-repo`.

---

## 🚀 Training

Under the `run/` directory, multiple training scripts are provided, each corresponding to a specific configuration:

* `ce.sh` – Standard cross-entropy training
* `ours.sh` – Custom training strategy (proposed method)

### 📊 Default Settings

| Hyperparameter | Value        |
| -------------- | ------------ |
| Epochs         | 10           |
| Batch Size     | 64           |
| Optimizer      | AdamW        |
| Learning Rate  | Configurable |

Example: run baseline training with cross-entropy:

```bash
run/dinov3/ce.sh
run/dinov3/ours.sh
```

---

## 🔧 Advanced Training: Linear Head + Full Fine-Tuning

For scenarios where **DINOv3 backbone** is frozen initially and **fine-tuned later**, you can follow a two-stage training strategy:

1. **Stage 1:** Train only the linear head for 1 epoch with a higher learning rate.
2. **Stage 2:** Unfreeze the backbone and fine-tune the entire model with a lower learning rate.

This approach is recommended for transfer learning or OOD-related tasks.

---

## ⚙️ Additional Notes

* Ensure `--train-dir` and `--val-dir` are correctly specified in all training scripts.
* All hyperparameters are aligned with **OpenOOD** defaults for fair comparison.
* **Evaluation and testing are not included** in this repository. Please use the OpenOOD toolkit for post-training evaluation.

---

## 📁 Repository Structure

```
.
├── run/                     # Training scripts
├── configs/                # YAML config files
├── models/                 # Model definitions (DINOv3 + classifier)
├── utils/                  # Training utilities
├── option.py              # Argument definitions
└── requirements.txt       # Dependencies
```

