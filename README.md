---

# ImageNet-1K ResNet50 Training Framework

This repository contains a training framework for **ResNet50** on **ImageNet-1K**, with training configurations aligned with **OpenOOD**.

> **Note:** This code **only supports training**. For evaluation/testing, please use modified [OpenOOD](https://github.com/LIYangggggg/SURE-v2/tree/openood).

---

## Environment Setup

The environment is straightforward to configure. A pre-exported Conda environment is provided:

* **Python:** 3.8
* **PyTorch:** 1.12.1+cu113

To create the environment:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

For training, only the **training set** and **validation set** are required.

1. **Training Set** :Downloadable directly from the ImageNet website. Specify the directory in the training script using the `--train-dir` flag.
2. [**Validation Set**](https://drive.google.com/file/d/1rQvyeQNek1mGZxul9xar2A9skSgsfz7p/view?usp=sharing): Place validation images in:

```text
./imagenet/val
```

Specify this directory in the training script using the `--val-dir` flag.

---

## Training

* Specify the [PixMix dataset](https://drive.google.com/file/d/1wnjYlCNArbjOXDlhCGZC53UxM-t-UvHV/view?usp=sharing) path in `option.py`.
* Download the pretrained weights for [ResNet-50](https://download.pytorch.org/models/resnet50-0676ba61.pth) and place the file under `./pretrained_model/resnet50-0676ba61.pth`.

Under the `run/` directory, six different training scripts are provided, each corresponding to a specific training configuration:


* **ce**
* **cutmix**
* **sure**
* **ours**

### Training Settings

* Each script performs **3 independent runs**.
* **Epochs:** 100
* **Batch Size:** 256 

Example command to run baseline training:

```bash
bash run/ce.sh
```

---

## Notes

* Make sure the training/validation paths are correctly specified in each script.
* All training scripts are configured to match OpenOOD’s hyperparameters for fair comparison.
* Testing and evaluation must be performed using OpenOOD.


