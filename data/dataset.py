# data/loaders.py
"""
Refactored data loader module.
Supports:
- cifar100 (image size 32)
- imagenet1k (image size 224)
- PixMix mixing dataset support (keeps PixMix logic; AugMix removed)
- Multi-GPU / distributed training (use `distributed=True` and provide rank/world_size via torch.distributed)
Notes:
- All comments are in English.
- No BCE / AugMix / imbalance generation code included.
"""

import os
from copy import deepcopy
import random
import numpy as np
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import data.augmentations as augmentations
import utils.pixmix_utils as pixmix_utils



def _get_mean_std(dataset_name: str):
    """Return normalization mean/std for supported datasets."""
    if dataset_name == "cifar100":
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    elif dataset_name == "imagenet1k":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def _build_transforms(dataset_name: str):
    """
    Build train/test transforms for supported datasets.
    - cifar100: train size 32.
    - imagenet1k: train size 224.
    model_name kept for possible model-specific normalization (e.g., deit).
    """
    mean, std = _get_mean_std(dataset_name)

    if dataset_name == "cifar100":
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            # T.ToTensor(),
            # T.Normalize(mean, std)
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        mixing_transform = T.Compose([
            T.Resize(36),
            T.RandomCrop(32)
        ])
    else:  # imagenet1k
        train_transform = T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            # T.ToTensor(),
            # T.Normalize(mean, std)
        ])
        test_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        mixing_transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224)
        ])

    return train_transform, test_transform, mixing_transform


# ----------------------
# PixMix related helper
# ----------------------
def _augment_input(image):
    aug_list = pixmix_utils.augmentations_all 
    op = np.random.choice(aug_list)
    return op(image.copy(), 3)


def pixmix_mixing(orig, mixing_img, preprocess):
    """
    Perform PixMix-style mixing and return normalized torch tensor.
    - orig: PIL.Image original image
    - mixing_img: PIL.Image chosen from mixing set
    - preprocess: dict with keys {'tensorize': ToTensor(), 'normalize': Normalize()}
    - augmentations_list: list of augmentation functions (optional)
    - mixings: list of mixing functions (optional)
    """
    mixings = pixmix_utils.mixings
    tensorize = preprocess['tensorize']
    normalize = preprocess['normalize']

    # choose either augmented original or original
    if random.random() < 0.5:
        mixed = tensorize(_augment_input(orig))
    else:
        mixed = tensorize(orig)

    # choose number of mixing iterations (1..4)
    for _ in range(np.random.randint(1, 4 + 1)):

        if random.random() < 0.5:
            partner = tensorize(_augment_input(orig)) 
        else:
            partner = tensorize(mixing_img)


    mixing_op = random.choice(mixings)
    mixed = mixing_op(mixed, partner, 3)
    mixed = torch.clip(mixed, 0, 1)
    return normalize(mixed)


# ----------------------
# Dataset wrappers
# ----------------------
class SimpleImageFolder(ImageFolder):
    """
    Simple wrapper around torchvision.datasets.ImageFolder that returns:
    (image_tensor, label, index, path)
    No one-hot labels, no imbalance logic.
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        img, label = super().__getitem__(index)
        return img, label, index, path


class PixMixImageFolder(ImageFolder):
    """
    ImageFolder wrapper for PixMix training.
    Returns:
      (clean_tensor, mixed_tensor, label, index, path)
    - mixed_tensor is produced by pixmix_mixing using mixing_set.
    - If pixmix_utils is not available, mixed_tensor == clean_tensor.
    """
    def __init__(self, root, transform=None, mixing_set: Optional[ImageFolder] = None, dataset_name: str = "cifar100"):
        super(PixMixImageFolder, self).__init__(root, transform=transform)
        self.mixing_set = mixing_set
        self.dataset_name = dataset_name

        # preprocessing used by pixmix_mixing
        mean, std = _get_mean_std(dataset_name)
        self.preprocess = {
            'tensorize': T.ToTensor(),
            'normalize': T.Normalize(mean, std)
        }


    def __getitem__(self, index):
        # path, _ = self.samples[index]
        img, label = super(PixMixImageFolder, self).__getitem__(index)  # transform applied -> tensor or PIL depending on transform
        # For mixing partner, randomly sample one from mixing_set (if provided)
        rnd_idx = np.random.choice(len(self.mixing_set))
        mixing_img, _ = self.mixing_set[rnd_idx]

        # attempt PixMix using PIL images
        mixed = pixmix_mixing(img, mixing_img, self.preprocess)

        clean = T.Compose([T.ToTensor(), T.Normalize(*_get_mean_std(self.dataset_name))])(deepcopy(img))

        return clean,  mixed,  label, index


# ----------------------
# DataLoader builders
# ----------------------
def _make_dataloader(dataset: Dataset, batch_size: int, num_workers: int, distributed: bool, shuffle: bool, drop_last: bool):
    """
    Create DataLoader; if distributed is True, wrap dataset with DistributedSampler.
    """
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=True, drop_last=drop_last)
    else:
        dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True, drop_last=drop_last)
    return dl


def get_loaders(
        train_dir: str,
        val_dir: str,
        test_dir: str,
        pixmix_dir: Optional[str],
        dataset_name: str,
        batch_size: int = 128,
        num_workers: int = 8,
        distributed: bool = False,
        model_name: str = "default",
        drop_last: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Build train/val/test DataLoaders for supported datasets.

    Args:
      train_dir, val_dir, test_dir: dataset root dirs (ImageFolder-style).
      pixmix_dir: optional path containing images for PixMix mixing set. If None, PixMix mixing will be disabled.
      dataset_name: 'cifar100' or 'imagenet1k'
      batch_size: global batch size per process (if distributed True, this is per-process batch).
      num_workers: dataloader num_workers
      distributed: whether running distributed training (uses DistributedSampler)
      model_name: reserved for model-specific changes (kept for compatibility)
      drop_last: whether to drop_last in train loader

    Returns:
      train_loader, val_loader, test_loader, num_classes
    """
    assert dataset_name in ("cifar100", "imagenet1k"), "Only cifar100 and imagenet1k are supported."

    train_t, test_t, mixing_t = _build_transforms(dataset_name)
    num_classes = 100 if dataset_name == "cifar100" else 1000

    # ------- Mixing set (for PixMix) -------
    mixing_set = ImageFolder(pixmix_dir, transform=mixing_t)

    # ------- Train dataset -------
    train_dataset = PixMixImageFolder(train_dir, transform=train_t, mixing_set=mixing_set, dataset_name=dataset_name)

    train_shuffle = not distributed
    train_loader = _make_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                    distributed=distributed, shuffle=train_shuffle, drop_last=drop_last)


    # ------- Validation & Test datasets -------
    val_dataset = SimpleImageFolder(val_dir, transform=test_t)
    test_dataset = SimpleImageFolder(test_dir, transform=test_t)
    val_loader = _make_dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers, distributed=False, shuffle=False, drop_last=False)
    test_loader = _make_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers, distributed=False, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, num_classes


# Small helper for users to adapt for distributed launch:
def set_seed(seed: int = 42):
    """Set seed for python, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
