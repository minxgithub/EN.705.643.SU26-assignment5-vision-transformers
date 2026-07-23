"""CIFAR-100 datasets and data loaders."""

import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(mean, std, image_size=32):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, test_transform


def get_split(targets, split_file, val_size=5000, seed=42):
    """Load the saved train/validation split, or make it once."""
    if os.path.exists(split_file):
        split = np.load(split_file)
        if "train" in split:
            return split["train"], split["val"]
        return split["train_indices"], split["validation_indices"]

    indices = np.arange(len(targets))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=targets,
    )
    folder = os.path.dirname(split_file)
    if folder:
        os.makedirs(folder, exist_ok=True)
    np.savez(split_file, train=train_indices, val=val_indices)
    return train_indices, val_indices


def get_dataloaders(data_config, batch_size, seed=42):
    """Return train, validation, and test DataLoaders."""
    root = data_config["root"]
    mean = data_config["mean"]
    std = data_config["std"]
    image_size = data_config.get("image_size", 32)
    train_transform, test_transform = get_transforms(mean, std, image_size)

    train_data = datasets.CIFAR100(
        root, train=True, download=data_config.get("download", True),
        transform=train_transform,
    )
    val_data = datasets.CIFAR100(
        root, train=True, download=data_config.get("download", True),
        transform=test_transform,
    )
    test_data = datasets.CIFAR100(
        root, train=False, download=data_config.get("download", True),
        transform=test_transform,
    )

    train_indices, val_indices = get_split(
        train_data.targets,
        data_config["split_file"],
        data_config.get("val_size", 5000),
        seed,
    )
    train_data = Subset(train_data, train_indices)
    val_data = Subset(val_data, val_indices)

    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": data_config.get("num_workers", 2),
        "pin_memory": torch.cuda.is_available(),
    }
    loaders = {
        "train": DataLoader(train_data, shuffle=True, generator=generator, **loader_options),
        "val": DataLoader(val_data, shuffle=False, **loader_options),
        "test": DataLoader(test_data, shuffle=False, **loader_options),
    }
    return loaders, test_data.classes
