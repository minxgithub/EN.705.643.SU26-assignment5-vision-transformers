"""A few shared helper functions."""

import random

import numpy as np
import torch
import yaml


def load_config(path):
    """Load and return a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def seed_everything(seed):
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return a CUDA device when available, otherwise return the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
