"""Tests for CIFAR-100 splitting and evaluation transforms."""

import numpy as np
import torch
from PIL import Image

from src.data import get_split, get_transforms


def test_split(tmp_path):
    """Check split sizes, class balance, and train-validation disjointness."""
    targets = np.repeat(np.arange(100), 500)
    train, val = get_split(targets, tmp_path / "split.npz", seed=42)

    assert len(train) == 45000
    assert len(val) == 5000
    assert len(np.intersect1d(train, val)) == 0
    assert np.all(np.bincount(targets[val]) == 50)


def test_validation_transform_is_not_random():
    """Check that the evaluation transform produces deterministic results."""
    _, transform = get_transforms([0.5] * 3, [0.5] * 3)
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    assert torch.equal(transform(image), transform(image))
