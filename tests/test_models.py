"""Basic tests for the torchvision model factory."""

import pytest
import torch

from src.models import build_model


@pytest.mark.parametrize(
    "config",
    [
        {
            "name": "vit",
            "image_size": 32,
            "patch_size": 4,
            "num_classes": 100,
            "embed_dim": 96,
            "depth": 2,
            "num_heads": 3,
            "dropout": 0.1,
        },
        {
            "name": "swin",
            "patch_size": 2,
            "window_size": 4,
            "num_classes": 100,
            "embed_dim": 24,
            "depths": [2, 2, 2, 2],
            "num_heads": [2, 4, 8, 16],
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "stochastic_depth": 0.1,
        },
    ],
)
def test_model_output_shape(config):
    """Check that each model produces 100 logits per input image."""
    model = build_model(config).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 100)


def test_unknown_model_raises_error() -> None:
    """Check that an unknown model name raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown"):
        build_model({"name": "unknown"})
