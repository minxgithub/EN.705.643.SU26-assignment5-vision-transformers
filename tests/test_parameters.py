"""Tests for parameter counting and final model matching."""

import yaml

from src.models import build_model


def test_configured_models_are_parameter_matched() -> None:
    """Check that the ViT parameter count is within 10 percent of Swin."""
    with open("configs/primary.yaml", encoding="utf-8") as file:
        primary = build_model(yaml.safe_load(file)["model"])
    with open("configs/vit_baseline.yaml", encoding="utf-8") as file:
        baseline = build_model(yaml.safe_load(file)["model"])
    primary_count = sum(p.numel() for p in primary.parameters() if p.requires_grad)
    baseline_count = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    difference = abs(baseline_count - primary_count) / primary_count
    assert difference <= 0.10, (
        f"ViT differs from Swin by {difference:.1%}: "
        f"{baseline_count:,} versus {primary_count:,} parameters"
    )
