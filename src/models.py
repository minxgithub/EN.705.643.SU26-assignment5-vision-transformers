"""The two torchvision models used in the experiment."""

from torchvision.models import SwinTransformer, VisionTransformer


def build_model(config):
    """Build the configured torchvision Swin Transformer or Vision Transformer."""
    name = config["name"]

    if name == "swin":
        return SwinTransformer(
            patch_size=[config["patch_size"], config["patch_size"]],
            embed_dim=config["embed_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            window_size=[config["window_size"], config["window_size"]],
            mlp_ratio=config["mlp_ratio"],
            dropout=config["dropout"],
            stochastic_depth_prob=config["stochastic_depth"],
            num_classes=config["num_classes"],
        )

    if name == "vit":
        return VisionTransformer(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            num_layers=config["depth"],
            num_heads=config["num_heads"],
            hidden_dim=config["embed_dim"],
            mlp_dim=config["embed_dim"] * 4,
            dropout=config["dropout"],
            attention_dropout=config["dropout"],
            num_classes=config["num_classes"],
        )

    raise ValueError(f"Unknown model: {name}")
