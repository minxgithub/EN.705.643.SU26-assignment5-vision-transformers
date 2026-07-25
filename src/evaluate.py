"""Evaluate a model on a validation or test set."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import nn
from tqdm import tqdm

from src.data import get_dataloaders
from src.metrics import calculate_metrics
from src.models import build_model
from src.utils import get_device, load_config, seed_everything


def evaluate(model, dataloader, criterion, device, collect_errors=False):  # pylint: disable=too-many-locals
    """Evaluate a model and optionally collect its misclassified examples."""
    model.eval()
    total_loss = 0.0
    probabilities = []
    targets = []
    errors = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item() * labels.size(0)
            batch_probabilities = torch.softmax(output, dim=1).cpu()
            probabilities.extend(batch_probabilities.tolist())
            targets.extend(labels.cpu().tolist())

            if collect_errors:
                batch_predictions = batch_probabilities.argmax(dim=1)
                for index in range(labels.size(0)):
                    target = labels[index].item()
                    prediction = batch_predictions[index].item()
                    if prediction != target:
                        errors.append({
                            "image": images[index].cpu(),
                            "true": target,
                            "predicted": prediction,
                            "probability": batch_probabilities[index, prediction].item(),
                        })

    predictions = np.array(probabilities).argmax(axis=1)
    correct = sum(predictions == np.array(targets))
    loss = total_loss / len(targets)
    accuracy = correct / len(targets)
    return loss, accuracy, probabilities, targets, errors


def save_error_examples(errors, class_names, mean, std, output_file):
    """Save the 12 most confident incorrect predictions."""
    selected = sorted(errors, key=lambda item: item["probability"], reverse=True)[:12]
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))

    for axis, error in zip(axes.flat, selected):
        image = (error["image"] * std + mean).clamp(0, 1)
        axis.imshow(image.permute(1, 2, 0))
        axis.set_title(
            f"True: {class_names[error['true']]}\n"
            f"Pred: {class_names[error['predicted']]} "
            f"({error['probability']:.2f})",
            fontsize=8,
        )
        axis.axis("off")

    for axis in axes.flat[len(selected):]:
        axis.axis("off")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def main():  # pylint: disable=too-many-locals
    """Load a saved checkpoint and evaluate it on validation or test data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])
    device = get_device()
    loaders, class_names = get_dataloaders(
        config["data"], config["training"]["batch_size"], config["seed"]
    )
    model = build_model(config["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    wandb.init(project=config["wandb_project"], name=config["name"] + "-evaluation")
    loss, acc, probabilities, targets, errors = evaluate(  # pylint: disable=unused-variable
        model, loaders[args.split], nn.CrossEntropyLoss(), device, collect_errors=True
    )
    metrics = calculate_metrics(probabilities, targets, class_names)
    predictions = np.array(probabilities).argmax(axis=1).tolist()
    error_figure = f"figures/{config['name']}_{args.split}_errors.png"
    save_error_examples(
        errors,
        class_names,
        config["data"]["mean"],
        config["data"]["std"],
        error_figure,
    )

    wandb_results = {
        f"{args.split}/loss": loss,
        **{
            f"{args.split}/{name}": value
            for name, value in metrics.items()
            if isinstance(value, float)
        },
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=predictions, y_true=targets, class_names=class_names
        ),
        "error_examples": wandb.Image(error_figure),
    }
    wandb.log(wandb_results)

    output = args.output or f"results/{config['name']}_{args.split}.json"
    output_folder = os.path.dirname(output)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump({"loss": loss, **metrics}, file, indent=2)

    print(f"{args.split} loss: {loss:.4f}")
    print(f"top-1 accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"macro F1: {metrics['macro_f1']:.4f}")
    print(f"results saved to {output}")
    print(f"error examples saved to {error_figure}")
    wandb.finish()


if __name__ == "__main__":
    main()
