"""Train either the Swin Transformer or ViT."""

import argparse
import csv
import os
import time

import torch
import wandb
from torch import nn
from tqdm import tqdm

from src.data import get_dataloaders
from src.evaluate import evaluate
from src.models import build_model
from src.utils import get_device, load_config, seed_everything


def train_batch(model, images, labels, criterion, optimizer, clip_norm):
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    correct = (output.argmax(1) == labels).sum().item()
    return loss.item(), correct


def train_epoch(model, dataloader, criterion, optimizer, device, clip_norm):
    model.train()
    total_loss = 0.0
    total_correct = 0

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        loss, correct = train_batch(
            model, images, labels, criterion, optimizer, clip_norm
        )
        total_loss += loss * labels.size(0)
        total_correct += correct

    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)


def train(model, loaders, config, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=config["warmup_epochs"]
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"] - config["warmup_epochs"]
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config["warmup_epochs"]],
    )
    os.makedirs(os.path.dirname(config["checkpoint"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["log_file"]), exist_ok=True)
    log_columns = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rate",
        "epoch_duration_seconds",
    ]
    with open(config["log_file"], "w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=log_columns).writeheader()

    best_accuracy = -1.0
    epochs_without_improvement = 0

    for epoch in range(config["epochs"]):
        start_time = time.time()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            config["gradient_clip_norm"],
        )
        val_loss, val_accuracy, _, _, _ = evaluate(
            model, loaders["val"], criterion, device
        )
        duration = time.time() - start_time
        scheduler.step()

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": learning_rate,
            "epoch_duration_seconds": duration,
        }
        with open(config["log_file"], "a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=log_columns).writerow(epoch_log)

        wandb.log(epoch_log)
        print(
            f"Epoch {epoch + 1}/{config['epochs']} - "
            f"train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, "
            f"val accuracy: {val_accuracy:.4f}, time: {duration:.1f}s"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config["checkpoint"])
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["patience"]:
            print(
                f"Early stopping: validation accuracy did not improve for "
                f"{config['patience']} epochs."
            )
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])
    device = get_device()
    loaders, _ = get_dataloaders(
        config["data"], config["training"]["batch_size"], config["seed"]
    )
    model = build_model(config["model"]).to(device)

    wandb.init(project=config["wandb_project"], name=config["name"], config=config)
    train(model, loaders, config["training"], device)
    wandb.finish()


if __name__ == "__main__":
    main()
