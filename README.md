# EN.705.643 Deep Learning Developments with PyTorch
### Assignment 5: Locality and Hierarchy in Vision Transformers
##### Min Xie

## Project overview

This project compares a Swin Transformer with a parameter-matched plain Vision
Transformer (ViT) on CIFAR-100. Both models use torchvision, start from random
weights, and share the same data split and training controls.

## Project Github repository

https://github.com/minxgithub/EN.705.643.SU26-assignment5-vision-transformers.git

## Environment Setup

1. **Python Version:**
This project requires Python 3.10 or higher.

2. **Dependencies:**
Install the required Python packages by running:

```bash
pip install -r requirements.txt
wandb login
```

Weights & Biases is used for experiment tracking and confusion matrices.
Complete epoch logs are also saved locally as CSV files.

## Dataset

The pipeline downloads CIFAR-100 through `torchvision.datasets.CIFAR100`.
The official 50,000-image training set is divided into 45,000 training and
5,000 validation images using a stratified split with seed 42. The official
10,000-image test set is unchanged.

The saved split is:

```text
data_splits/cifar100_split_seed42.npz
```

The Colab configs use `/content/datasets` as the dataset root. Change `data.root`
in both YAML files when running elsewhere. If the dataset is absent and
`download: true`, torchvision downloads it automatically.

Training uses random resized cropping, horizontal flipping, RandAugment,
Random Erasing, and batch-level CutMix. Validation and test transforms are
deterministic.

## Train

Train the primary Swin Transformer:

```bash
python -m src.train --config configs/primary.yaml
```

Train the parameter-matched ViT baseline:

```bash
python -m src.train --config configs/vit_baseline.yaml
```

Each run trains for at most 400 epochs with AdamW, learning-rate warmup, cosine
decay, gradient clipping, and early stopping. The checkpoint with the highest
validation accuracy is retained.

Training outputs:

```text
checkpoints/swin_best.pt
checkpoints/vit_best.pt
logs/primary_training.csv
logs/vit_training.csv
```

## Evaluate

Evaluate the best Swin checkpoint on validation data:

```bash
python -m src.evaluate \
  --config configs/primary.yaml \
  --checkpoint checkpoints/swin_best.pt \
  --split val \
  --output results/swin-primary_val.json
```

Evaluate it once on the final test set:

```bash
python -m src.evaluate \
  --config configs/primary.yaml \
  --checkpoint checkpoints/swin_best.pt \
  --split test \
  --output results/swin-primary_test.json
```

Replace the config and checkpoint with `configs/vit_baseline.yaml` and
`checkpoints/vit_best.pt` to evaluate ViT.

Evaluation reports top-1 accuracy, macro precision, macro recall, macro F1,
weighted F1, per-class metrics, one-vs-rest macro ROC-AUC, and a confusion
matrix. JSON results are written to the path supplied with `--output`.

## Figures and outputs

- Training curves can be reproduced from the CSV logs or exported from W&B.
- Confusion matrices are logged to W&B during evaluation.
- Evaluation automatically saves the 12 most confident incorrect predictions
  under `figures/`.
- Complete per-class metrics and common confusion patterns are stored in the
  evaluation JSON files.

## Automated tests

From the repository root, run:

```bash
python -m pytest tests -v
```

The tests cover dataset-split disjointness, model output shape, and parameter
matching.

## Hardware and runtime

Final training was performed in Google Colab on an NVIDIA A100 GPU using FP32.
Each epoch took approximately 23 seconds, so a 400-epoch run required about
2.5 hours per model. Runtime can vary with hardware and data-loading speed.

## Known limitations

- Experiments used one dataset split and one random seed.
- Both models were trained from scratch on only 45,000 unique training images.
- The comparison estimates attention-score cost rather than complete FLOPs.
- The final Swin model reached 69.97% test accuracy, three correct predictions
  below the 70% target.
