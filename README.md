# Practical 8 — Deep CNN Architectures

**Notebook:** `MODELS.ipynb`
**Companion manual:** *Deep CNN Architectures: Build, Train and Compare in PyTorch*
**Architectures:** LeNet-5 · AlexNet · VGG-16 · PlacesNet · ResNet-18
**Framework:** PyTorch · **Dataset:** CIFAR-10

## Overview

For each of five landmark CNNs the notebook does three things:

1. Builds the **full-size** network and prints the output shape and parameter count of every layer.
2. Checks selected numbers by hand: conv output size, parameter counts, receptive field and so on.
3. Trains a **scaled-down ("lite") version** on CIFAR-10 with identical training code, so the networks can be compared fairly.

A final section compares all five in one table and one chart, followed by an explanation round (written answers to six discussion questions).

## Structure

| Part | Content |
|---|---|
| Step 0 — Setup | Imports, `count_params`, `trace`, data loading, `accuracy`, `fit` |
| Part 1 — LeNet-5 | Shape trace; C1 = 456 and C3 = 2,416 parameters by hand; dense-equivalent comparison (14.5M) |
| Part 2 — AlexNet | Per-layer parameters; conv vs FC split; train AlexNet-lite |
| Part 3 — VGG-16 | Repeated 3×3 blocks; one 5×5 vs two 3×3 (27.9% saving); receptive field growth; train VGG-16-lite |
| Part 4 — PlacesNet | 365-way head; softmax and cross-entropy; receptive field per layer; train PlacesNet-lite |
| Part 5 — ResNet-18 | Skip connections (`y = F(x) + x`); gradient flow plain vs residual; train ResNet-18-lite |
| Part 6 — Final comparison | Results table and chart |
| Explanation round | Answers to six discussion questions |

## Shared training setup

| Setting | Value |
|---|---|
| Training data | first 10,000 CIFAR-10 training images |
| Test data | first 2,000 CIFAR-10 test images |
| Normalization | mean 0.5, std 0.5 per channel |
| Optimizer | Adam, lr 1e-3 |
| Batch size | 64 |
| Epochs | 4 |
| Loss | cross-entropy |
| Seed | `torch.manual_seed(0)` |

Chance level is 10%. Because of the small subset and short training, accuracies are far below full-training numbers. Compare the networks with each other, not with published results.

## Results recorded in the notebook

| Network | Parameters | Time (s) | Test accuracy |
|---|---:|---:|---:|
| LeNet-5 | 62,006 | 3 | 0.421 |
| AlexNet-lite | 543,946 | 3 | 0.475 |
| VGG-16-lite | 1,024,282 | 5 | 0.357 |
| PlacesNet-lite | 635,181 | 3 | 0.446 |
| ResNet-18-lite | 701,466 | 8 | 0.504 |

Single run, one seed, so small gaps are noise. The large gaps are what matter: ResNet-18-lite is best, and LeNet-5 is by far the smallest.

### Full-size parameter counts

| Network | Parameters | Where they sit |
|---|---:|---|
| VGG-16 | 138.4M | about 89% in FC layers |
| AlexNet | 62.4M | about 94% in FC layers |
| PlacesNet | 59.8M | same body as AlexNet, 365-way head |
| ResNet-18 | 11.7M | mostly convolutions; FC only about 0.5M |
| LeNet-5 | 62k | mostly the C5 convolution |

### Key observations

- **Residual connections help gradients.** The gradient norm at layer 1 was 3.03e-11 for the plain network and 6.14e+02 for the residual one.
- **AlexNet and PlacesNet share a body.** Their accuracy gap in the lite runs is noise, not architecture.
- **Stacked 3×3 convs are cheaper than one 5×5.** Two 3×3 layers have the same 5×5 receptive field with 27.9% fewer parameters.
- **Best for a phone:** ResNet-18-lite, which has the best accuracy and no large FC head.

## Requirements

- Python 3.9+
- `torch`, `torchvision`, `matplotlib`
- CPU works (roughly 10 minutes for all five networks on two cores). A GPU brings this to about a minute.
- Full-size VGG-16 needs about 1 GB of free RAM
- CIFAR-10 (about 170 MB) downloads automatically to `./data`

```bash
pip install torch torchvision matplotlib
```

## How to run

1. Open `MODELS.ipynb` in Jupyter, JupyterLab or Google Colab.
2. Run **Step 0 first**. Every later part reuses its helpers and appends a row to the shared `results` dictionary.
3. Run the parts in order. The final comparison reads `results`, so skipped parts will be missing from the table.
4. If a number does not match the expected output, find out why before moving on.
