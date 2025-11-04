# Reproduce Image Classification on ImageNet

## Introduction

This repository provides an unofficial reproduction of training popular image classification models on the ImageNet-1K dataset. It implements a simple and efficient training pipeline based on PyTorch, supporting multiple architectures and optimization techniques.

The repo focuses on reproducing classic machine learning workloads with clean, readable code that can be easily adapted for research and experimentation purposes.

## Supported Models

This repository supports training of several popular image classification architectures:

- ResNet-18
- ResNet-50
- VGG-16 (with and without batch normalization)
- Vision Transformer (ViT-B/16)

## Supported Optimizers

The repository implements various optimization algorithms:

- **Adam** - Adaptive moment estimation
- **SGD** - Stochastic gradient descent with momentum
- **Schedule-Free SGD** - Schedule-free variant of SGD [1]
- **Schedule-Free AdamW** - Schedule-free variant of AdamW [1]
- **SAM (Sharpness-Aware Minimization)** - Both original two-step SAM (v1) and single-step SAM (v2) [2]

## Results

For ResNet-50 trained on ImageNet using various optimizers, and without complex data augmentation (only random cropping and random horizontal flipping are used in this repo), we achieve the following top-1 accuracy:

For the experiments, models are trained using four A40 GPUs. The reproduced results are from the average of 3 runs, and the error bands represent the interval of $\pm2$ standard deviations.

| ![](./doc/resnet50_imagenet/step_vs_acc1.png) | ![](./doc/resnet50_imagenet/step_vs_acc5.png)
|:--:| :--: |
| # of Iterations vs. Top-1 Acc. | # of Iterations vs. Top-5 Acc. |

| ![](./doc/resnet50_imagenet/time_vs_acc1.png) | ![](./doc/resnet50_imagenet/time_vs_acc5.png)
|:--:| :--: |
| Training time vs. Top-1 Acc. | Training time vs. Top-5 Acc. |

The table below reports the total number of iterations, the accuracies evaluated by the trained model at the last iteration, and the total training time.

|  Epoch  | Steps  |        Optimizer         |   Top-1 Acc.    | Top-5 Acc. | Training Time (hours) |
|:-------:|:------:|:------------------------:|:---------------:|:----------:|:---------------------:|
|   90    | 112590 | Adam                     | 76.0133 ± 0.1125 | 92.9093 ± 0.0571 | 5.2771 ± 0.0026 |
|   90    | 112590 | Schedule-free SGD        | 76.7047 ± 0.0500 | 93.2113 ± 0.0698 | 5.1703 ± 0.0058 |
|   90    | 112590 | SGD                      | 77.3860 ± 0.1773 | 93.5580 ± 0.0675 | 5.1837 ± 0.0258 |
|   90    | 112590 | SAM-v1                   | 77.6467 ± 0.2698 | 93.7420 ± 0.0663 |        -        |
|   90    | 112590 | SAM-v2                   | 77.4520 ± 0.1920 | 93.5690 ± 0.0780 |        -        |

## Reproduce Experiments

> [!NOTE]
> The instructions are for Linux systems with SLURM as workload manager. Modifications may be needed for other platforms.

### Python Environment

This project uses [uv](https://docs.astral.sh/uv/) for environment management. To set up the environment:

```bash
uv sync
```

> [!NOTE]
> A list of dependencies is available at [`requirements.txt`](./requirements.txt) for reference.

### Prepare Data

#### Download ImageNet (ILSVRC 2012)

Since ImageNet requires agreement for downloading, no direct download link can be provided here.

Download the ImageNet (ILSVRC 2012) dataset from [here](https://www.image-net.org/).

Put the data under `./data/Imagenet` and arrange the data like:
```
data/Imagenet/
├── dev
│   └── ILSVRC2012_devkit_t12
├── meta.bin
├── train
│   ├── n01440764
│   ├── n01443537
│   ├── n01484850
│   ├── n01491361
│   ├── n01494475
│   ├── ...
├── val
│   ├── n01440764
│   ├── n01443537
│   ├── n01484850
│   ├── n01491361
│   ├── n01494475
│   ├── ...
```

#### DALI Dataloader

No further processing is needed for the DALI dataloader.

### Training

The experiments are conducted on a data center using Slurm as the scheduler. To run the training with four A40 GPUs:

```bash
sbatch -A <PROJECT_ACCOUNT> scripts/train/4xA40.sh $(which torchrun) config/data/imagenet.toml config/train/resnet50-adam.toml
```

Where `<PROJECT_ACCOUNT>` is your Slurm project account.

For systems not based on Slurm, you can extract the command from [`scripts/train/4xA40.sh`](./scripts/train/4xA40.sh) to run separately.

Evaluation on the validation set is performed along with training.

#### Configuration Files

The repository uses TOML configuration files to specify training parameters:

- [`config/data/imagenet.toml`](./config/data/imagenet.toml): Data configuration
- Training configurations for different models and optimizers:
  - [`config/train/resnet50-adam.toml`](./config/train/resnet50-adam.toml): ResNet-50 with Adam optimizer
  - [`config/train/resnet50-sgd.toml`](./config/train/resnet50-sgd.toml): ResNet-50 with SGD optimizer
  - [`config/train/resnet50-sf-sgd.toml`](./config/train/resnet50-sf-sgd.toml): ResNet-50 with schedule-free SGD optimizer [1]
  - [`config/train/resnet50-sgd-sam-v1.toml`](./config/train/resnet50-sgd-sam-v1.toml): ResNet-50 with SAM-v1 optimizer [2]
  - [`config/train/resnet50-sgd-sam-v2.toml`](./config/train/resnet50-sgd-sam-v2.toml): ResNet-50 with SAM-v2 optimizer [2]
  - [`config/train/resnet18-sgd.toml`](./config/train/resnet18-sgd.toml): ResNet-18 with SGD optimizer
  - [`config/train/resnet18-sgd-sam-v1.toml`](./config/train/resnet18-sgd-sam-v1.toml): ResNet-18 with SAM-v1 optimizer
  - [`config/train/resnet18-sgd-sam-v2.toml`](./config/train/resnet18-sgd-sam-v2.toml): ResNet-18 with SAM-v2 optimizer
  - [`config/train/vit_base_16.toml`](./config/train/vit_base_16.toml): Vision Transformer with SGD optimizer

See [`src/conf.py`](./src/conf.py) for detailed structure of the configuration files.

#### Training Different Models

To train different models, simply change the configuration file in the training command:

```bash
# Train ResNet-18 with SGD
sbatch -A <PROJECT_ACCOUNT> scripts/train/4xA40.sh $(which torchrun) config/data/imagenet.toml config/train/resnet18-sgd.toml

# Train Vision Transformer
sbatch -A <PROJECT_ACCOUNT> scripts/train/4xA40.sh $(which torchrun) config/data/imagenet.toml config/train/vit_base_16.toml
```

#### Training with Different Hardware Configurations

The repository provides scripts for various hardware configurations:
- [`scripts/train/4xA40.sh`](./scripts/train/4xA40.sh): 4x A40 GPUs
- [`scripts/train/4xA100.sh`](./scripts/train/4xA100.sh): 4x A100 GPUs
- [`scripts/train/4xV100.sh`](./scripts/train/4xV100.sh): 4x V100 GPUs
- [`scripts/train/8xA40.sh`](./scripts/train/8xA40.sh): 8x A40 GPUs
- [`scripts/train/8xA100.sh`](./scripts/train/8xA100.sh): 8x A100 GPUs
- [`scripts/train/8xV100.sh`](./scripts/train/8xV100.sh): 8x V100 GPUs
- [`scripts/train/16xA40.sh`](./scripts/train/16xA40.sh): 16x A40 GPUs
- [`scripts/train/16xA100.sh`](./scripts/train/16xA100.sh): 16x A100 GPUs
- [`scripts/train/32xT4.sh`](./scripts/train/32xT4.sh): 32x T4 GPUs
- [`scripts/train/local.sh`](./scripts/train/local.sh): Local training (single machine)

## References

[1] Defazio, Aaron, et al. "The road less scheduled." arXiv preprint arXiv:2405.15682 (2024). [Repo Link](https://github.com/facebookresearch/schedule_free).

[2] Foret, Pierre, et al. "Sharpness-aware minimization for efficiently improving generalization." arXiv preprint arXiv:2010.01412 (2020).
