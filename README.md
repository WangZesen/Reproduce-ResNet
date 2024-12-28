# Reproduce Image Classification on ImageNet

## Introduction

This repo is an unofficial reproduction of training ResNet-50 on ImageNet which is one of the most classic machine learning workloads based on PyTorch 2.3.0 (latest stable version by the time of setting up this repo). The repo provides a simple and efficient implement, which could be easiliy adapted for other optimization purpose.

## Results

For ResNet-50 trained on ImageNet using Adam optimizer, and without complex data augmentation (only random cropping and random horizontal flipping are used in this repo), the top-1 accuracy is roughly 76%.

For the experiment, the model is trained by four A40 GPUs. The reproduced results are from the average of 3 runs and the error bands represnet the interval of $\pm2$ standard deviations.

| ![](./doc/resnet50_imagenet/step_vs_acc1.png) | ![](./doc/resnet50_imagenet/step_vs_acc5.png)
|:--:| :--: |
| # of Iterations vs. Top-1 Acc. | # of Iterations vs. Top-5 Acc. |

| ![](./doc/resnet50_imagenet/time_vs_acc1.png) | ![](./doc/resnet50_imagenet/time_vs_acc5.png)
|:--:| :--: |
| Training time vs. Top-1 Acc. | Training time vs. Top-5 Acc. |

The table below reports the total number of iterations, the accuracies evaluated by the trained model at the last iteration, and the total training time.

|  Epoch  | Steps |        Optimizer         |   Top-1 Acc.    | Top-5 Acc. | Training Time (hours) |
|:------:|:---:|:------------------:|:----:|:---------------:|:---------------------:|
| 90 | 112590 | Adam | 76.0133 ± 0.1125 |    92.9093 ± 0.0571    | 5.2771 ± 0.0026 |
| 90 | 112590 | Schedule-free SGD | 76.7047 ± 0.0500 |    93.2113 ± 0.0698    | 5.1703 ± 0.0058 |


## Reproduce Experiments

> [!NOTE]
> The instruction is for Linux systems with SLURM as workload manager. It may be subject to change for other platforms.

### Python Environment

Create a virtual environment using Conda. Activate the environment and install the dependencies by

```bash
# install latest version of pytorch: see https://pytorch.org/get-started/locally/ for other platforms
pip install torch torchvision torchaudio
# install dali with cuda-11.0 as it comes with cuda dependencies
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110
# install ffcv
conda install cupy pkg-config libjpeg-turbo opencv numba -c conda-forge -c pytorch
pip install ffcv
# install schedule-free optimizer
pip install schedulefree
# install other dependencies
pip install wandb seaborn loguru scipy tqdm tomli-w pydantic
```

One has to login to wandb for uploading the metrics before runing the experiments (wandb logging is on by default).
```
wandb login
```

### Prepare Data

#### Download ImageNet (ILSVRC 2012)

Since it needs to sign an agreement for downloading the dataset, no direct download link to the processed dataset can be provided here.

Download the ImageNet (ILSVRC 2012) dataset from [here](https://www.image-net.org/).

Put the data under `./data/Imagenet` and arrage the data like
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

#### ffcv Dataloader (Recommend)

Use ffcv dataloader to load the whole dataset into the memory to avoid disk IO bottleneck. It's recommended to use ffcv if there is >=64GB memory per GPU.

Create folder for processed dataset
```bash
mkdir ./data/ffcv/
```

Transform the dataset into bento format by
```bash
python -m src.ffcv_writer --data-cfg config/data/imagenet.toml
```

It will generate the processed datasets for training and validation under `./data/ffcv`. The total size is around 66.9GB. The preprocess configurations are 1. `500` pixel as max resolution, 2. `90` as JPG quality, and 3. all images are compressed as JPG. Please see [ffcv documentation](https://docs.ffcv.io/benchmarks.html#dataset-sizes) for details.

#### DALI Dataloader

No further processes need to be done for DALI dataloader.

### Train

The experiments are conducted on a data center using Slurm as the scheduler. To run the training with four A40 GPUs, 

```
sbatch -A <PROJECT_ACCOUNT> scripts/train/4xA40.sh $(which torchrun) config/data/imagenet.toml config/train/resnet50-adam.toml
```
where `<PROJECT_ACCOUNT>` is the slurm project account.

One can extract the command in [`scripts/train/4xA40.sh`](./scripts/train/4xA40.sh) to run seperately if the system is not based on slurm.

The evaulation on the validation set is done along with the training.

#### Configuration files

- [`config/data/imagenet.toml`](./config/data/imagenet.toml): data configuration
- [`config/train/resnet-adam.toml`](./config/train/resnet50-adam.toml): training configuration for ResNet-50 with Adam optimizer
- [`config/train/resnet-sf-sgd.toml`](./config/train/resnet50-sf-sgd.toml): training configuration for ResNet-50 with schedule-free SGD optimizer [1]

See [`src/conf.py`](./src/conf.py) for detailed structure of the configuration files.

## Reference

[1] Defazio, Aaron, et al. "The road less scheduled." arXiv preprint arXiv:2405.15682 (2024). [Repo Link](https://github.com/facebookresearch/schedule_free).
