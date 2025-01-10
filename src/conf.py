import os
import argparse
import tomllib
from typing import Literal, Optional, Union, Final
from typing_extensions import TypeAlias
from pydantic import BaseModel, Field, computed_field, ConfigDict

PROJECT_DIR: Final[str] = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

class _BaseModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class FFCVConfig(_BaseModel):
    name: Literal['ffcv'] = 'ffcv'
    processed_data_dir: str = Field(default=os.path.join(PROJECT_DIR, 'data/ffcv'))
    max_resolution: int = Field(default=384)
    compress_probability: float = Field(default=1.0)
    jpeg_quality: int = Field(default=90)
    num_data_workers: int = Field(default=12)
    in_memory: bool = Field(default=True)

    @computed_field
    @property
    def tag(self) -> str:
        return f'ffcv_{self.max_resolution}_{self.compress_probability:.3f}_{self.jpeg_quality}'

    @computed_field
    @property
    def train_data_dir(self) -> str:
        return os.path.join(self.processed_data_dir, self.tag + '_train.ffcv')

    @computed_field
    @property
    def val_data_dir(self) -> str:
        return os.path.join(self.processed_data_dir, self.tag + '_val.ffcv')

class DaliConfig(_BaseModel):
    name: Literal['dali'] = 'dali'
    preload: bool = Field(default=False)
    sharded_data_dir: str = Field(default='./data/Imagenet-sharded')
    num_data_workers: int = Field(default=4)

ALL_DATALOADERS: TypeAlias = Union[FFCVConfig, DaliConfig]

class Data(_BaseModel):
    data_dir: str = Field(default='./data/Imagenet')
    dataloader: ALL_DATALOADERS = Field(default_factory=FFCVConfig,
                                        discriminator='name')
    num_classes: int = Field(default=1000)

class Preprocess(_BaseModel):
    preload_local: bool = Field(default=False)
    interpolation: str = Field(default='bilinear')
    train_crop_size: int = Field(default=176)
    val_image_size: int = Field(default=256)
    val_crop_size: int = Field(default=224)

class AdamConfig(_BaseModel):
    name: Literal['adam'] = 'adam'
    weight_decay: float = Field(default=1e-4)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)
    epsilon: float = Field(default=1e-8)

class SGDConfig(_BaseModel):
    name: Literal['sgd'] = 'sgd'
    weight_decay: float = Field(default=1e-4)
    momentum: float = Field(default=0.875)

class SGDScheduleFreeConfig(_BaseModel):
    name: Literal['sgd-schedule-free'] = 'sgd-schedule-free'
    warmup_epochs: int = Field(default=5)
    weight_decay: float = Field(default=1e-4)
    momentum: float = Field(default=0.9)
    r: float = Field(default=0.0)
    weight_lr_power: float = Field(default=2.0)
    num_samples_for_stats: int = Field(default=102400)

class AdamWScheduleFreeConfig(_BaseModel):
    name: Literal['adamw-schedule-free'] = 'adamw-schedule-free'
    warmup_epochs: int = Field(default=5)
    weight_decay: float = Field(default=1e-1)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)
    epsilon: float = Field(default=1e-8)
    r: float = Field(default=0.0)
    weight_lr_power: float = Field(default=2.0)
    num_samples_for_stats: int = Field(default=102400)

ALL_OPTIMS: TypeAlias = Union[AdamConfig, SGDConfig, SGDScheduleFreeConfig, AdamWScheduleFreeConfig]
SCHEDULEFREE_OPTIMS = [SGDScheduleFreeConfig, AdamWScheduleFreeConfig]

class CosineLRSchedulerConfig(_BaseModel):
    name: Literal['cosine'] = Field(default='cosine')
    warmup_epochs: int = Field(default=5)
    warmup_decay: float = Field(default=0.01)


class ConstantLRSchedulerConfig(_BaseModel):
    name: Literal['constant'] = Field(default='constant')


ALL_LR_SCHEDULERS: TypeAlias = Union[CosineLRSchedulerConfig, ConstantLRSchedulerConfig]


class Reproduce(_BaseModel):
    seed: int = Field(default=810975)


class Log(_BaseModel):
    log_freq: int = Field(default=100)
    wandb_on: bool = Field(default=True)
    wandb_project: str = Field(default='reproduce_resnet')
    checkpoint_freq: int = Field(default=45)

    @computed_field
    @property
    def job_id(self) -> str:
        return os.environ.get('JOB_ID', '0') if 'JOB_ID' in os.environ else \
            os.environ.get('SLURM_JOB_ID', '0')

    @computed_field
    @property
    def log_dir(self) -> str:
        return os.path.join(PROJECT_DIR, 'log', self.job_id)


class Network(_BaseModel):
    @computed_field(repr=False)
    @property
    def world_size(self) -> int:
        return int(os.environ.get('WORLD_SIZE', '1'))

    @computed_field(repr=False)
    @property
    def rank(self) -> int:
        return int(os.environ.get('RANK', '0'))
    
    @computed_field(repr=False)
    @property
    def local_rank(self) -> int:
        return int(os.environ.get('LOCAL_RANK', '0'))
    
    @computed_field(repr=False)
    @property
    def local_world_size(self) -> int:
        return int(os.environ.get('LOCAL_WORLD_SIZE', '1'))

    @computed_field(repr=False)
    @property
    def node_list(self) -> str:
        return os.environ.get('SLURM_NODELIST', 'localhost')


class Train(_BaseModel):
    batch_size: int = Field(default=1024)
    max_epochs: int = Field(default=90)
    lr: float = Field(default=0.001)
    label_smoothing: float = Field(default=0.1)
    grad_clip_norm: float = Field(default=0.0)
    checkpoint_dir: str = Field(default="")
    arch: str = Field(default='resnet50')
    use_amp: bool = Field(default=True)
    preprocess: Preprocess = Field(default_factory=Preprocess)
    optim: ALL_OPTIMS = Field(default_factory=AdamConfig, discriminator='name')
    lr_scheduler: ALL_LR_SCHEDULERS = Field(default_factory=CosineLRSchedulerConfig, discriminator='name')
    reproduce: Reproduce = Field(default_factory=Reproduce)
    log: Log = Field(default_factory=Log)
    network: Network = Field(default_factory=Network)

    @computed_field(repr=False)
    @property
    def batch_size_per_local_batch(self) -> int:
        return self.batch_size // self.network.world_size


class Config(_BaseModel):
    data: Data = Field(default_factory=Data)
    train: Train = Field(default_factory=Train)


def parse_config() -> Config:
    def _load_toml(config_dir):
        with open(config_dir, 'rb') as f:
            config = tomllib.load(f)
        return config

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-cfg', type=str, required=True)
    parser.add_argument('--train-cfg', type=str, required=True)
    args = parser.parse_args()
    cfg = {
        'data': _load_toml(args.data_cfg),
        'train': _load_toml(args.train_cfg)
    }
    return Config.model_validate(cfg)
