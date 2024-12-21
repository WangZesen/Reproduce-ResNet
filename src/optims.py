import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, SGD, AdamW
from torch.optim.lr_scheduler import LRScheduler
from src.conf import Config, AdamConfig, SGDConfig

def get_params(model: nn.Module, weight_decay: float) -> list:
    bn_params = [v for n, v in model.named_parameters() if ('bn' in n) or ('bias' in n)]
    rest_params = [v for n, v in model.named_parameters() if not (('bn' in n) or ('bias' in n))]
    return [
        {"params": bn_params, "weight_decay": 0},
        {"params": rest_params, "weight_decay": weight_decay}
    ]

def get_optim(cfg: Config, model: nn.Module) -> Optimizer:
    optim_cfg = cfg.train.optim
    match optim_cfg.name.lower():
        case "adam":
            assert isinstance(optim_cfg, AdamConfig)
            return Adam(get_params(model, optim_cfg.weight_decay),
                        lr=cfg.train.lr,
                        betas=(optim_cfg.beta1,optim_cfg.beta2),
                        eps=optim_cfg.epsilon)
        case "sgd":
            assert isinstance(optim_cfg, SGDConfig)
            return SGD(get_params(model, optim_cfg.weight_decay),
                       lr=cfg.train.lr,
                       momentum=optim_cfg.momentum)
        case _:
            raise ValueError(f"Unknown optimizer: {cfg.train.optim.name}")

def get_lr_scheduler(cfg: Config, optim: Optimizer, num_steps_per_epoch: int) -> LRScheduler:
    match cfg.train.lr_scheduler.name.lower():
        case "cosine":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=cfg.train.lr_scheduler.warmup_decay,
                total_iters=num_steps_per_epoch * cfg.train.lr_scheduler.warmup_epochs
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                T_max=num_steps_per_epoch * (cfg.train.max_epochs - cfg.train.lr_scheduler.warmup_epochs),
                eta_min=1e-5
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optim,
                schedulers=[warmup_lr_scheduler, main_scheduler],
                milestones=[num_steps_per_epoch * cfg.train.lr_scheduler.warmup_epochs]
            )
        case _:
            raise ValueError(f"Unknown LR scheduler: {cfg.train.lr_scheduler.name}")

