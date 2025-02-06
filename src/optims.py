import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, SGD
from torch.optim.lr_scheduler import LRScheduler
from src.conf import Config, AdamConfig, SGDConfig, SGDScheduleFreeConfig, \
    CosineLRSchedulerConfig, AdamWScheduleFreeConfig, SGDSAMConfig
from schedulefree import SGDScheduleFree, AdamWScheduleFree
from src.custom_optims.sam import SAM, SAMV1, S2SAM
from typing import Callable, List, Tuple
from functools import partial

def get_param_groups(model: nn.Module, weight_decay: float) -> list:
    non_decay_params = [v for n, v in model.named_parameters() if ('bn' in n) or ('bias' in n)]
    decay_params = [v for n, v in model.named_parameters() if not (('bn' in n) or ('bias' in n))]
    return [
        {"params": non_decay_params, "weight_decay": 0},
        {"params": decay_params, "weight_decay": weight_decay}
    ]

def get_param_groups_from_list(params: List[Tuple[torch.Tensor, str]], weight_decay: float) -> list:
    non_decay_params = [v for v, n in params if ('bn' in n) or ('bias' in n)]
    decay_params = [v for v, n in params if not (('bn' in n) or ('bias' in n))]
    return [
        {"params": non_decay_params, "weight_decay": 0},
        {"params": decay_params, "weight_decay": weight_decay}
    ]

def get_params_list(model: nn.Module) -> List[Tuple[torch.Tensor, str]]:
    return [(v, n) for n, v in model.named_parameters()]

def get_optim_fn(cfg: Config, num_steps_per_epoch: int) -> Callable[[List[Tuple[torch.Tensor, str]]], Optimizer]:
    optim_cfg = cfg.train.optim
    match optim_cfg.name.lower():
        case "adam":
            assert isinstance(optim_cfg, AdamConfig)
            def adam_fn(params: List[Tuple[torch.Tensor, str]],
                        lr: float,
                        betas: Tuple[float, float],
                        eps: float = optim_cfg.epsilon) -> Optimizer:
                return Adam(get_param_groups_from_list(params, optim_cfg.weight_decay),
                            lr=lr,
                            betas=betas,
                            eps=eps)
            return partial(adam_fn, lr=cfg.train.lr, betas=(optim_cfg.beta1, optim_cfg.beta2), eps=optim_cfg.epsilon)
        case "sgd":
            assert isinstance(optim_cfg, SGDConfig)
            def sgd_fn(params: List[Tuple[torch.Tensor, str]],
                       lr: float,
                       momentum: float) -> Optimizer:
                return SGD(get_param_groups_from_list(params, optim_cfg.weight_decay),
                           lr=lr,
                           momentum=momentum)
            return partial(sgd_fn, lr=cfg.train.lr, momentum=optim_cfg.momentum)
        case "sgd-schedule-free":
            assert isinstance(optim_cfg, SGDScheduleFreeConfig)
            def sgd_schedule_free_fn(params: List[Tuple[torch.Tensor, str]],
                                     lr: float,
                                     momentum: float,
                                     weight_decay: float,
                                     warmup_steps: int,
                                     r: float,
                                     weight_lr_power: float) -> Optimizer:
                return SGDScheduleFree(get_param_groups_from_list(params, weight_decay),
                                       lr=lr,
                                       momentum=momentum,
                                       weight_decay=weight_decay,
                                       warmup_steps=warmup_steps,
                                       r=r,
                                       weight_lr_power=weight_lr_power)
            return partial(sgd_schedule_free_fn,
                           lr=cfg.train.lr,
                           momentum=optim_cfg.momentum,
                           weight_decay=optim_cfg.weight_decay,
                           warmup_steps=num_steps_per_epoch * optim_cfg.warmup_epochs,
                           r=optim_cfg.r,
                           weight_lr_power=optim_cfg.weight_lr_power)
        case "adamw-schedule-free":
            assert isinstance(optim_cfg, AdamWScheduleFreeConfig)
            def adamw_schedule_free_fn(params: List[Tuple[torch.Tensor, str]],
                                       lr: float,
                                       betas: Tuple[float, float],
                                       eps: float,
                                       weight_decay: float,
                                       warmup_steps: int,
                                       r: float,
                                       weight_lr_power: float) -> Optimizer:
                return AdamWScheduleFree(get_param_groups_from_list(params, weight_decay),
                                         lr=lr,
                                         betas=betas,
                                         eps=eps,
                                         weight_decay=weight_decay,
                                         warmup_steps=warmup_steps,
                                         r=r,
                                         weight_lr_power=weight_lr_power)
            return partial(adamw_schedule_free_fn,
                           lr=cfg.train.lr,
                           betas=(optim_cfg.beta1, optim_cfg.beta2),
                           eps=optim_cfg.epsilon,
                           weight_decay=optim_cfg.weight_decay,
                           warmup_steps=num_steps_per_epoch * optim_cfg.warmup_epochs,
                           r=optim_cfg.r,
                           weight_lr_power=optim_cfg.weight_lr_power)
        case "sgd-sam":
            assert isinstance(optim_cfg, SGDSAMConfig)
            def sgd_sam_fn(params: List[Tuple[torch.Tensor, str]],
                           lr: float,
                           momentum: float,
                           rho: float,
                           adaptive: bool,
                           v2: bool) -> Optimizer:
                if not v2:
                    return SAMV1(get_param_groups_from_list(params, optim_cfg.weight_decay),
                                 base_optimizer=SGD,
                                 lr=lr,
                                 momentum=momentum,
                                 rho=rho)
                else:
                    return S2SAM(get_param_groups_from_list(params, optim_cfg.weight_decay),
                                 base_optimizer=SGD,
                                 lr=lr,
                                 momentum=momentum,
                                 rho=rho)
            return partial(sgd_sam_fn,
                           lr=cfg.train.lr,
                           momentum=optim_cfg.momentum,
                           rho=optim_cfg.rho,
                           adaptive=optim_cfg.adaptive,
                           v2=optim_cfg.v2)
        case _:
            raise ValueError(f"Unknown optimizer: {cfg.train.optim.name}")

def get_optim(cfg: Config, model: nn.Module, num_steps_per_epoch: int) -> Optimizer:
    return get_optim_fn(cfg, num_steps_per_epoch)(get_params_list(model))

def get_lr_scheduler_fn(cfg: Config, num_steps_per_epoch: int) -> Callable[[Optimizer], LRScheduler]:
    lr_scheduler_cfg = cfg.train.lr_scheduler
    match lr_scheduler_cfg.name.lower():
        case "cosine":
            assert isinstance(lr_scheduler_cfg, CosineLRSchedulerConfig)
            def cosine_fn(optim: Optimizer,
                          warmup_epochs: int,
                          warmup_decay: float,
                          max_epochs: int,
                          eta_min: float) -> LRScheduler:
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optim,
                    start_factor=warmup_decay,
                    total_iters=num_steps_per_epoch * warmup_epochs
                )
                main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim,
                    T_max=num_steps_per_epoch * (max_epochs - warmup_epochs),
                    eta_min=eta_min
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    optim,
                    schedulers=[warmup_lr_scheduler, main_scheduler],
                    milestones=[num_steps_per_epoch * warmup_epochs]
                )
            return partial(cosine_fn,
                           warmup_epochs=lr_scheduler_cfg.warmup_epochs,
                           warmup_decay=lr_scheduler_cfg.warmup_decay,
                           max_epochs=cfg.train.max_epochs,
                           eta_min=lr_scheduler_cfg.eta_min)
        case "constant":
            def constant_fn(optim: Optimizer) -> LRScheduler:
                return torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1)
            return constant_fn
        case _:
            raise ValueError(f"Unknown LR scheduler: {cfg.train.lr_scheduler.name}")

def get_lr_scheduler(cfg: Config, optim: Optimizer, num_steps_per_epoch: int) -> LRScheduler:
    return get_lr_scheduler_fn(cfg, num_steps_per_epoch)(optim)

