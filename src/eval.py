'''
    Set the environment variables for distributed training before importing any packages.
    Note: It's assumed that each node has a single GPU, and the choice of RDMA interface
          specified by NCCL_IB_HCA is subject to the actual network configuration under test.
'''

import os
import sys
import glob
import subprocess

from loguru import logger
logger.remove()
logger.add(sys.stdout)

import time
import wandb
import torch
import random
import tomli_w
import pandas as pd
from typing import Any, Tuple, cast
from torch.nn import Module
import torch.distributed as dist
from torch.optim import Optimizer
from torch import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import schedule, profile, ProfilerActivity
from src.conf import parse_config_from_eval_dir, Config, SCHEDULEFREE_OPTIMS
from src.optims import get_optim, get_lr_scheduler
from src.data.preload import preload_to_local
from src.data.dataloader import get_dali_train_loader, get_dali_valid_loader, DALIWrapper, get_ffcv_train_loader, get_ffcv_valid_loader
from ffcv.loader import Loader
from src.utils import initialize_dist, gather_statistics, SmoothedValue, get_accuracy, sync_model_buffers
from src.models import load_model
from src.custom_optims.sam import SAM
from tqdm import tqdm

'''
    Functions
'''

@torch.no_grad()
def collect_bn_stats(cfg: Config, model: Any, stats_ds: DALIWrapper | Loader) -> None:
    model.train()
    for images, _ in tqdm(stats_ds):
        with torch.autocast(device_type='cuda', enabled=cfg.train.use_amp):
            model(images)

@torch.no_grad()
def valid(cfg: Config,
          model: Any,
          valid_ds: DALIWrapper | Loader,
          criterion: Module) -> Tuple[float, float, float, int]:
    model.eval()
    total_loss = 0.
    total_acc1 = 0.
    total_acc5 = 0.
    total_samples = 0

    for images, labels in tqdm(valid_ds):
        with torch.autocast(device_type='cuda', enabled=cfg.train.use_amp):
            logit = model(images)
            loss = criterion(logit, labels.view(-1))
        acc1, acc5 = get_accuracy(logit, labels, topk=(1, 5))
        total_loss += loss.item() * images.size(0)
        total_acc1 += acc1.item() * images.size(0)
        total_acc5 += acc5.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss, total_acc1, total_acc5, total_samples

def main():
    '''
        Parse the arguments and load the configurations.
    '''

    eval_dir = sys.argv[1]
    cfg = parse_config_from_eval_dir(eval_dir)
    cfg.train.reproduce.seed = 42 # set seed to 42 for reproducible evaluation

    '''
        Initialize the distributed process group.
    '''

    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.train.reproduce.seed)
    torch.cuda.manual_seed(cfg.train.reproduce.seed)
    random.seed(cfg.train.reproduce.seed)

    '''
        Load data
    '''
    if cfg.data.dataloader.name == 'dali':
        raise NotImplementedError()
        if cfg.train.preprocess.preload_local:
            preload_to_local(cfg)
        train_ds, num_batches = get_dali_train_loader(cfg)
        valid_ds = get_dali_valid_loader(cfg)
    else:
        train_ds, _ = get_ffcv_train_loader(cfg, distributed=False, batch_size=1024)
        valid_ds = get_ffcv_valid_loader(cfg, distributed=False, batch_size=1024)

    '''
        Initialize the model, optimizer, and learning rate scheduler.
    '''

    model = load_model(cfg.train.arch, num_classes=cfg.data.num_classes)
    model = model.to(memory_format=torch.channels_last) # type: ignore
    model = model.to('cuda')
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing).cuda()

    checkpoints = glob.glob(os.path.join(eval_dir, 'checkpoints', '*.pt'))
    logger.info('Checkpoints: ' + str(checkpoints))

    test_results = pd.DataFrame(columns=['epoch', 'val_loss', 'val_acc1', 'val_acc5', 'val_samples'])

    for checkpoint_dir in checkpoints:
        checkpoint = torch.load(checkpoint_dir, weights_only=True)
        modified_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('module.'):
                modified_state_dict[k[7:]] = v
            else:
                modified_state_dict[k] = v
        model.load_state_dict(modified_state_dict)
        collect_bn_stats(cfg, model, train_ds)
        val_loss, val_acc1, val_acc5, val_samples = valid(cfg, model, valid_ds, criterion)
        test_results.loc[len(test_results)] = [checkpoint['epoch'],
                                               val_loss / val_samples,
                                               val_acc1 / val_samples,
                                               val_acc5 / val_samples,
                                               val_samples]
        print(test_results)

    test_results.to_csv(os.path.join(eval_dir, 'test_results.csv'), index=False)


if __name__ == '__main__':
    main()
