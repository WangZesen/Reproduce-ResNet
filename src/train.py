'''
    Set the environment variables for distributed training before importing any packages.
    Note: It's assumed that each node has a single GPU, and the choice of RDMA interface
          specified by NCCL_IB_HCA is subject to the actual network configuration under test.
'''

import os
import sys
import subprocess

rank = int(os.environ.get('RANK', 0))
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
if local_world_size > 1:
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    assert len(devices) == local_world_size, 'Each process must have a single GPU.'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[local_rank]
gpu_models = subprocess.check_output('nvidia-smi --query-gpu=name --format=csv,noheader', shell=True).decode('utf-8').split('\n')
if 'Tesla T4' in gpu_models:
    os.environ['NCCL_IB_HCA'] = '=mlx5_0:1'

from loguru import logger
logger.remove()
logger.add(sys.stdout)

import time
import wandb
import torch
import neptune
from neptune import Run
from neptune.integrations.python_logger import NeptuneHandler
import random
import tomli_w
import pandas as pd
from typing import Any, Tuple, cast, Optional
from torch.nn import Module
import torch.distributed as dist
from torch.optim import Optimizer
from torch import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import schedule, profile, ProfilerActivity
from src.conf import parse_config, Config, SCHEDULEFREE_OPTIMS, SGDSAMConfig
from src.optims import get_optim, get_lr_scheduler
from src.data.preload import preload_to_local
from src.data.dataloader import get_dali_train_loader, get_dali_valid_loader, DALIWrapper, get_ffcv_train_loader, get_ffcv_valid_loader
from ffcv.loader import Loader
from src.utils import initialize_dist, gather_statistics, SmoothedValue, get_accuracy, sync_model_buffers
from src.models import load_model
from src.custom_optims.sam import SAM, SAMV1, S2SAM

'''
    Functions
'''

def is_schedule_free_optim(cfg: Config) -> bool:
    return any(isinstance(cfg.train.optim, opt) for opt in SCHEDULEFREE_OPTIMS)

def train_epoch(cfg: Config,
                model: DDP,
                train_ds: DALIWrapper | Loader,
                criterion: Module,
                optimizer: Optimizer,
                lr_scheduler: LRScheduler,
                epoch: int,
                step: int,
                scaler: GradScaler,
                profiler: Any,
                neptune_run: Optional[Run]) -> Tuple[float, int, float]:
    start_time = time.time()
    model.train()
    if is_schedule_free_optim(cfg):
        optimizer.train() # type: ignore

    loss_metric = SmoothedValue(cfg.train.log.log_freq)
    throughput_metric = SmoothedValue(cfg.train.log.log_freq)
    if rank == 0:
        logger.info(f'[Train Epoch {epoch+1}]')

    for images, labels in train_ds:
        iter_start_time = time.time()
        # Forward pass
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', enabled=cfg.train.use_amp):
            pred = model(images)
            loss = criterion(pred, labels.view(-1))

        # Backward pass
        scaler.scale(loss).backward()
        if cfg.train.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        # Update metrics
        _loss = loss.detach().item()
        loss_metric.update(_loss)
        lr = optimizer.param_groups[0]['lr'] if not is_schedule_free_optim(cfg) else optimizer.param_groups[0]['scheduled_lr']
        step += 1

        if rank == 0:
            if cfg.train.log.wandb_on and (step % cfg.train.log.log_freq == 0):
                wandb.log({'loss': loss_metric.avg, 'lr': lr}, step=step)
            if cfg.train.log.neptune_on and (step % cfg.train.log.log_freq == 0):
                assert neptune_run is not None
                neptune_run['train/loss'].append(value=loss_metric.avg, step=step)
                neptune_run['train/lr'].append(value=lr, step=step)
            if step % cfg.train.log.log_freq == 0:
                throughput_metric.update((images.size(0) * cfg.train.network.world_size) / (time.time() - iter_start_time), images.size(0))
                logger.info(f'step: {step} ({throughput_metric.avg:5.2f} imgs/s), loss: {loss_metric.avg:.6f}' + \
                        f', lr: {lr:.6f}, mem: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB')
        profiler.step()
        del images, labels, pred, loss
    return loss_metric.global_avg, step, time.time() - start_time

def train_epoch_for_sam(cfg: Config,
                        model: DDP,
                        train_ds: DALIWrapper | Loader,
                        criterion: Module,
                        optimizer: SAM | SAMV1 | S2SAM,
                        lr_scheduler: LRScheduler,
                        epoch: int,
                        step: int,
                        scaler: GradScaler,
                        profiler: Any,
                        neptune_run: Optional[Run]) -> Tuple[float, int, float]:
    optim_cfg = cast(SGDSAMConfig, cfg.train.optim)
    assert cfg.train.grad_clip_norm == 0
    start_time = time.time()
    model.train()

    loss_metric = SmoothedValue(cfg.train.log.log_freq)
    throughput_metric = SmoothedValue(cfg.train.log.log_freq)
    if rank == 0:
        logger.info(f'[Train Epoch {epoch+1}]')

    for images, labels in train_ds:
        iter_start_time = time.time()
        # First step
        if not optim_cfg.v2:
            with model.no_sync():
                with torch.autocast(device_type='cuda', enabled=cfg.train.use_amp):
                    pred = model(images)
                    loss = criterion(pred, labels.view(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                optimizer.first_step()
                scaler.update()
                optimizer.zero_grad()
        else:
            optimizer.first_step()
            optimizer.zero_grad()

        # Forward pass
        with torch.autocast(device_type='cuda', enabled=cfg.train.use_amp):
            pred = model(images)
            loss = criterion(pred, labels.view(-1))

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Second step
        optimizer.second_step()
        scaler.update()
        optimizer.zero_grad()

        lr_scheduler.step()
        # Update metrics
        _loss = loss.detach().item()
        loss_metric.update(_loss)
        lr = optimizer.param_groups[0]['lr'] if not is_schedule_free_optim(cfg) else optimizer.param_groups[0]['scheduled_lr']
        step += 1

        if rank == 0:
            if cfg.train.log.wandb_on and (step % cfg.train.log.log_freq == 0):
                wandb.log({'loss': loss_metric.avg, 'lr': lr}, step=step)
            if cfg.train.log.neptune_on and (step % cfg.train.log.log_freq == 0):
                assert neptune_run is not None
                neptune_run['train/loss'].append(value=loss_metric.avg, step=step)
                neptune_run['train/lr'].append(value=lr, step=step)
            if step % cfg.train.log.log_freq == 0:
                throughput_metric.update((images.size(0) * cfg.train.network.world_size) / (time.time() - iter_start_time), images.size(0))
                logger.info(f'step: {step} ({throughput_metric.avg:5.2f} imgs/s), loss: {loss_metric.avg:.6f}' + \
                        f', lr: {lr:.6f}, mem: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB')
        profiler.step()
        del images, labels, pred, loss
    return loss_metric.global_avg, step, time.time() - start_time

def collect_bn_stats(cfg: Config, model: Any, stats_ds: DALIWrapper | Loader) -> None:
    model.train()
    cnt = 0
    for images, _ in stats_ds:
        if cnt < cfg.train.num_samples_for_stats:
            with torch.autocast(device_type='cuda', enabled=cfg.train.use_amp):
                with torch.no_grad():
                    model(images)
            cnt += images.size(0) * cfg.train.network.world_size

@torch.no_grad()
def valid(cfg: Config,
          model: Any,
          optimizer: Optimizer,
          stats_ds: DALIWrapper | Loader,
          valid_ds: DALIWrapper | Loader,
          criterion: Module,
          epoch: int) -> Tuple[float, float, float, int]:
    if is_schedule_free_optim(cfg):
        optimizer.eval() # type: ignore
        collect_bn_stats(cfg, model, stats_ds)
        sync_model_buffers(model)
    elif epoch == cfg.train.max_epochs - 1:
        collect_bn_stats(cfg, model, stats_ds)
        sync_model_buffers(model)

    model.eval()
    total_loss = 0.
    total_acc1 = 0.
    total_acc5 = 0.
    total_samples = 0

    if rank == 0:
        logger.info(f'[Validate Epoch {epoch+1}]')

    for images, labels in valid_ds:
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

    cfg = parse_config()

    '''
        Initialize the distributed process group.
    '''

    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.train.reproduce.seed)
    torch.cuda.manual_seed(cfg.train.reproduce.seed)
    random.seed(cfg.train.reproduce.seed)

    initialize_dist()

    '''
        Load data
    '''
    if cfg.data.dataloader.name == 'dali':
        if cfg.train.preprocess.preload_local:
            preload_to_local(cfg)
        train_ds, num_batches = get_dali_train_loader(cfg)
        valid_ds = get_dali_valid_loader(cfg)
    else:
        train_ds, _ = get_ffcv_train_loader(cfg)
        valid_ds = get_ffcv_valid_loader(cfg)
        num_batches = len(train_ds)
    logger.info(f'Number of training batches per epoch: {num_batches}')

    '''
        Initialize the model, optimizer, and learning rate scheduler.
    '''

    model = load_model(cfg.train.arch, num_classes=cfg.data.num_classes)
    # model = model.to(memory_format=torch.channels_last) # type: ignore
    model = model.to('cuda')
    model.forward = torch.compile(model.forward)

    if rank == 0:
        trainable_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
        logger.info(f'[INFO] Model trainable parameters: {trainable_params}')

    optimizer = get_optim(cfg, model, num_batches)
    lr_scheduler = get_lr_scheduler(cfg, optimizer, num_batches)
    model = DDP(model, gradient_as_bucket_view=True, broadcast_buffers=True)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    scaler = torch.GradScaler(device='cuda', enabled=cfg.train.use_amp)

    neptune_run = None
    if rank == 0:
        if cfg.train.log.wandb_on:
            wandb.init(
                project=cfg.train.log.project,
                config=cfg.model_dump(),
                name=cfg.train.log.job_id,
                dir=os.environ.get('TMPDIR', '/tmp')
            )
        if cfg.train.log.neptune_on:
            neptune_run = neptune.init_run(
                project=cfg.train.log.neptune_workspace + '/' + cfg.train.log.project,
                name=f"{cfg.train.arch}-{cfg.train.optim.name}-{cfg.train.network.world_size}GPU",
                tags=[cfg.train.arch, cfg.train.optim.name, f"{cfg.train.network.world_size}GPU", gpu_models[0]],
                source_files=[],
                monitoring_namespace='monitoring',
                capture_hardware_metrics=True,
                capture_stderr=True,
                capture_stdout=True
            )
            neptune_run['config'] = cfg.model_dump()
            logger.add(NeptuneHandler(run=neptune_run))
        logger.info(cfg)
        logger.info(model)
        with open(os.path.join(cfg.train.log.log_dir, 'data_cfg.dump.toml'), 'wb') as f:
            tomli_w.dump(cfg.data.model_dump(exclude_none=True), f)
        with open(os.path.join(cfg.train.log.log_dir, 'train_cfg.dump.toml'), 'wb') as f:
            tomli_w.dump(cfg.train.model_dump(exclude_none=True), f)

    '''
        Load the model from checkpoint if specified.
    '''

    global_step = 0
    start_epoch = 0
    total_train_time = 0.

    if cfg.train.checkpoint_dir:
        raise NotImplementedError('Checkpoint loading is not implemented yet.')

    dist.barrier()

    '''
        Training loop.
    '''
    if rank == 0:
        train_log = pd.DataFrame(columns=['epoch', 'step', 'train_loss', 'val_loss', 'val_acc1', 'val_acc5', 'time', 'checkpoint_dir'])

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=64,
            warmup=2,
            active=8,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            os.path.join(cfg.train.log.log_dir, f'tb_trace'),
            worker_name=f'worker_{rank:02d}'
        ),
    ) as profiler:
        for epoch in range(start_epoch, cfg.train.max_epochs):
            if isinstance(optimizer, SAM) or isinstance(optimizer, SAMV1) or isinstance(optimizer, S2SAM):
                train_loss, global_step, epoch_train_time = train_epoch_for_sam(cfg,
                                                                                model,
                                                                                train_ds,
                                                                                criterion,
                                                                                optimizer,
                                                                                lr_scheduler,
                                                                                epoch,
                                                                                global_step,
                                                                                scaler,
                                                                                profiler,
                                                                                neptune_run)
            else:
                train_loss, global_step, epoch_train_time = train_epoch(cfg,
                                                                        model,
                                                                        train_ds,
                                                                        criterion,
                                                                        optimizer,
                                                                        lr_scheduler,
                                                                        epoch,
                                                                        global_step,
                                                                        scaler,
                                                                        profiler,
                                                                        neptune_run)
            total_train_time += epoch_train_time
            val_loss, val_acc1, val_acc5, val_samples = valid(cfg,
                                                              model,
                                                              optimizer,
                                                              train_ds,
                                                              valid_ds,
                                                              criterion,
                                                              epoch)
            stats = gather_statistics(train_loss, val_loss, val_acc1, val_acc5, val_samples)
            checkpoint_dir = ""
            if ((epoch + 1) % cfg.train.log.checkpoint_freq == 0) and (rank == 0):
                os.makedirs(os.path.join(cfg.train.log.log_dir, 'checkpoints'), exist_ok=True)
                checkpoint_dir = os.path.join(cfg.train.log.log_dir, 'checkpoints', f'model_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'train_loss': stats[0],
                    'val_loss': stats[1],
                    'val_acc1': stats[2],
                    'val_acc5': stats[3],
                    'total_train_time': total_train_time,
                }, checkpoint_dir)

            if rank == 0:
                logger.info(f'Epoch: {epoch+1}, Train Loss: {stats[0]:.6f}, Val Loss: {stats[1]:.6f}' \
                            +f', Val Acc1: {stats[2]:.6f}, Val Acc5: {stats[3]:.6f}' \
                            +f', Epoch Train Time: {epoch_train_time:.2f} s')
                if cfg.train.log.wandb_on:
                    log_data = {
                        'epoch_train_time': epoch_train_time,
                        'val_loss': stats[1],
                        'val_acc1': stats[2],
                        'val_acc5': stats[3],
                        'epoch': epoch+1,
                        'total_train_time': total_train_time
                    }
                    wandb.log(log_data, step=global_step)
                if cfg.train.log.neptune_on:
                    assert neptune_run is not None
                    neptune_run['train/epoch_time'].append(value=epoch_train_time, step=global_step)
                    neptune_run['val/loss'].append(value=stats[1], step=global_step)
                    neptune_run['val/acc@1'].append(value=stats[2], step=global_step)
                    neptune_run['val/acc@5'].append(value=stats[3], step=global_step)
                    neptune_run['epoch'].append(value=epoch+1, step=global_step)
                    neptune_run['total_train_time'].append(value=total_train_time, step=global_step)

                train_log.loc[epoch - start_epoch] = [epoch + 1, # type: ignore
                                                      global_step,
                                                      stats[0],
                                                      stats[1],
                                                      stats[2],
                                                      stats[3],
                                                      total_train_time,
                                                      os.path.abspath(checkpoint_dir) if checkpoint_dir else '']
            if rank == 0:
                train_log.to_csv(os.path.join(cfg.train.log.log_dir, 'train_log.csv'), index=False) # type: ignore
            dist.barrier()

    if rank == 0:
        logger.info('Training finished.')
        if cfg.train.log.wandb_on:
            wandb.finish()
        if cfg.train.log.neptune_on:
            assert neptune_run is not None
            neptune_run.stop()

if __name__ == '__main__':
    main()
