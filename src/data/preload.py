import os
import subprocess
import time
import torch.distributed as dist
from loguru import logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..conf import Config

def preload_to_local(cfg: 'Config'):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    tmp_dir = os.environ["TMPDIR"]
    num_shards = int(subprocess.check_output(f'ls {cfg.data.sharded_data_dir} | grep zip | wc -l', shell=True).decode().strip())

    if os.path.exists(os.path.join(tmp_dir, "train")):
        logger.info(f'Preload Complete [rank {rank}]. "data_dir" is changed to: {tmp_dir}')
        cfg.data.data_dir = tmp_dir
        return

    for i in range(rank, num_shards, world_size):
        logger.info(f'Preloading data to local storage [rank {rank}]')
        start_time = time.time()
        subprocess.check_call(f'cp {cfg.data.sharded_data_dir}/shard{i}.zip {tmp_dir}', shell=True)
        logger.info(f'Done [rank {rank}]. Elapsed time: {time.time() - start_time:.2f} s')
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()
    
    if rank == 0:
        logger.info(f'Unzipping data [rank {rank}]')
    
    start_time = time.time()
    for i in range(rank, num_shards, world_size):
        subprocess.check_call(f"mkdir {tmp_dir}/shard{i}", shell=True)        
        subprocess.check_call(f"unzip -q {tmp_dir}/shard{i}.zip -d {tmp_dir}/shard{i}", shell=True)
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()
    if rank == 0:
        logger.info(f'Done. Elapsed time: {time.time() - start_time:.2f} s')
    
    if rank == 0:
        start_time = time.time()
        logger.info(f'Rearrange data in local storage [rank {rank}]')
        local_folder = tmp_dir
        os.makedirs(os.path.join(local_folder, "train"))
        os.makedirs(os.path.join(local_folder, "val"))
        for i in range(num_shards):
            subprocess.call(f"mv {tmp_dir}/shard{i}/*/train/* {tmp_dir}/train/", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.call(f"mv {tmp_dir}/shard{i}/*/val/* {tmp_dir}/val/", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f'Done [rank {rank}]. Elapsed time: {time.time() - start_time:.2f} s')
    
    if rank == 0:
        logger.info(f'Preload Complete [rank {rank}]. "data_dir" is changed to: {tmp_dir}')

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()

    cfg.data.data_dir = tmp_dir

