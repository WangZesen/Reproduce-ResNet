import os
import glob
import subprocess
import time
import torch.distributed as dist
from loguru import logger
from src.conf import Config, DaliConfig


def preload_to_local(cfg: "Config"):
    dataloader_cfg = cfg.data.dataloader
    assert isinstance(dataloader_cfg, DaliConfig)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    tmp_dir = os.environ["TMPDIR"]

    shards_list = sorted(glob.glob(os.path.join(cfg.data.data_dir, "*.tar")))
    num_shards = len(shards_list)

    local_shards_list = sorted(glob.glob(os.path.join(tmp_dir, "*.tar")))

    if len(local_shards_list) == num_shards:
        logger.debug(f'[rank {rank}] Preload Complete "data_dir" is changed to: {tmp_dir}')
        cfg.data.data_dir = tmp_dir
        return

    start = time.time()
    cnt = 0
    for i in range(local_rank, num_shards, local_world_size):
        subprocess.call(
            f"cp -f {shards_list[i]} {tmp_dir}/",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.call(
            f"cp -f {shards_list[i]}.idx {tmp_dir}/",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cnt += 1
        if cnt % 10 == 0:
            logger.info(f"[rank {rank}] Preloading data... {cnt} copied.")

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()

    if rank == 0:
        logger.info(f'Preload Complete ({time.time() - start:.2f}s). "data_dir" is changed to: {tmp_dir}')

    cfg.data.data_dir = tmp_dir
