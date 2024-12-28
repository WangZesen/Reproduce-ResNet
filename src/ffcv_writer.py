import argparse
import tomllib
import os
from src.conf import Data as DataConfig
from torchvision.datasets import ImageFolder
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from typing import Final

WRITE_MODE: Final[str] = 'proportion'
CHUNK_SIZE: Final[int] = 100
NUM_WORKERS: Final[int] = (lambda x: max(x, 1) if x is not None else 1)(os.cpu_count())

def load_toml(config_dir):
    with open(config_dir, 'rb') as f:
        config = tomllib.load(f)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-cfg', type=str, required=True)
    args = parser.parse_args()
    cfg = load_toml(args.data_cfg)
    cfg = DataConfig.model_validate(cfg)

    for split in ['val', 'train']:
        dataset = ImageFolder(os.path.join(cfg.data_dir, split))
        write_path = os.path.join(cfg.ffcv_data_dir, cfg.ffcv_preprocess.tag + f"_{split}.ffcv")
        writer = DatasetWriter(write_path, {
            'image': RGBImageField(write_mode=WRITE_MODE,
                                   max_resolution=cfg.ffcv_preprocess.max_resolution,
                                   compress_probability=cfg.ffcv_preprocess.compress_probability,
                                   jpeg_quality=cfg.ffcv_preprocess.jpeg_quality),
            'label': IntField(),
        }, num_workers=NUM_WORKERS)
        writer.from_indexed_dataset(dataset, chunksize=CHUNK_SIZE)
