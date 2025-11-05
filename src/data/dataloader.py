import os
import glob
import numpy as np
from src.conf import Config, DaliConfig
from typing import List
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


def image_processing_func(images, image_size, interpolation, is_training, decoder_device):
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == "mixed" else 0
    preallocate_height_hint = 6430 if decoder_device == "mixed" else 0
    interpolation_type = {
        "bicubic": types.DALIInterpType.INTERP_CUBIC,
        "bilinear": types.DALIInterpType.INTERP_LINEAR,
        "triangular": types.DALIInterpType.INTERP_TRIANGULAR,
    }[interpolation]

    if is_training:
        images = fn.decoders.image_random_crop(
            images,
            device=decoder_device,
            output_type=types.DALIImageType.RGB,
            preallocate_width_hint=preallocate_width_hint,
            preallocate_height_hint=preallocate_height_hint,
            random_aspect_ratio=[0.8, 1.25],
            random_area=[0.1, 1.0],
            num_attempts=100,
        )
        images = fn.resize(
            images,
            resize_x=image_size,
            resize_y=image_size,
            interp_type=interpolation_type,
        )
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images, device=decoder_device, output_type=types.DALIImageType.RGB)
        images = fn.resize(
            images,
            size=image_size,
            mode="not_smaller",
            interp_type=interpolation_type,
        )
        mirror = False

    images = fn.crop_mirror_normalize(
        images.gpu(),  # type: ignore
        dtype=types.DALIDataType.FLOAT,
        output_layout="CHW",
        crop=(image_size, image_size),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        mirror=mirror,
    )
    return images


@pipeline_def
def create_dali_pipeline(
    shards_list: List[str],
    index_list: List[str],
    image_size: int,
    interpolation: str,
    shard_id: int,
    num_shards: int,
    dali_cpu: bool,
    is_training: bool,
):
    images, labels = fn.readers.webdataset(  # type: ignore
        paths=shards_list,
        index_paths=index_list,
        ext=["jpg", "cls"],
        dtypes=[types.DALIDataType.UINT8, types.DALIDataType.INT64],
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=is_training,
        pad_last_batch=True,
        missing_component_behavior="error",
        name="Reader",
        dont_use_mmap=True,
        initial_fill=20000,
        read_ahead=is_training,
    )
    decoder_device = "cpu" if dali_cpu else "mixed"
    images = image_processing_func(images, image_size, interpolation, is_training, decoder_device)
    return images, labels.gpu()


def get_dali_train_loader(cfg: Config):
    assert isinstance(cfg.data.dataloader, DaliConfig)
    shards_list = sorted(glob.glob(os.path.join(cfg.data.data_dir, "train-*.tar")))
    index_list = [shard + ".idx" for shard in shards_list]

    train_pipe = create_dali_pipeline(
        num_threads=cfg.data.dataloader.num_data_workers,
        batch_size=cfg.train.batch_size_per_local_batch,
        device_id=0,
        shards_list=shards_list,
        index_list=index_list,
        image_size=cfg.train.preprocess.train_crop_size,
        interpolation=cfg.train.preprocess.interpolation,
        shard_id=cfg.train.network.rank,
        num_shards=cfg.train.network.world_size,
        dali_cpu=cfg.train.preprocess.dali_cpu,
        is_training=True,
    )
    train_pipe.build()

    train_loader = DALIGenericIterator(
        train_pipe, ["data", "label"], reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP, auto_reset=True
    )

    return train_loader, int(
        train_pipe.epoch_size("Reader") / cfg.train.batch_size_per_local_batch / cfg.train.network.world_size  # type: ignore
    )


def get_dali_valid_loader(cfg: Config):
    assert isinstance(cfg.data.dataloader, DaliConfig)
    shards_list = sorted(glob.glob(os.path.join(cfg.data.data_dir, "val-*.tar")))
    index_list = [shard + ".idx" for shard in shards_list]

    val_pipe = create_dali_pipeline(
        num_threads=cfg.data.dataloader.num_data_workers,
        batch_size=cfg.train.batch_size_per_local_batch,
        device_id=0,
        shards_list=shards_list,
        index_list=index_list,
        image_size=cfg.train.preprocess.val_crop_size,
        interpolation=cfg.train.preprocess.interpolation,
        shard_id=cfg.train.network.rank,
        num_shards=cfg.train.network.world_size,
        dali_cpu=cfg.train.preprocess.dali_cpu,
        is_training=False,
    )
    val_pipe.build()
    val_loader = DALIGenericIterator(
        val_pipe, ["data", "label"], reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True
    )
    return val_loader
