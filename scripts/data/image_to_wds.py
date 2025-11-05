import os
import io
import tarfile
import json
import random
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image
import multiprocessing as mp
import math

def get_class_to_idx(root):
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    return {cls_name: idx for idx, cls_name in enumerate(classes)}

def process_sample(img_path, class_idx, split_root):
    """读取图像并生成 image bytes 和 metadata JSON bytes"""
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            width, height = img.size
        
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        rel_path = os.path.relpath(img_path, split_root).replace("\\","/")

        meta = {
            "original_path": rel_path,
            "width": width,
            "height": height,
            "label": class_idx
        }
        meta_bytes = json.dumps(meta).encode("utf-8")
        return img_bytes, class_idx, meta_bytes, meta
    except Exception as e:
        print(f"[WARN] Skip {img_path}: {e}")
        return None, None, None, None

def write_shard(args):
    shard_id, samples, output_dir, prefix, split_root = args
    tar_path = os.path.join(output_dir, f"{prefix}-shard-{shard_id:06d}.tar")
    shard_manifest = []

    with tarfile.open(tar_path, "w") as tar:
        for i, (img_path, class_idx) in enumerate(samples):
            img_bytes, cls_idx, meta_bytes, meta = process_sample(img_path, class_idx, split_root)
            if img_bytes is None:
                continue

            # 写图像
            img_info = tarfile.TarInfo(f"{i:08d}.jpg")
            img_info.size = len(img_bytes)
            tar.addfile(img_info, io.BytesIO(img_bytes))

            # 写类别
            cls_info = tarfile.TarInfo(f"{i:08d}.cls")
            cls_bytes = str(cls_idx).encode("utf-8")
            cls_info.size = len(cls_bytes)
            tar.addfile(cls_info, io.BytesIO(cls_bytes))

            # 写元数据 JSON
            json_info = tarfile.TarInfo(f"{i:08d}.json")
            json_info.size = len(meta_bytes)
            tar.addfile(json_info, io.BytesIO(meta_bytes))

            shard_manifest.append(meta)

    return {"path": tar_path, "num_samples": len(shard_manifest), "samples_info": shard_manifest}

def make_webdataset(split_name, input_dir, output_dir, num_shards=128, num_workers=8, shuffle=False, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📦 Processing split: {split_name}")

    class_to_idx = get_class_to_idx(input_dir)

    # 收集样本
    samples = []
    for cls_name, cls_idx in tqdm(class_to_idx.items(), desc=f"Indexing {split_name}"):
        img_files = glob(os.path.join(input_dir, cls_name, "*.JPEG"))
        for f in img_files:
            samples.append((f, cls_idx))

    print(f"  Found {len(samples)} images across {len(class_to_idx)} classes")

    if shuffle:
        print(f"  Shuffling samples with seed={seed}")
        random.Random(seed).shuffle(samples)

    total_samples = len(samples)
    samples_per_shard = math.ceil(total_samples / num_shards)

    shards = []
    for shard_id in range(num_shards):
        start_idx = round(shard_id * total_samples / num_shards)
        end_idx = round((shard_id + 1) * total_samples / num_shards)
        shard_samples = samples[start_idx:end_idx]
        shards.append((shard_id, shard_samples, output_dir, split_name, input_dir))

    print(f"  Will create {len(shards)} shards (~{samples_per_shard} samples each)")

    manifest = []
    with mp.Pool(num_workers) as pool:
        for info in tqdm(pool.imap(write_shard, shards), total=len(shards), desc=f"Writing {split_name}"):
            manifest.append(info)

    # 写 manifest.json
    manifest_path = os.path.join(output_dir, f"{split_name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "split": split_name,
            "num_shards": len(manifest),
            "total_samples": total_samples,
        }, f, indent=2)

    print(f"✅ {split_name} done. Manifest written to {manifest_path}")

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    splits = []

    # if os.path.isdir(os.path.join(input_dir, "train")):
    #     splits.append(("train", os.path.join(input_dir, "train")))
    if os.path.isdir(os.path.join(input_dir, "val")):
        splits.append(("val", os.path.join(input_dir, "val")))

    if not splits:
        raise RuntimeError(f"No train/ or val/ found under {input_dir}")

    for split_name, split_dir in splits:
        make_webdataset(
            split_name,
            split_dir,
            output_dir,
            num_shards=args.num_shards if split_name=="train" else 64,
            num_workers=args.num_workers,
            shuffle=(split_name=="train"),
            seed=args.seed
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ImageNet to WebDataset (fixed shards + JSON inside tar)")
    parser.add_argument("--input-dir", required=True, help="Root of ImageNet (train/val)")
    parser.add_argument("--output-dir", required=True, help="Output directory for .tar shards")
    parser.add_argument("--num-shards", type=int, default=128, help="Number of shards")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    args = parser.parse_args()
    main(args)
