"""Shared paths/configuration for the fixed-shape BEVFusion deployment."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

OFFICIAL_CONFIG = "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"

def parser(description, *, export=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--mmdet3d-root", type=Path, default=os.environ.get("MMDET3D_ROOT"))
    p.add_argument("--config", type=Path)
    p.add_argument("--checkpoint", type=Path, required=export)
    p.add_argument("--python-path", type=Path, action="append", default=[])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workspace-gib", type=int, default=3)
    return p

def setup(args):
    args.work_dir = args.work_dir.resolve()
    if args.mmdet3d_root:
        args.mmdet3d_root = args.mmdet3d_root.resolve()
        sys.path.insert(0, str(args.mmdet3d_root))
    for path in args.python_path:
        sys.path.append(str(path.resolve()))
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    return args

def get_args(description, *, export=False, stage=False):
    p = parser(description, export=export)
    if stage:
        p.add_argument("--stage", choices=("camera_vtransform", "fuser"), required=True)
    return setup(p.parse_args())

def load_config(args, data_root=None):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    if args.config:
        cfg_path = args.config.resolve()
    elif args.mmdet3d_root:
        cfg_path = args.mmdet3d_root / OFFICIAL_CONFIG
    else:
        raise ValueError("Pass --mmdet3d-root or --config")
    cfg = Config.fromfile(str(cfg_path))
    cfg.model.img_backbone.init_cfg = None
    def adapt(dataset):
        if "dataset" in dataset:
            adapt(dataset["dataset"])
        elif "datasets" in dataset:
            for child in dataset["datasets"]:
                adapt(child)
        else:
            dataset.setdefault("metainfo", {})["version"] = "v1.0-mini"
            if data_root:
                dataset["data_root"] = str(Path(data_root).resolve()) + "/"
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if key in cfg:
            adapt(cfg[key]["dataset"])
    if data_root:
        for key in ("val_evaluator", "test_evaluator"):
            cfg[key]["data_root"] = str(Path(data_root).resolve()) + "/"
            cfg[key]["ann_file"] = str(Path(data_root).resolve() / Path(cfg[key]["ann_file"]).name)
    init_default_scope(cfg.get("default_scope", "mmdet3d"))
    return cfg

def seed_everything(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
