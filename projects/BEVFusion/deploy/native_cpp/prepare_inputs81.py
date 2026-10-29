"""Prepare normalized binary camera/LiDAR bundles from the official mini pipeline."""
from common import parser, setup, load_config, write_json
from pathlib import Path
p = parser(__doc__)
p.add_argument("--data-root", required=True)
p.add_argument("--output-dir", type=Path)
p.add_argument("--max-samples", type=int)
args = setup(p.parse_args())
import numpy as np
import projects.BEVFusion.bevfusion  # registry
from mmdet3d.registry import DATASETS
cfg = load_config(args, args.data_root)
dataset = DATASETS.build(cfg.test_dataloader.dataset)
limit = len(dataset) if args.max_samples is None else args.max_samples
if not 1 <= limit <= len(dataset):
    raise ValueError("max-samples must be in [1, dataset_size]")
out = args.output_dir or args.work_dir / "inputs"
out.mkdir(parents=True, exist_ok=False)
pre = cfg.model.data_preprocessor
if pre.get("bgr_to_rgb", False):
    raise ValueError("This fixed runtime expects the official BGR pipeline")
mean = np.asarray(pre["mean"], np.float32).reshape(1, 3, 1, 1)
std = np.asarray(pre["std"], np.float32).reshape(1, 3, 1, 1)
manifest = {"dataset": "nuScenes v1.0-mini val", "dataset_size": len(dataset),
            "count": limit, "samples": [], "normalization": {"mean": pre["mean"], "std": pre["std"], "bgr_to_rgb": False}}
for index in range(limit):
    sample = dataset[index]
    meta = sample["data_samples"].metainfo
    image = np.ascontiguousarray((sample["inputs"]["img"].numpy().astype(np.float32) - mean) / std, dtype=np.float16)
    points = np.ascontiguousarray(sample["inputs"]["points"].numpy(), dtype=np.float16)
    if image.shape != (6, 3, 256, 704) or points.ndim != 2 or points.shape[1] != 5 or not 0 < len(points) <= 300000:
        raise ValueError(f"Unexpected input shapes at {index}: {image.shape}, {points.shape}")
    if not np.isfinite(image).all() or not np.isfinite(points).all():
        raise ValueError("Nonfinite model input")
    folder = out / f"{index:03d}"
    folder.mkdir()
    image.tofile(folder / "img.f16")
    points.tofile(folder / "points.f16")
    for key in ("cam2lidar", "cam2img", "lidar2img", "img_aug_matrix"):
        value = np.ascontiguousarray(meta[key], dtype=np.float32)
        if value.shape != (6, 4, 4) or not np.isfinite(value).all():
            raise ValueError(f"Invalid calibration {key}")
        value.tofile(folder / (key + ".f32"))
    manifest["samples"].append({"dataset_index": index, "sample_idx": int(meta["sample_idx"]), "directory": f"{index:03d}"})
    if (index + 1) % 20 == 0 or index + 1 == limit:
        print(f"prepared {index + 1}/{limit}", flush=True)
write_json(out / "manifest.json", manifest)
