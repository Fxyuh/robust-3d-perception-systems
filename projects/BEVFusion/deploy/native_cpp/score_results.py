"""Score native ROS2 predictions with the official nuScenes mini evaluator."""
from common import parser, setup, load_config, write_json
from pathlib import Path
import json
p = parser(__doc__)
p.add_argument("--data-root", required=True)
p.add_argument("--lidar-only", action="store_true")
p.add_argument("--inference-dir", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
args = setup(p.parse_args())
import numpy as np
import torch
from mmengine.structures import InstanceData
import projects.BEVFusion.bevfusion
from mmdet3d.registry import DATASETS, METRICS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
cfg = load_config(args, args.data_root)
dataset = DATASETS.build(cfg.test_dataloader.dataset)
run = json.loads((args.inference_dir / "run.json").read_text())
manifest = json.loads(Path(run["input_manifest"]).read_text())
entries = manifest["samples"][:run["count"]]
if len(entries) != len(dataset) or [e["dataset_index"] for e in entries] != list(range(len(dataset))):
    raise ValueError("Official metrics require every mini-val frame exactly once")
args.output_dir.mkdir(parents=True, exist_ok=False)
metric_cfg = dict(cfg.test_evaluator)
metric_cfg["jsonfile_prefix"] = str(args.output_dir / "nuscenes_results")
evaluator = METRICS.build(metric_cfg)
if args.lidar_only:
    evaluator.modality = dict(use_camera=False, use_lidar=True)
evaluator.dataset_meta = dataset.metainfo
for ordinal, entry in enumerate(entries, 1):
    index = entry["dataset_index"]
    result = json.loads((args.inference_dir / "predictions" / f"{index:03d}.json").read_text())
    if result["sample_idx"] != entry["sample_idx"]:
        raise ValueError(f"sample identity mismatch: {index}")
    boxes = np.asarray(result["boxes_3d"], np.float32).reshape(-1, 9)
    scores = np.asarray(result["scores"], np.float32)
    labels = np.asarray(result["labels"], np.int64)
    if not len(boxes) == len(scores) == len(labels) == result["num_detections"]:
        raise ValueError(f"Detection count mismatch at {index}")
    prediction = InstanceData()
    prediction.bboxes_3d = LiDARInstance3DBoxes(torch.from_numpy(boxes), box_dim=9)
    prediction.scores_3d = torch.from_numpy(scores)
    prediction.labels_3d = torch.from_numpy(labels)
    sample = dataset[index]["data_samples"]
    sample.pred_instances_3d = prediction
    sample.pred_instances = InstanceData()
    evaluator.process(data_batch={}, data_samples=[sample.to_dict()])
    if ordinal % 20 == 0 or ordinal == len(entries):
        print(f"scored {ordinal}/{len(entries)}", flush=True)
metrics = evaluator.evaluate(len(dataset))
# Official class metrics can contain NaN for absent/undefined attributes.
(args.output_dir / "all_metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
summary = json.loads((args.inference_dir / "inference_summary.json").read_text())
prefix = "NuScenes metric/pred_instances_3d_NuScenes/"
summary.update(dataset="nuScenes v1.0-mini val", mAP=float(metrics[prefix + "mAP"]), NDS=float(metrics[prefix + "NDS"]))
write_json(args.output_dir / "summary.json", summary)
print(json.dumps(summary, indent=2), flush=True)
