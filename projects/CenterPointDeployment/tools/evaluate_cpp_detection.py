#!/usr/bin/env python3

"""Evaluate C++ TensorRT detections and compare them with PyTorch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.utils.splits import create_splits_scenes


DEFAULT_RESULT = Path(
    "work_dirs/centerpoint_deployment/detection_eval/results_nusc.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "work_dirs/centerpoint_deployment/detection_eval/official_metrics"
)
DEFAULT_PYTORCH_METRICS = Path(
    "work_dirs/centerpoint_pillar02_nuscenes_mini_test/"
    "20260901_144554/20260901_144554.json"
)
METRIC_PREFIX = "NuScenes metric/pred_instances_3d_NuScenes/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pytorch-metrics", type=Path, default=DEFAULT_PYTORCH_METRICS
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--eval-set", default="mini_val")
    return parser.parse_args()


def atomic_json(path: Path, content: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def expected_sample_tokens(nusc: NuScenes, eval_set: str) -> set[str]:
    split_scenes = set(create_splits_scenes()[eval_set])
    scene_names = {scene["token"]: scene["name"] for scene in nusc.scene}
    return {
        sample["token"]
        for sample in nusc.sample
        if scene_names[sample["scene_token"]] in split_scenes
    }


def finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a length-{length} list")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def validate_result(
    result_path: Path,
    expected_tokens: set[str],
    class_names: set[str],
) -> tuple[dict[str, Any], int]:
    submission = json.loads(result_path.read_text(encoding="utf-8"))
    if set(submission) != {"meta", "results"}:
        raise ValueError("submission must contain only meta and results")
    expected_meta = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }
    if submission["meta"] != expected_meta:
        raise ValueError(f"unexpected modality metadata: {submission['meta']}")
    results = submission["results"]
    if not isinstance(results, dict):
        raise ValueError("results must be an object keyed by sample token")
    actual_tokens = set(results)
    if actual_tokens != expected_tokens:
        missing = sorted(expected_tokens - actual_tokens)
        extra = sorted(actual_tokens - expected_tokens)
        raise ValueError(
            f"result token mismatch: missing={missing[:3]} extra={extra[:3]}"
        )

    required_fields = {
        "sample_token",
        "translation",
        "size",
        "rotation",
        "velocity",
        "detection_name",
        "detection_score",
        "attribute_name",
    }
    box_count = 0
    for sample_token, boxes in results.items():
        if not isinstance(boxes, list) or len(boxes) > 500:
            raise ValueError(f"invalid box list for sample {sample_token}")
        for box in boxes:
            if set(box) != required_fields:
                raise ValueError(f"unexpected box fields for sample {sample_token}")
            if box["sample_token"] != sample_token:
                raise ValueError(f"box token mismatch for sample {sample_token}")
            if box["detection_name"] not in class_names:
                raise ValueError(f"unknown detection class {box['detection_name']}")
            translation = finite_vector(box["translation"], 3, "translation")
            del translation
            size = finite_vector(box["size"], 3, "size")
            if not all(dimension > 0.0 for dimension in size):
                raise ValueError("box size must be positive")
            rotation = finite_vector(box["rotation"], 4, "rotation")
            quaternion_norm = math.sqrt(sum(value * value for value in rotation))
            if abs(quaternion_norm - 1.0) > 1e-5:
                raise ValueError("box quaternion is not normalized")
            finite_vector(box["velocity"], 2, "velocity")
            score = float(box["detection_score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("detection score must be within [0, 1]")
            if not isinstance(box["attribute_name"], str):
                raise ValueError("attribute_name must be a string")
            box_count += 1
    return submission, box_count


def baseline_summary(
    path: Path, class_names: list[str], distance_thresholds: list[float]
) -> dict[str, Any]:
    content = json.loads(path.read_text(encoding="utf-8"))
    class_ap = {}
    for class_name in class_names:
        values = [
            float(content[f"{METRIC_PREFIX}{class_name}_AP_dist_{threshold}"])
            for threshold in distance_thresholds
        ]
        class_ap[class_name] = statistics.mean(values)
    return {
        "mAP": float(content[f"{METRIC_PREFIX}mAP"]),
        "NDS": float(content[f"{METRIC_PREFIX}NDS"]),
        "class_AP": class_ap,
    }


def main() -> None:
    args = parse_args()
    for required in (args.result, args.pytorch_metrics):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(
        version=args.version, dataroot=str(args.data_root), verbose=False
    )
    detection_config = config_factory("detection_cvpr_2019")
    class_names = list(detection_config.class_names)
    tokens = expected_sample_tokens(nusc, args.eval_set)
    _, box_count = validate_result(
        args.result, tokens, set(class_names)
    )
    print(
        f"Validated {len(tokens)} samples and {box_count} C++ TensorRT boxes."
    )

    evaluator = DetectionEval(
        nusc=nusc,
        config=detection_config,
        result_path=str(args.result),
        eval_set=args.eval_set,
        output_dir=str(args.output_dir),
        verbose=True,
    )
    tensorrt_raw = evaluator.main(plot_examples=0, render_curves=False)
    tensorrt = {
        "mAP": float(tensorrt_raw["mean_ap"]),
        "NDS": float(tensorrt_raw["nd_score"]),
        "class_AP": {
            name: float(tensorrt_raw["mean_dist_aps"][name])
            for name in class_names
        },
    }
    pytorch = baseline_summary(
        args.pytorch_metrics, class_names, list(detection_config.dist_ths)
    )
    delta = {
        "mAP": tensorrt["mAP"] - pytorch["mAP"],
        "NDS": tensorrt["NDS"] - pytorch["NDS"],
        "class_AP": {
            name: tensorrt["class_AP"][name] - pytorch["class_AP"][name]
            for name in class_names
        },
    }
    comparison = {
        "dataset": f"nuScenes {args.version} {args.eval_set}",
        "sample_count": len(tokens),
        "tensorrt_result": str(args.result),
        "pytorch_metrics": str(args.pytorch_metrics),
        "PyTorch": pytorch,
        "TensorRT_Cpp": tensorrt,
        "TensorRT_minus_PyTorch": delta,
    }
    atomic_json(args.output_dir / "backend_comparison.json", comparison)

    fieldnames = ["backend", "mAP", "NDS"] + [
        f"{name}_AP" for name in class_names
    ]
    rows = []
    for backend, metrics in (("PyTorch", pytorch), ("TensorRT_Cpp", tensorrt)):
        row = {"backend": backend, "mAP": metrics["mAP"], "NDS": metrics["NDS"]}
        row.update(
            {f"{name}_AP": metrics["class_AP"][name] for name in class_names}
        )
        rows.append(row)
    delta_row = {"backend": "TensorRT_minus_PyTorch", "mAP": delta["mAP"], "NDS": delta["NDS"]}
    delta_row.update(
        {f"{name}_AP": delta["class_AP"][name] for name in class_names}
    )
    rows.append(delta_row)
    atomic_csv(args.output_dir / "backend_comparison.csv", fieldnames, rows)

    print("\nBackend comparison:")
    print("backend       mAP       NDS")
    print(f"PyTorch       {pytorch['mAP']:.6f}  {pytorch['NDS']:.6f}")
    print(f"TensorRT C++  {tensorrt['mAP']:.6f}  {tensorrt['NDS']:.6f}")
    print(f"delta          {delta['mAP']:+.6f} {delta['NDS']:+.6f}")
    print(f"Comparison files written to {args.output_dir}")


if __name__ == "__main__":
    main()
