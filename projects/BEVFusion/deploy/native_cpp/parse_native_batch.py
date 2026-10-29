"""Convert the optional direct C++ batch runner log to evaluator JSON files."""
from pathlib import Path
import argparse
import json
import numpy as np
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--log-file", type=Path, required=True)
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--input-dir", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
a = p.parse_args()
manifest = json.loads(a.manifest.read_text())
a.output_dir.mkdir(parents=True, exist_ok=False)
predictions = a.output_dir / "predictions"
predictions.mkdir()
frames, current = {}, None
for line in a.log_file.read_text().splitlines():
    if line.startswith("FRAME "):
        _, index, count, ms = line.split()
        current = int(index)
        if current in frames:
            raise ValueError("Duplicate frame")
        frames[current] = {"count": int(count), "ms": float(ms), "boxes": []}
    elif line.startswith("BOX "):
        parts = line.split()
        label, score = int(parts[1]), float(parts[2])
        x, y, z, w, length, height, vx, vy, yaw = map(float, parts[3:])
        frames[current]["boxes"].append((label, score, [x, y, z, w, length, height, yaw, vx, vy]))
    elif line.startswith("END_FRAME "):
        if int(line.split()[1]) != current or len(frames[current]["boxes"]) != frames[current]["count"]:
            raise ValueError("Incomplete frame")
        current = None
if current is not None or set(frames) != {e["dataset_index"] for e in manifest["samples"]}:
    raise ValueError("Incomplete native batch log")
for entry in manifest["samples"]:
    index = entry["dataset_index"]
    rows = frames[index]["boxes"]
    result = {"sample_idx": entry["sample_idx"], "camera_count": 6,
              "point_count": (a.input_dir / f"{index:03d}" / "points.f16").stat().st_size // 10,
              "num_detections": len(rows), "boxes_3d": [r[2] for r in rows],
              "scores": [r[1] for r in rows], "labels": [r[0] for r in rows],
              "model_ms": frames[index]["ms"]}
    (predictions / f"{index:03d}.json").write_text(json.dumps(result) + "\n")
latencies = [frames[e["dataset_index"]]["ms"] for e in manifest["samples"]]
summary = {"count": len(frames), "backend": "native C++/CUDA/TensorRT",
           "model_p50_ms": float(np.percentile(latencies, 50)), "model_p95_ms": float(np.percentile(latencies, 95))}
(a.output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(a.output_dir / "run.json").write_text(json.dumps({"input_manifest": str(a.manifest.resolve()), "count": len(frames), "score_threshold": 0}, indent=2) + "\n")
