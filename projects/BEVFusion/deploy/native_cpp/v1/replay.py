"""Replay model-ready camera/LiDAR packets through the native ROS2 node."""
from array import array
from collections import Counter
import argparse
import csv
import json
from pathlib import Path
import struct
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--input-dir", type=Path, required=True)
p.add_argument("--manifest", type=Path)
p.add_argument("--max-samples", type=int)
args = p.parse_args()
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray, String
manifest_path = (args.manifest or args.input_dir / "manifest.json").resolve()
entries = json.loads(manifest_path.read_text())["samples"]
if args.max_samples is not None:
    if not 1 <= args.max_samples <= len(entries):
        raise ValueError("Invalid max-samples")
    entries = entries[:args.max_samples]
args.output_dir.mkdir(parents=True, exist_ok=False)
predictions = args.output_dir / "predictions"
predictions.mkdir()

class Replay(Node):
    def __init__(self):
        super().__init__("bevfusion_native_replay")
        self.pub = self.create_publisher(UInt8MultiArray, "/bevfusion/v1_input", 1)
        self.sub = self.create_subscription(String, "/bevfusion/v1_detections_json", self.on_result, 1)
        self.result = None
    def on_result(self, msg):
        self.result = json.loads(msg.data)

rclpy.init()
node = Replay()
rows = []
try:
    deadline = time.monotonic() + 30
    while node.pub.get_subscription_count() < 1:
        if time.monotonic() > deadline:
            raise TimeoutError("Native node subscriber missing")
        rclpy.spin_once(node, timeout_sec=0.1)
    for ordinal, entry in enumerate(entries, 1):
        index, sample_idx = entry["dataset_index"], entry["sample_idx"]
        folder = args.input_dir / entry.get("directory", f"{index:03d}")
        image = (folder / "img.f16").read_bytes()
        points = (folder / "points.f32").read_bytes()
        matrices = b"".join((folder / (name + ".f32")).read_bytes()
                            for name in ("cam2lidar", "cam2img", "lidar2img", "img_aug_matrix"))
        health = json.loads((folder / "health.json").read_text())
        count = len(points) // 20
        if len(image) != 6 * 3 * 256 * 704 * 2 or len(matrices) != 1536 or len(points) % 20 or not 0 < count <= 300000:
            raise ValueError(f"Invalid packet shapes at {index}")
        packet = struct.pack("<4s7I", b"BFV1", 2, sample_idx, count,
                             len(image), len(points), len(matrices), health["available_mask"]) + image + points + matrices
        msg = UInt8MultiArray()
        msg.data = array("B", packet)
        node.result = None
        started = time.monotonic()
        node.pub.publish(msg)
        while node.result is None:
            if time.monotonic() - started > 30:
                raise TimeoutError(f"Missing result for sample {sample_idx}")
            rclpy.spin_once(node, timeout_sec=0.1)
        result = node.result
        if "error" in result:
            raise RuntimeError(result["error"])
        if result["sample_idx"] != sample_idx:
            raise ValueError("Sample identity mismatch")
        result["ros2_roundtrip_ms"] = (time.monotonic() - started) * 1000
        (predictions / f"{index:03d}.json").write_text(json.dumps(result) + "\n")
        rows.append({"sample_idx": sample_idx, "point_count": count,
                     "model_ms": result["model_ms"], "ros2_roundtrip_ms": result["ros2_roundtrip_ms"], "route": result["route"]})
        if ordinal % 10 == 0 or ordinal == len(entries):
            print(f"replayed {ordinal}/{len(entries)}", flush=True)
    with (args.output_dir / "frame_timings.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"backend": "Frozen v1 ROS2 native C++/CUDA/TensorRT, no PyTorch inference runtime",
               "count": len(rows), "route_counts": dict(Counter(r["route"] for r in rows)), "latency_scope": "CPU reliability cues/FP32-to-FP16 point conversion + calibration update + H2D + native CUDA/TRT + C++ decode; ROS2 roundtrip includes packet transport; offline image preprocessing and disk reads excluded",
               "first_model_ms": rows[0]["model_ms"], "first_ros2_roundtrip_ms": rows[0]["ros2_roundtrip_ms"]}
    for column in ("model", "ros2_roundtrip"):
        for percentile in (50, 95):
            summary[f"{column}_p{percentile}_ms"] = float(np.percentile([r[column + "_ms"] for r in rows], percentile))
    (args.output_dir / "run.json").write_text(json.dumps({"input_manifest": str(manifest_path), "count": len(rows), "score_threshold": 0}, indent=2) + "\n")
    (args.output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
