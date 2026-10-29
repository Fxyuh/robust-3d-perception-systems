"""Export one checkpoint and build the four engines in a fresh work directory."""
from common import get_args, sha256, write_json
from pathlib import Path
import json
import subprocess
import sys

args = get_args(__doc__, export=True)
if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
args.work_dir.mkdir(parents=True, exist_ok=False)
folder = Path(__file__).resolve().parent
base = ["--work-dir", str(args.work_dir), "--checkpoint", str(args.checkpoint), "--seed", str(args.seed),
        "--workspace-gib", str(args.workspace_gib)]
if args.mmdet3d_root:
    base += ["--mmdet3d-root", str(args.mmdet3d_root)]
if args.config:
    base += ["--config", str(args.config)]
for path in args.python_path:
    base += ["--python-path", str(path)]
stages = [
    ("export_camera_native.py", []), ("export_sparse_native_v2.py", []),
    ("export_native_stages.py", []), ("export_head_native.py", []),
    ("camera_fp16_io.py", []), ("convert_native_stages_fp16.py", []),
    ("rewrite_head_topk.py", []), ("convert_head_native.py", []),
    ("build_native_camera_fp16_io.py", []),
    ("build_native_stage.py", ["--stage", "camera_vtransform"]),
    ("build_native_stage.py", ["--stage", "fuser"]),
    ("build_head_native.py", []),
]
for index, (name, extra) in enumerate(stages, 1):
    print(f"export/build {index}/{len(stages)}: {name}", flush=True)
    subprocess.run([sys.executable, str(folder / name), *base, *extra], check=True)
models = args.work_dir / "model"
models.mkdir()
mapping = {
    "camera.engine": "camera_native_02/camera_native_fp16.engine",
    "camera_vtransform.engine": "stages_native_02/camera_vtransform.engine",
    "fuser.engine": "stages_native_02/fuser.engine",
    "head_bbox.engine": "head_native_01/head_bbox.engine",
    "lidar_sparse.onnx": "sparse_native_02/lidar_sparse.onnx",
}
for name, relative in mapping.items():
    (models / name).symlink_to(Path("..") / relative)
write_json(args.work_dir / "artifacts.json", {
    "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_name": args.checkpoint.name,
    "seed": args.seed, "files": {name: {"sha256": sha256(models / name), "bytes": (models / name).stat().st_size} for name in mapping}
})
print("model directory:", models, flush=True)
