from __future__ import annotations
from common import get_args, load_config, seed_everything
args = get_args("export_head_native.py", export=True, stage=False)
seed_everything(args.seed)
import gc
import sys
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
import projects.BEVFusion.bevfusion
from mmdet3d.registry import MODELS
import onnx

ROOT = (args.work_dir / "head_export_02")
ROOT.mkdir(parents=True, exist_ok=False)
cfg = load_config(args)
init_default_scope(cfg.get("default_scope", "mmdet3d"))
model = MODELS.build(cfg.model)
load_checkpoint(model, str(args.checkpoint), map_location="cpu")
head = model.bbox_head.cuda().eval()
del model
gc.collect()

class HeadOnly(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        result = self.module.forward_single(x, [])[0]
        return (
            result["heatmap"], result["center"], result["height"],
            result["dim"], result["rot"], result["vel"],
            result["query_heatmap_score"], self.module.query_labels)

module = HeadOnly(head).eval()
import torch.nn.functional as F
_original_pool = F.max_pool2d
def _trt_safe_pool(x, *args, **kwargs):
    if x.dim() == 3:
        return _original_pool(x.unsqueeze(1), *args, **kwargs).squeeze(1)
    return _original_pool(x, *args, **kwargs)
F.max_pool2d = _trt_safe_pool
x = torch.randn((1, 512, 180, 180), device="cuda", dtype=torch.float32)
with torch.inference_mode():
    outputs = module(x)
print("outputs", [tuple(t.shape) for t in outputs], flush=True)
out = ROOT / "head.onnx"
torch.onnx.export(
    module, x, str(out), opset_version=17,
    input_names=["bev_features"],
    output_names=["heatmap", "center", "height", "dim", "rot", "vel", "query_heatmap_score", "query_labels"],
    do_constant_folding=True, dynamo=False)
onnx.checker.check_model(str(out))
print("onnx_ok", out, out.stat().st_size, flush=True)
