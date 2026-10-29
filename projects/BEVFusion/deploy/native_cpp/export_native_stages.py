from __future__ import annotations
from common import get_args, load_config, seed_everything
args = get_args("export_native_stages.py", export=True, stage=False)
seed_everything(args.seed)
import sys
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
import projects.BEVFusion.bevfusion
from mmdet3d.registry import MODELS
import onnx

class DenseBEVTrunk(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fuser = model.fusion_layer
        self.backbone = model.pts_backbone
        self.neck = model.pts_neck
    def forward(self, camera_bev, lidar_bev):
        return self.neck(self.backbone(self.fuser([camera_bev, lidar_bev])))[0]

out=(args.work_dir / "stages_native_01")
out.mkdir(parents=True,exist_ok=False)
cfg=load_config(args)
init_default_scope(cfg.get("default_scope","mmdet3d"))
model=MODELS.build(cfg.model)
load_checkpoint(model,str(args.checkpoint),map_location="cpu")
downsample=model.view_transform.downsample.cuda().eval()
bev=torch.zeros((1,80,360,360),device="cuda",dtype=torch.float32)
with torch.inference_mode():
    output=downsample(bev)
assert tuple(output.shape)==(1,80,180,180)
p=out/"camera_vtransform.onnx"
torch.onnx.export(downsample,bev,str(p),input_names=["feat_in"],output_names=["feat_out"],opset_version=17,do_constant_folding=True,dynamo=False)
onnx.checker.check_model(str(p))
print("downsample",tuple(output.shape),p.stat().st_size,flush=True)
trunk=DenseBEVTrunk(model).cuda().eval()
cam=torch.zeros((1,80,180,180),device="cuda",dtype=torch.float32)
lidar=torch.zeros((1,256,180,180),device="cuda",dtype=torch.float32)
with torch.inference_mode():
    output=trunk(cam,lidar)
assert tuple(output.shape)==(1,512,180,180)
p=out/"fuser.onnx"
torch.onnx.export(trunk,(cam,lidar),str(p),input_names=["camera","lidar"],output_names=["middle"],opset_version=17,do_constant_folding=True,dynamo=False)
onnx.checker.check_model(str(p))
print("fuser",tuple(output.shape),p.stat().st_size,flush=True)
