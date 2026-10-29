from __future__ import annotations
from common import get_args, load_config, seed_everything
args = get_args("export_camera_native.py", export=True, stage=False)
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

OUT=(args.work_dir / "camera_native_01")
OUT.mkdir(parents=True,exist_ok=False)
cfg=load_config(args)
init_default_scope(cfg.get("default_scope","mmdet3d"))
model=MODELS.build(cfg.model)
load_checkpoint(model,str(args.checkpoint),map_location="cpu")
class CameraNative(torch.nn.Module):
    def __init__(self,m):
        super().__init__()
        self.backbone=m.img_backbone
        self.neck=m.img_neck
        self.dtransform=m.view_transform.dtransform
        self.depthnet=m.view_transform.depthnet
        self.D=m.view_transform.D
        self.C=m.view_transform.C
    def forward(self,img,depth):
        features=self.neck(self.backbone(img))
        if not isinstance(features,torch.Tensor): features=features[0]
        d=self.dtransform(depth)
        logits=self.depthnet(torch.cat((d,features),dim=1))
        weights=logits[:,:self.D].softmax(dim=1)
        camera_feature=logits[:,self.D:self.D+self.C].permute(0,2,3,1)
        return weights,camera_feature
camera=CameraNative(model).cuda().eval()
del model
gc.collect()
img=torch.zeros((6,3,256,704),device="cuda",dtype=torch.float32)
depth=torch.zeros((6,1,256,704),device="cuda",dtype=torch.float32)
with torch.inference_mode():
    outputs=camera(img,depth)
print("shapes",[tuple(o.shape) for o in outputs],flush=True)
onnx_path=OUT/"camera_with_depth.onnx"
torch.onnx.export(camera,(img,depth),str(onnx_path),input_names=["img","depth"],output_names=["camera_depth_weights","camera_feature"],opset_version=17,do_constant_folding=True,dynamo=False)
onnx.checker.check_model(str(onnx_path))
print("onnx_ok",onnx_path,onnx_path.stat().st_size,flush=True)
