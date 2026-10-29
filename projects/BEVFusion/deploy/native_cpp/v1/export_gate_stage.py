"""Export frozen v1's reliability gate or the independent LiDAR dense trunk."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import parser, setup, load_config, seed_everything
p = parser(__doc__, export=True)
p.add_argument("--mode", choices=("fusion", "lidar"), required=True)
a = setup(p.parse_args())
seed_everything(a.seed)
import torch
import onnx
import projects.BEVFusion.bevfusion
from mmengine.runner.checkpoint import load_checkpoint
from mmdet3d.registry import MODELS

cfg = load_config(a)
model = MODELS.build(cfg.model)
checkpoint = load_checkpoint(model, str(a.checkpoint), map_location="cpu")
state = checkpoint["state_dict"]
out = a.work_dir / "stages_v1"
out.mkdir(parents=True, exist_ok=False)

class GatedTrunk(torch.nn.Module):
    def __init__(self, m, state):
        super().__init__()
        self.fuser, self.backbone, self.neck = m.fusion_layer, m.pts_backbone, m.pts_neck
        self.gate = torch.nn.Sequential(torch.nn.Conv2d(5,16,3,padding=1),torch.nn.ReLU(),torch.nn.Conv2d(16,2,1))
        prefix = "fusion_layer.reliability_net."
        self.gate.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}, strict=True)
    def forward(self, camera, lidar, reliability_cues):
        camera_energy = torch.log1p(camera.float().abs().mean(1, keepdim=True))
        lidar_energy = torch.log1p(lidar.float().abs().mean(1, keepdim=True))
        cues = torch.cat((reliability_cues, camera_energy, lidar_energy), dim=1)
        log_scales = torch.tanh(self.gate(cues))
        # This branch is entered only when at least one camera is available.
        weighted = torch.cat((camera * torch.exp(log_scales[:,:1]), lidar * torch.exp(log_scales[:,1:])),dim=1)
        fused = self.fuser[2](self.fuser[1](self.fuser[0](weighted)))
        return self.neck(self.backbone(fused))[0]

class LidarTrunk(torch.nn.Module):
    def __init__(self,m):
        super().__init__()
        self.backbone,self.neck=m.pts_backbone,m.pts_neck
    def forward(self,lidar):
        return self.neck(self.backbone(lidar))[0]

def export(module, inputs, path, names):
    module=module.cuda().eval()
    with torch.inference_mode():
        torch.onnx.export(module,inputs,str(path),input_names=names,output_names=["middle"],
                          opset_version=17,do_constant_folding=True,dynamo=False)
    onnx.checker.check_model(str(path))
    print("exported",path,flush=True)

if a.mode=="fusion":
    downsample=model.view_transform.downsample.cuda().eval()
    with torch.inference_mode():
        torch.onnx.export(downsample,torch.zeros((1,80,360,360),device="cuda"),str(out/"camera_vtransform_raw.onnx"),
                          input_names=["feat_in"],output_names=["feat_out"],opset_version=17,dynamo=False)
    gate=GatedTrunk(model,state).cuda().eval()
    inputs=(torch.zeros((1,80,180,180),device="cuda"),torch.zeros((1,256,180,180),device="cuda"),torch.zeros((1,3,180,180),device="cuda"))
    export(gate,inputs,out/"gate_raw.onnx",["camera","lidar","reliability_cues"])
else:
    export(LidarTrunk(model),torch.zeros((1,256,180,180),device="cuda"),out/"lidar_trunk_raw.onnx",["lidar"])
