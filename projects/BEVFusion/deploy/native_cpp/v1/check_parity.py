"""Check native cue maps and the exported gated trunk against the frozen PyTorch gate."""
import sys,argparse,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import parser,setup,load_config,write_json
p=parser(__doc__,export=True)
p.add_argument("--input-dir",type=Path,required=True)
p.add_argument("--cue-binary",type=Path,required=True)
p.add_argument("--gate-engine",type=Path,required=True)
p.add_argument("--output-dir",type=Path,required=True)
a=setup(p.parse_args())
import numpy as np
import torch
import tensorrt as trt
import projects.BEVFusion.bevfusion
from reference_gate import ReliabilityConvFuser
from mmdet3d.registry import MODELS
from mmengine.runner.checkpoint import load_checkpoint
a.output_dir.mkdir(parents=True,exist_ok=False)
cfg=load_config(a);model=MODELS.build(cfg.model)
checkpoint=load_checkpoint(model,str(a.checkpoint),map_location="cpu")
prefix="fusion_layer."
gate=ReliabilityConvFuser([80,256],256,[-54,54],[-54,54]).cuda().eval()
gate.load_state_dict({k[len(prefix):]:v for k,v in checkpoint["state_dict"].items() if k.startswith(prefix)},strict=True)
backbone,neck=model.pts_backbone.cuda().eval(),model.pts_neck.cuda().eval()
report={"conditions":{}}
saved=None
for name in ("normal","camera_drop_3_lidar_drop_75"):
    d=a.input_dir/name/"000"
    health=json.loads((d/"health.json").read_text())
    points=torch.from_numpy(np.fromfile(d/"points.f32",np.float32).reshape(-1,5)).cuda()
    images=torch.from_numpy(np.fromfile(d/"img.f16",np.float16).reshape(1,6,3,256,704)).cuda().float()
    l2i=np.fromfile(d/"lidar2img.f32",np.float32).reshape(6,4,4)
    aug=np.fromfile(d/"img_aug_matrix.f32",np.float32).reshape(6,4,4)
    metas=[{"lidar2img":l2i,"img_aug_matrix":aug,"sensor_degradation":{"dropped_camera_indices":health["dropped_camera_indices"]}}]
    with torch.no_grad():
        density=gate._density_map([points],180,180,points.device)
        coverage,_=gate._camera_coverage(images,metas,180,180)
        centers=gate._bev_centers(180,180,points.device,torch.float32)
        distance=(centers[:2].square().sum(dim=0).sqrt().reshape(1,1,180,180)/76.4).clamp(max=1)
        expected=torch.cat((density,coverage,distance),dim=1)
    output=a.output_dir/(name+"_cues.f32")
    subprocess.run([str(a.cue_binary),str(d),str(health["available_mask"]),str(output)],check=True)
    native=np.fromfile(output,np.float32).reshape(1,3,180,180)
    error=np.abs(native-expected.cpu().numpy())
    report["conditions"][name]={"max_abs_per_map":error.reshape(3,-1).max(1).astype(float).tolist(),
                              "coverage_differing_cells":int(np.count_nonzero(error[:,1]>1e-5))}
    if np.max(error)>1e-5:raise AssertionError(report)
    saved=(points,images,metas,torch.from_numpy(native).cuda())
torch.manual_seed(0)
camera=torch.randn((1,80,180,180),device="cuda",dtype=torch.float16).float()*0.1
lidar=torch.randn((1,256,180,180),device="cuda",dtype=torch.float16).float()*0.1
camera=camera.half();lidar=lidar.half()
points,images,metas,cues=saved
with torch.inference_mode():
    expected=neck(backbone(gate([camera.float(),lidar.float()],[points],images,metas)))[0]
logger=trt.Logger(trt.Logger.WARNING);runtime=trt.Runtime(logger)
engine=runtime.deserialize_cuda_engine(a.gate_engine.read_bytes())
context=engine.create_execution_context()
output=torch.empty((1,512,180,180),device="cuda",dtype=torch.float16)
for name,value in {"camera":camera,"lidar":lidar,"reliability_cues":cues,"middle":output}.items():
    context.set_tensor_address(name,value.data_ptr())
if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):raise RuntimeError("TRT execution failed")
torch.cuda.synchronize()
ref=expected.float().cpu().numpy().reshape(-1)
actual=output.float().cpu().numpy().reshape(-1)
diff=np.abs(ref-actual)
report["gate_trunk"]={"mean_abs":float(diff.mean()),"max_abs":float(diff.max()),
"relative_l2":float(np.linalg.norm(ref-actual)/max(np.linalg.norm(ref),1e-12)),
"correlation":float(np.corrcoef(ref,actual)[0,1])}
if report["gate_trunk"]["correlation"]<0.995:raise AssertionError(report)
write_json(a.output_dir/"parity.json",report)
print(json.dumps(report,indent=2),flush=True)
