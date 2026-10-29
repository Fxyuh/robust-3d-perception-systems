"""Export the exact frozen v1 gate and LiDAR fallback into separate native models."""
import argparse,json,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import sha256,write_json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--mmdet3d-root",type=Path,required=True)
p.add_argument("--fusion-checkpoint",type=Path,required=True)
p.add_argument("--lidar-checkpoint",type=Path,required=True)
p.add_argument("--output-dir",type=Path,required=True)
p.add_argument("--python-path",action="append",default=[])
a=p.parse_args()
here=Path(__file__).resolve().parent;parent=here.parent
contract=json.loads((here/"frozen_contract.json").read_text())
for name,path in (("local_gate",a.fusion_checkpoint),("lidar_fallback",a.lidar_checkpoint)):
    if sha256(path)!=contract["weights"][name]["sha256"]:raise ValueError("Not the frozen v1 weight: "+name)
a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=False)
extra=[]
for path in a.python_path:extra+=["--python-path",path]
def shared(script,branch,checkpoint,options=()):
    cmd=[sys.executable,str(parent/script),"--work-dir",str(a.output_dir/branch),
         "--mmdet3d-root",str(a.mmdet3d_root),"--checkpoint",str(checkpoint),*extra,*options]
    print("stage",branch,script,flush=True);subprocess.run(cmd,check=True)
def custom(script,options):
    print("stage",script,*options[:2],flush=True)
    subprocess.run([sys.executable,str(here/script),*options,*extra],check=True)
lidar_config=a.mmdet3d_root/"projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
for branch,checkpoint,config in (("fusion",a.fusion_checkpoint,[]),("lidar",a.lidar_checkpoint,["--config",str(lidar_config)])):
    if branch=="fusion":
        shared("export_camera_native.py",branch,checkpoint)
    shared("export_sparse_native_v2.py",branch,checkpoint,config)
    shared("export_head_native.py",branch,checkpoint,config)
    custom("export_gate_stage.py",["--mode",branch,"--work-dir",str(a.output_dir/branch),
        "--mmdet3d-root",str(a.mmdet3d_root),"--checkpoint",str(checkpoint),*config])
    shared("rewrite_head_topk.py",branch,checkpoint)
    shared("convert_head_native.py",branch,checkpoint)
    shared("build_head_native.py",branch,checkpoint)
    if branch=="fusion":
        shared("camera_fp16_io.py",branch,checkpoint)
        shared("build_native_camera_fp16_io.py",branch,checkpoint)
        stages=[("camera_vtransform_raw","camera_vtransform",[]),("gate_raw","gate_fuser",["--keep-fp32","reliability_cues"])]
    else:
        stages=[("lidar_trunk_raw","lidar_trunk",[])]
    for source,target,opts in stages:
        folder=a.output_dir/branch/"stages_v1"
        custom("fp16_io.py",["--source",str(folder/(source+".onnx")),"--output",str(folder/(target+".onnx")),*opts])
        custom("build_engine.py",["--onnx",str(folder/(target+".onnx")),"--output",str(folder/(target+".engine"))])
model=a.output_dir/"model"
mapping={
"fusion":{"camera.engine":"camera_native_02/camera_native_fp16.engine",
"camera_vtransform.engine":"stages_v1/camera_vtransform.engine","fuser.engine":"stages_v1/gate_fuser.engine",
"head_bbox.engine":"head_native_01/head_bbox.engine","lidar_sparse.onnx":"sparse_native_02/lidar_sparse.onnx"},
"lidar":{"trunk.engine":"stages_v1/lidar_trunk.engine","head_bbox.engine":"head_native_01/head_bbox.engine","lidar_sparse.onnx":"sparse_native_02/lidar_sparse.onnx"}}
hashes={}
for branch,items in mapping.items():
    (model/branch).mkdir(parents=True)
    for name,path in items.items():
        target=model/branch/name;target.symlink_to(Path("../..")/branch/path)
        hashes[branch+"/"+name]={"sha256":sha256(target),"bytes":target.stat().st_size}
write_json(a.output_dir/"artifacts.json",{"contract":contract["name"],
"weights":contract["weights"],"models":hashes,"dense_precision":"FP16 I/O except FP32 reliability cues; internal dense graph FP32"})
print("v1_model_ready",model,flush=True)
