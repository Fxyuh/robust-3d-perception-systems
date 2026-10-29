"""Reproduce frozen Robust BEVFusion v1 native deployment and local visualization."""
import argparse,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import sha256,write_json
from pipeline import ros_environment
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--settings",type=Path,required=True)
p.add_argument("--output-dir",type=Path,required=True)
p.add_argument("--resume",action="store_true")
a=p.parse_args()
settings=json.loads(a.settings.read_text());out=a.output_dir.resolve();here=Path(__file__).resolve().parent;parent=here.parent
contract=json.loads((here/"frozen_contract.json").read_text())
for name,key in (("local_gate","fusion_checkpoint"),("lidar_fallback","lidar_checkpoint")):
    if sha256(settings[key])!=contract["weights"][name]["sha256"]:raise ValueError("Frozen weight mismatch: "+key)
if a.resume:
    state=json.loads((out/"pipeline_state.json").read_text())
    if state["settings_sha256"]!=sha256(a.settings):raise ValueError("Settings differ from the original run")
else:
    out.mkdir(parents=True,exist_ok=False)
    state={"settings_sha256":sha256(a.settings),"completed":[],"paths":{},"weights":contract["weights"]}
logs=out/"logs";logs.mkdir(exist_ok=a.resume)
completed=set(state["completed"])
env=os.environ.copy();env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]="1"
py=settings["model_python"]
extra=[]
for path in settings.get("extra_python_paths",[]):extra+=["--python-path",path]
def fresh(path):
    if not path.exists():return path
    i=2
    while path.with_name(path.name+f"_attempt{i}").exists():i+=1
    return path.with_name(path.name+f"_attempt{i}")
def place(stage,name):
    if stage in completed:return Path(state["paths"][name])
    path=fresh(out/name);state["paths"][name]=str(path)
    write_json(out/"pipeline_state.json",state);return path
def run(name,cmd,environment=env):
    if name in completed:print("preserved",name,flush=True);return
    print("starting",name,flush=True);log=fresh(logs/(name+".log"))
    with log.open("w") as file:process=subprocess.run(cmd,env=environment,stdout=file,stderr=subprocess.STDOUT)
    if process.returncode:
        print(log.read_text()[-6000:],flush=True);raise subprocess.CalledProcessError(process.returncode,cmd)
    state["completed"].append(name);completed.add(name);write_json(out/"pipeline_state.json",state)
    print("finished",name,flush=True)
common=["--mmdet3d-root",settings["mmdet3d_root"],"--work-dir",str(out)]
export=place("export","export")
run("export",[py,str(here/"export_v1.py"),"--mmdet3d-root",settings["mmdet3d_root"],
              "--fusion-checkpoint",settings["fusion_checkpoint"],"--lidar-checkpoint",settings["lidar_checkpoint"],
              "--output-dir",str(export),*extra])
inputs=place("prepare","inputs")
run("prepare",[py,str(here/"prepare_inputs.py"),*common,"--data-root",settings["data_root"],"--output-dir",str(inputs)])
core=place("core","core")
cmd=[settings.get("system_python","/usr/bin/python3"),str(parent/"build_native_core.py"),"--source-dir",settings["native_source"],
     "--build-dir",str(core),"--cuda-root",settings["cuda_root"],"--tensorrt-include",settings["tensorrt_include"],
     "--tensorrt-lib-dir",settings["tensorrt_lib_dir"],"--protobuf-root",settings["protobuf_root"],
     "--abseil-root",settings["abseil_root"],"--jobs",str(settings.get("jobs",4))]
if settings.get("spconv_library"):cmd+=["--spconv-library",settings["spconv_library"]]
run("core",cmd)
runtime=json.loads((core/"runtime_paths.json").read_text());ros_env=ros_environment(settings["ros_setup"],runtime["library_dirs"])
ros=place("ros_build","ros")
run("ros_build",["colcon","--log-base",str(ros/"log"),"build","--base-paths",str(parent/"ros2_ws/src"),
                "--build-base",str(ros/"build"),"--install-base",str(ros/"install"),
                "--packages-select","bevfusion_native_ros2","--cmake-args","-DCUDAToolkit_ROOT="+settings["cuda_root"],
                "-DBEVFUSION_SOURCE_DIR="+settings["native_source"],"-DBEVFUSION_CORE_LIBRARY="+runtime["core_library"]],ros_env)
cues=place("cues_compile","cues_check")
run("cues_compile",["g++","-std=c++17","-O2",str(here/"cues_check.cpp"),"-o",str(cues)])
parity=place("parity","parity")
run("parity",[py,str(here/"check_parity.py"),*common,"--checkpoint",settings["fusion_checkpoint"],
              "--input-dir",str(inputs),"--cue-binary",str(cues),"--gate-engine",str(export/"model/fusion/fuser.engine"),
              "--output-dir",str(parity),*extra])
input_check=place("input_check","input_parity.json")
run("input_check",[py,str(here/"check_inputs.py"),*common,"--data-root",settings["data_root"],
                   "--input-dir",str(inputs),"--output",str(input_check)])
validation=place("validation","validation")
run("validation",[settings.get("monitor_python",py),str(here/"run_validation.py"),"--export-dir",str(export),
                  "--input-dir",str(inputs),"--ros-install-dir",str(ros/"install"),"--core-runtime-json",str(core/"runtime_paths.json"),
                  "--mmdet3d-root",settings["mmdet3d_root"],"--data-root",settings["data_root"],"--model-python",py,
                  "--ros-python",settings.get("ros_python","/usr/bin/python3"),"--ros-setup",settings["ros_setup"],
                  "--output-dir",str(validation)])
viewer=place("viewer","visualization")
run("viewer",[py,str(here/"make_viewer.py"),"--input-dir",str(inputs),"--validation-dir",str(validation),
              "--ann-file",settings.get("ann_file",str(Path(settings["data_root"])/"nuscenes_infos_val.pkl")),
              "--output-dir",str(viewer)])
print("Open the local replay after starting:",flush=True)
print(f"python3 -m http.server 8766 --bind 127.0.0.1 --directory {viewer}",flush=True)
