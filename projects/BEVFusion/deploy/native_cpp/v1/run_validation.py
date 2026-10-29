"""Validate both frozen-v1 native routes, replay all conditions and score them."""
import sys,argparse,json,os,subprocess,signal,time,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import write_json,sha256
from pipeline import ros_environment
p=argparse.ArgumentParser(description=__doc__)
for name in ("export-dir","input-dir","ros-install-dir","core-runtime-json","mmdet3d-root","data-root","output-dir"):
    p.add_argument("--"+name,type=Path,required=True)
p.add_argument("--model-python",required=True)
p.add_argument("--ros-python",default="/usr/bin/python3")
p.add_argument("--ros-setup",default="/opt/ros/humble/setup.bash")
a=p.parse_args()
here=Path(__file__).resolve().parent;parent=here.parent
a.output_dir.mkdir(parents=True,exist_ok=False);logs=a.output_dir/"logs";logs.mkdir()
paths=json.loads(a.core_runtime_json.read_text());env=ros_environment(a.ros_setup,paths["library_dirs"])
conditions=["normal","camera_drop_3_lidar_drop_75","camera_drop_6_lidar_drop_75"]
def call(name,cmd,environment):
    with (logs/(name+".log")).open("w") as log:
        process=subprocess.run(cmd,env=environment,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:
        print((logs/(name+".log")).read_text()[-6000:],flush=True)
        raise subprocess.CalledProcessError(process.returncode,cmd)
    print("finished",name,flush=True)
memory={"available":False};samples=[];process_samples=[];stop=threading.Event()
try:
    import pynvml as nv
    nv.nvmlInit();handle=nv.nvmlDeviceGetHandleByIndex(0)
    baseline=int(nv.nvmlDeviceGetMemoryInfo(handle).used)
    memory.update(available=True,baseline_device_gib=baseline/2**30)
except (ImportError,Exception) as error:
    nv=None;memory["reason"]=type(error).__name__
node_binary=a.ros_install_dir/"bevfusion_native_ros2/lib/bevfusion_native_ros2/bevfusion_v1_node"
def monitor(pid):
    while not stop.is_set():
        try:samples.append(int(nv.nvmlDeviceGetMemoryInfo(handle).used))
        except Exception:pass
        try:
            for item in nv.nvmlDeviceGetComputeRunningProcesses(handle):
                value=item.usedGpuMemory
                if item.pid==pid and value is not None and 0<value<2**50:process_samples.append(int(value))
        except Exception:pass
        stop.wait(0.005)
with (logs/"node.log").open("w") as log:
    node=subprocess.Popen([str(node_binary),"--ros-args","-p","model_dir:="+str(a.export_dir/"model")],
                          env=env,stdout=log,stderr=subprocess.STDOUT)
    thread=threading.Thread(target=monitor,args=(node.pid,),daemon=True) if nv else None
    if thread:thread.start()
    try:
        deadline=time.monotonic()+120
        while "Frozen v1 native ready" not in (logs/"node.log").read_text():
            if node.poll() is not None:raise RuntimeError((logs/"node.log").read_text()[-6000:])
            if time.monotonic()>deadline:raise TimeoutError("v1 node startup timeout")
            time.sleep(0.2)
        for condition in conditions:
            target=a.output_dir/"smoke"/condition
            call("smoke_"+condition,[a.ros_python,str(here/"replay.py"),"--input-dir",str(a.input_dir/condition),
                 "--output-dir",str(target),"--max-samples","1"],env)
            result=json.loads((target/"predictions/000.json").read_text())
            expected="lidar_fallback" if condition.startswith("camera_drop_6") else "gated_fusion"
            if result["route"]!=expected or result["num_detections"]<1:
                raise ValueError("Route smoke check failed")
        for condition in conditions:
            call("infer_"+condition,[a.ros_python,str(here/"replay.py"),"--input-dir",str(a.input_dir/condition),
                 "--output-dir",str(a.output_dir/"inference"/condition)],env)
    finally:
        if node.poll() is None:
            node.send_signal(signal.SIGINT)
            try:node.wait(timeout=10)
            except subprocess.TimeoutExpired:node.kill();node.wait()
        stop.set()
        if thread:thread.join(timeout=5)
        if nv:nv.nvmlShutdown()
memory.update(sample_interval_ms=5,device_samples=len(samples),process_samples=len(process_samples),
              process_peak_gib=max(process_samples)/2**30 if process_samples else None,
              whole_device_peak_increase_gib=(max(samples)-baseline)/2**30 if samples else None,
              scope="Both experts resident, startup + 3 smoke frames + 3x81 inference; whole-device delta is not process peak")
write_json(a.output_dir/"memory.json",memory)
contract=json.loads((here/"frozen_contract.json").read_text())
rows=[]
for condition in conditions:
    score=a.output_dir/"score"/condition
    model_env=os.environ.copy();model_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]="1"
    call("score_"+condition,[a.model_python,str(parent/"score_results.py"),"--work-dir",str(a.output_dir),
        "--mmdet3d-root",str(a.mmdet3d_root),"--data-root",str(a.data_root),
        "--inference-dir",str(a.output_dir/"inference"/condition),"--output-dir",str(score),
        *(["--lidar-only"] if condition.startswith("camera_drop_6") else [])],model_env)
    result=json.loads((score/"summary.json").read_text())
    frozen=next(row for row in contract["table"] if row["condition"]==condition)
    rows.append({"condition":condition,"native_v1":result,"frozen_pytorch_v1":frozen["robust_bevfusion_v1"],
                 "frozen_plain":frozen["plain"],
                 "native_minus_frozen_v1_mAP":result["mAP"]-frozen["robust_bevfusion_v1"]["mAP"],
                 "native_minus_frozen_plain_mAP":result["mAP"]-frozen["plain"]["mAP"]})
summary={"method":contract["name"],"protocol":contract["evaluation"],"warmup":"one smoke frame per route/condition before the 81-frame timing runs",
         "weights":contract["weights"],"results":rows,"memory":memory,
         "comparison_scope":"Historical frozen PyTorch results versus native backend; native-minus-Plain is descriptive across backends/training protocols"}
write_json(a.output_dir/"comparison.json",summary)
print(json.dumps(summary,indent=2),flush=True)
