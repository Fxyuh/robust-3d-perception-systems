"""Run checkpoint export, native build, ROS2 replay and full mini-val scoring."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from common import sha256, write_json

def ros_environment(setup, extra):
    command = ["bash", "-c", 'source "$1" >/dev/null && env -0', "ros-env", str(setup)]
    blob = subprocess.check_output(command)
    env = dict(item.decode().split("=", 1) for item in blob.split(b"\0") if b"=" in item)
    env["LD_LIBRARY_PATH"] = ":".join(extra) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Continue a known run; completed stages are preserved")
    a = p.parse_args()
    settings = json.loads(a.settings.read_text())
    out = a.output_dir.resolve()
    folder = Path(__file__).resolve().parent
    snapshot = {str(f.relative_to(folder)): sha256(f) for f in folder.rglob("*")
                if f.is_file() and "__pycache__" not in f.parts and f.suffix in (".py", ".cpp", ".sh")}
    settings_hash = sha256(a.settings)
    if a.resume:
        report = json.loads((out / "reproduction.json").read_text())
        if report["settings_sha256"] != settings_hash:
            raise ValueError("Resume requires the same settings content")
        report.setdefault("resume_source_sha256", []).append(snapshot)
    else:
        out.mkdir(parents=True, exist_ok=False)
        report = {"settings_sha256": settings_hash, "checkpoint_sha256": sha256(settings["checkpoint"]),
                  "completed_stages": [], "source_sha256": snapshot}
    logs = out / "logs"
    logs.mkdir(exist_ok=a.resume)
    completed = {stage["name"] for stage in report["completed_stages"]}
    def fresh(path):
        if not path.exists():
            return path
        number = 2
        while path.with_name(path.name + f"_attempt{number}").exists():
            number += 1
        return path.with_name(path.name + f"_attempt{number}")
    def output_path(stage, key, default):
        if stage in completed:
            return Path(report.get(key, str(out / default)))
        path = fresh(out / default)
        report[key] = str(path)
        write_json(out / "reproduction.json", report)
        return path
    py = settings["model_python"]
    def common(work):
        value = ["--work-dir", str(work), "--mmdet3d-root", settings["mmdet3d_root"]]
        for path in settings.get("extra_python_paths", []):
            value += ["--python-path", path]
        return value
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    def run(stage, command, run_env=env):
        if stage in completed:
            print("preserved:", stage, flush=True)
            return
        print("starting:", stage, flush=True)
        started = time.monotonic()
        log_path = fresh(logs / (stage + ".log"))
        with log_path.open("w") as log:
            process = subprocess.run(command, env=run_env, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            print(log_path.read_text()[-6000:], flush=True)
            raise subprocess.CalledProcessError(process.returncode, command)
        report["completed_stages"].append({"name": stage, "seconds": time.monotonic() - started,
                                          "log": str(log_path.relative_to(out)), "source_sha256": snapshot})
        completed.add(stage)
        write_json(out / "reproduction.json", report)
        print("finished:", stage, flush=True)
    export = output_path("export", "export_dir", "export")
    run("export", [py, str(folder / "export_all.py"), *common(export), "--checkpoint", settings["checkpoint"], "--seed", "0"])
    inputs = output_path("prepare", "inputs_dir", "inputs")
    run("prepare", [py, str(folder / "prepare_inputs81.py"), *common(export), "--data-root", settings["data_root"], "--output-dir", str(inputs)])
    core_dir = output_path("core", "core_dir", "core")
    core = [settings.get("system_python", "/usr/bin/python3"), str(folder / "build_native_core.py"),
            "--source-dir", settings["native_source"], "--build-dir", str(core_dir),
            "--cuda-root", settings["cuda_root"], "--tensorrt-include", settings["tensorrt_include"],
            "--tensorrt-lib-dir", settings["tensorrt_lib_dir"], "--protobuf-root", settings["protobuf_root"],
            "--abseil-root", settings["abseil_root"], "--jobs", str(settings.get("jobs", 4))]
    if settings.get("spconv_library"):
        core += ["--spconv-library", settings["spconv_library"]]
    run("core", core)
    paths = json.loads((core_dir / "runtime_paths.json").read_text())
    ros_env = ros_environment(settings["ros_setup"], paths["library_dirs"])
    ros = output_path("ros_build", "ros_dir", "ros")
    run("ros_build", ["colcon", "--log-base", str(ros / "log"), "build",
                     "--base-paths", str(folder / "ros2_ws/src"), "--build-base", str(ros / "build"),
                     "--install-base", str(ros / "install"), "--packages-select", "bevfusion_native_ros2",
                     "--cmake-args", "-DCUDAToolkit_ROOT=" + settings["cuda_root"],
                     "-DBEVFUSION_SOURCE_DIR=" + settings["native_source"],
                     "-DBEVFUSION_CORE_LIBRARY=" + paths["core_library"]], ros_env)
    inference = output_path("replay", "inference_dir", "inference")
    if "replay" not in completed:
        node_binary = ros / "install/bevfusion_native_ros2/lib/bevfusion_native_ros2/bevfusion_native_node"
        node_log = fresh(logs / "node.log")
        with node_log.open("w") as log:
            node = subprocess.Popen([str(node_binary), "--ros-args", "-p", "model_dir:=" + str(export / "model")],
                                    env=ros_env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 120
                while "Native BEVFusion ready" not in node_log.read_text():
                    if node.poll() is not None:
                        raise RuntimeError("Native node stopped: " + node_log.read_text()[-4000:])
                    if time.monotonic() > deadline:
                        raise TimeoutError("Native node did not become ready")
                    time.sleep(0.2)
                run("replay", [settings.get("ros_python", "/usr/bin/python3"), str(folder / "replay_native_ros2_full81.py"),
                               "--input-dir", str(inputs), "--output-dir", str(inference)], ros_env)
            finally:
                if node.poll() is None:
                    node.send_signal(signal.SIGINT)
                    try:
                        node.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        node.kill()
                        node.wait()
    score = output_path("score", "score_dir", "score")
    run("score", [py, str(folder / "score_results.py"), *common(export), "--data-root", settings["data_root"],
                  "--inference-dir", str(inference), "--output-dir", str(score)])
    report["summary"] = json.loads((score / "summary.json").read_text())
    report["artifact_manifest"] = json.loads((export / "artifacts.json").read_text())
    write_json(out / "reproduction.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)
if __name__ == "__main__":
    main()
