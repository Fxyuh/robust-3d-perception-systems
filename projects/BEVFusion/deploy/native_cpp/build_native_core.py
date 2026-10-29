"""Build the pinned native core using explicitly supplied, existing dependencies."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from common import sha256, write_json

VENDOR_COMMIT = "7c1623f234d5a9794fd862c9456f68f87ae7b900"
SPCONV_SHA256 = "5e4e58136e7962f23c4e68a869ba0f3ada20502996eefd2a616293d34bc2ae9d"

def regenerate_protobuf(source, build, protobuf, env):
    generated = build / "pb_generated"
    generated.mkdir()
    source = source / "CUDA-BEVFusion/src/onnx"
    backup = build / "vendor_pb_backup"
    backup.mkdir()
    names = ("onnx-ml", "onnx-operators-ml")
    for name in names:
        for suffix in (".pb.cpp", ".pb.h"):
            shutil.copy2(source / (name + suffix), backup / (name + suffix))
    subprocess.run([str(protobuf / "bin/protoc"), "-I", str(source), "--cpp_out=" + str(generated),
                    *[str(source / (name + ".proto")) for name in names]], env=env, check=True)
    for name in names:
        for source_suffix, dest_suffix in ((".pb.cc", ".pb.cpp"), (".pb.h", ".pb.h")):
            shutil.copy2(generated / (name + source_suffix), source / (name + dest_suffix))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "build-dir", "cuda-root", "tensorrt-include", "tensorrt-lib-dir", "protobuf-root", "abseil-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--spconv-library", type=Path)
    p.add_argument("--cuda-sm", default="120")
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    source, build = a.source_dir.resolve(), a.build_dir.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != VENDOR_COMMIT:
        raise ValueError(f"Expected NVIDIA source {VENDOR_COMMIT}, found {revision}")
    if a.cuda_sm != "120":
        raise ValueError("This recipe has been validated for compute capability 12.0 only")
    spconv = source / "libraries/3DSparseConvolution/libspconv/lib/x86_64_cuda12.8/libspconv.so"
    if a.spconv_library:
        if sha256(a.spconv_library) != SPCONV_SHA256:
            raise ValueError("libspconv SHA-256 mismatch")
        if sha256(spconv) != SPCONV_SHA256:
            if not spconv.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise FileExistsError(f"Refusing to replace unexpected library: {spconv}")
            shutil.copy2(a.spconv_library, spconv)
    if sha256(spconv) != SPCONV_SHA256:
        raise ValueError("Fetch the verified CUDA 12.8 libspconv object first")
    build.mkdir(parents=True, exist_ok=False)
    cuda, protobuf, abseil = a.cuda_root.resolve(), a.protobuf_root.resolve(), a.abseil_root.resolve()
    trt = build / "trt_lib"
    trt.mkdir()
    for name in ("nvinfer", "nvinfer_plugin"):
        choices = sorted(a.tensorrt_lib_dir.resolve().glob("lib" + name + ".so*"))
        if not choices:
            raise FileNotFoundError("TensorRT library " + name)
        (trt / ("lib" + name + ".so")).symlink_to(choices[0].resolve())
    cuda_include = cuda / "targets/x86_64-linux/include"
    cuda_lib = cuda / "targets/x86_64-linux/lib"
    if not cuda_include.exists():
        cuda_include, cuda_lib = cuda / "include", cuda / "lib64"
    lib_dirs = [build, spconv.parent, a.tensorrt_lib_dir.resolve(), protobuf / "lib", abseil / "lib", cuda / "lib", cuda_lib]
    env = os.environ.copy()
    env.update(USE_Python="OFF", SPCONV_CUDA_VERSION="12.8", CUDASM=a.cuda_sm,
               CUDA_Inc=str(cuda_include), CUDA_Lib=str(cuda_lib),
               CUDA_INC_PATH=str(cuda_include), CUDA_LIB_PATH=str(cuda_lib),
               TensorRT_Inc=str(a.tensorrt_include.resolve()), TensorRT_Lib=str(trt),
               LD_LIBRARY_PATH=":".join(map(str, lib_dirs)) + ":" + env.get("LD_LIBRARY_PATH", ""))
    regenerate_protobuf(source, build, protobuf, env)
    command = ["cmake", "-S", str(source / "CUDA-BEVFusion"), "-B", str(build),
               "-DCUDA_TOOLKIT_ROOT_DIR=" + str(cuda), "-DCUDA_NVCC_EXECUTABLE=" + str(cuda / "bin/nvcc"),
               "-DCUDA_TOOLKIT_INCLUDE=" + str(cuda_include),
               "-DProtobuf_INCLUDE_DIR=" + str(protobuf / "include"),
               "-DProtobuf_LIBRARY=" + str(protobuf / "lib/libprotobuf.so"),
               "-DProtobuf_PROTOC_EXECUTABLE=" + str(protobuf / "bin/protoc"),
               "-DCMAKE_CXX_FLAGS=-I" + str(abseil / "include"),
               "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link," + str(abseil / "lib")]
    subprocess.run(command, env=env, check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "bevfusion_core", "--parallel", str(a.jobs)], env=env, check=True)
    if not (build / "libbevfusion_core.so").is_file():
        raise FileNotFoundError("Core library was not built")
    write_json(build / "runtime_paths.json", {"source_dir": str(source), "core_library": str(build / "libbevfusion_core.so"),
               "cuda_root": str(cuda), "library_dirs": list(map(str, lib_dirs)), "vendor_commit": revision,
               "libspconv_sha256": SPCONV_SHA256})
if __name__ == "__main__":
    main()
