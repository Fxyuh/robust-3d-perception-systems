# BEVFusion native inference and ROS2 reproduction

This integration exports a user-supplied MMDetection3D BEVFusion Camera + LiDAR checkpoint into four TensorRT engines and a native CUDA sparse-convolution model, then runs the complete model forward path in a C++ ROS2 node. PyTorch is used for export and offline evaluation. The detector architecture is the official MMDetection3D v1.4.0 model.

The runtime uses NVIDIA CUDA-BEVFusion at commit `7c1623f234d5a9794fd862c9456f68f87ae7b900`. See [third-party attribution](THIRD_PARTY_NOTICES.md). No detector, weights, dataset, engine or native third-party library is vendored here.

## Reproduce with existing dependencies

You need a separate MMDetection3D v1.4.0 source checkout with the official BEVFusion project operators already compiled, nuScenes **v1.0-mini** and its information files, and a compatible Camera + LiDAR checkpoint. This recipe was validated on WSL2, RTX 5070 Ti (SM120), CUDA 12.8 and ROS2 Humble using the existing Python 3.10/MMDetection3D environment. The scripts do not install or upgrade packages.

1. Create a dedicated checkout of [Lidar_AI_Solution](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution) at the pinned commit above. Use a separate checkout because the build regenerates its protobuf files and keeps backups in the new build directory.
2. Fetch/verify its CUDA 12.8 sparse-convolution library:
   ```bash
   python3 projects/BEVFusion/deploy/native_cpp/fetch_verified_libspconv.py \
     --source-dir "$NATIVE_SOURCE"
   ```
   The SHA-256 must be `5e4e58136e7962f23c4e68a869ba0f3ada20502996eefd2a616293d34bc2ae9d`. Alternatively set `spconv_library` in the settings to an existing verified copy.
3. Copy [settings.example.json](settings.example.json) outside this repository and fill in the absolute paths to **your existing** dependencies, dataset and checkpoint. `extra_python_paths` is optional if ONNX/TensorRT are already importable in `model_python`. TensorRT headers must match the runtime libraries. CUDA, protobuf and abseil roots must contain their include/lib/bin folders.
4. Run from any working directory, choosing a new output directory:
   ```bash
   python3 /path/to/portfolio/projects/BEVFusion/deploy/native_cpp/pipeline.py \
     --settings /path/to/settings.local.json \
     --output-dir /path/to/new-bevfusion-run
   ```

The pipeline executes:

```text
same checkpoint -> image/depth + sparse LiDAR + dense BEV + head ONNX exports
                -> interface conversion + four TensorRT engines
official mini pipeline -> normalized six-camera/LiDAR/calibration packets
pinned NVIDIA source + existing dependencies -> C++ CUDA core -> ROS2 node
81 packets -> ROS2 replay -> saved predictions -> official nuScenes metrics
```

It records command logs under `logs/`, checkpoint/engine SHA-256 values in `export/artifacts.json`, completed stages in `reproduction.json`, frame timings in `inference/frame_timings.csv` and mAP/NDS in `score/summary.json`. Outputs are created in a new directory; an existing run directory is rejected. On failure, inspect the stage log. After fixing the issue, run the same command with --resume and unchanged settings: completed stages are preserved, failed output directories/logs are kept, and retried stages use new attempt directories. Source hashes are recorded for each completed stage.

The build receives CUDA, TensorRT, protobuf, abseil, vendor source and output paths from the settings. The ROS2 CMake package takes `BEVFUSION_SOURCE_DIR`, `BEVFUSION_CORE_LIBRARY` and `CUDAToolkit_ROOT` as parameters. There are no home-directory or earlier CenterPoint-build defaults.

## Scope of the deployed node

- Input topic: `/bevfusion/native_input`, `std_msgs/UInt8MultiArray`.
- Output topic: `/bevfusion/native_detections_json`, `std_msgs/String`.
- Input packet: versioned `BFN1` header, six normalized BGR images `[6,3,256,704]` FP16, nonempty `[N,5]` LiDAR points FP16, and four `[6,4,4]` FP32 calibration arrays. Labels are never sent to inference.
- C++/CUDA: voxelization, sparse convolution, LiDAR-to-camera depth projection, geometry/BEV pooling and box decoding.
- TensorRT: camera backbone/neck/depth network, camera BEV downsample, fusion/SECOND/SECONDFPN and TransFusion detection head. Engine boundaries are FP16; the exported dense graphs contain FP32 computation. This does not imply every TensorRT layer executes in FP16.
- The node accepts model-ready bundles. Raw six-camera/`PointCloud2` topic subscription, timestamp synchronization and in-node image preprocessing are outside the current interface.
- The fixed runtime assumes the official 6-camera, 256x704, voxel-0.075, top-200 configuration. It is not an exporter for arbitrary BEVFusion variants.

## Evidence and measurement

The independent clean-source rebuild completed all six stages with 81 frames:

| Backend | mAP | NDS | Model P50 / P95 | ROS2 round-trip P50 / P95 |
| --- | ---: | ---: | ---: | ---: |
| Historical PyTorch baseline | 0.535668 | 0.554942 | 188.949 / 196.393 ms | n/a |
| Native clean-source ROS2 rebuild | 0.530959 | 0.550292 | 29.946 / 31.144 ms | 45.341 / 55.563 ms |

The native result differs from the historical PyTorch baseline by -0.004710 mAP and -0.004650 NDS. These values verify deployment fidelity; they are not an algorithm improvement. The clean rebuild was completed using resume after two dependency/path corrections; the original failed logs and artifacts were retained.


Small, machine-readable summaries are in [results/](results/); checkpoints, binary inputs, engines and full prediction files are excluded from Git. The supplied checkpoint in the measured experiment is the project's Robust epoch-4 checkpoint. Users must provide their own checkpoint and data when reproducing.

Native `model_ms` includes calibration update, host-to-device copies, native CUDA/TensorRT forward and C++ decoding. ROS2 round-trip also includes model-ready packet transport and the JSON response. Both exclude offline image resize/normalization, dataset decoding and disk reads. Timing percentiles cover all 81 frames including the cold first frame. Historical PyTorch and native timing runs were separate local runs and do not establish a controlled speedup.

The historical NVML experiment measured a **2.185 GiB whole-device increase above baseline** during 81-frame C++ inference. WSL2 returned no per-process NVML samples. This value must not be reported as total process peak VRAM.

The clean-source verification uses a fresh official v1.4.0 source tree and the existing compiled operators/dependency environment. It verifies source and path independence, not installation on a new machine. All reported accuracy values are **mini-val**, not full nuScenes validation.

## Individual tools

Every entry point supports `--help`.

- `export_all.py --mmdet3d-root ... --checkpoint ... --work-dir ... [--python-path ...]`: export one checkpoint and build engines in a new directory.
- `prepare_inputs81.py --mmdet3d-root ... --data-root ... --work-dir ... --output-dir ...`: prepare binary inputs and their manifest from the official pipeline.
- `build_native_core.py` / `build_native_core.sh`: build with explicit dependency paths and verified sparse-convolution library.
- `replay_native_ros2_full81.py --input-dir ... --output-dir ...`: replay through the running C++ node; `--max-samples 1` is available for a smoke check.
- `score_results.py --mmdet3d-root ... --data-root ... --work-dir ... --inference-dir ... --output-dir ...`: score all mini-val frames with the official evaluator.
- `native_batch.cpp` and `parse_native_batch.py`: optional direct 81-frame C++ runner and log-to-evaluator conversion.

The ROS2 node requires its exported model directory: `--ros-args -p model_dir:=/absolute/path/to/export/model`. The pipeline builds, starts and stops this node automatically.
