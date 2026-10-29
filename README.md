# Robust 3D Perception Systems

[![Portfolio CI](https://github.com/Fxyuh/robust-3d-perception-systems/actions/workflows/portfolio-ci.yml/badge.svg)](https://github.com/Fxyuh/robust-3d-perception-systems/actions/workflows/portfolio-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A portfolio project for camera/LiDAR 3D perception. The BEVFusion milestone
covers native C++/CUDA/TensorRT inference, a ROS2 adapter and reproducible
mini-val evaluation. The CenterPoint milestone covers LiDAR detection,
visualization and multi-object tracking.

## BEVFusion Camera + LiDAR deployment

- Export and interface adaptation from the project's Robust epoch-4 checkpoint.
- Four TensorRT engines plus NVIDIA's native CUDA sparse-convolution runtime.
- C++ ROS2 inference node and automated export/build/replay/evaluation workflow.
- Clean official MMDetection3D v1.4.0 source verification using existing dependencies.

| Backend (81 mini-val frames) | mAP | NDS | Model P50 / P95 |
| --- | ---: | ---: | ---: |
| Historical PyTorch baseline | 0.535668 | 0.554942 | 188.949 / 196.393 ms |
| Native C++/CUDA/TensorRT + ROS2 | 0.530959 | 0.550292 | 29.946 / 31.144 ms |

ROS2 round-trip P50/P95 is 45.341/55.563 ms. The node accepts preprocessed
six-camera/LiDAR/calibration bundles. Raw sensor-topic synchronization is outside
the current interface. Timings exclude offline image preprocessing and come
from separate local runs; no controlled speedup claim is made.

See the [BEVFusion reproduction guide](projects/BEVFusion/deploy/native_cpp/README.md),
[machine-readable results](projects/BEVFusion/deploy/native_cpp/results/) and
[attribution](projects/BEVFusion/deploy/native_cpp/THIRD_PARTY_NOTICES.md).

## CenterPoint LiDAR deployment

[![ROS 2 detection and tracking demo](projects/CenterPointDeployment/docs/assets/centerpoint_ros2_tracking.png)](projects/CenterPointDeployment/docs/assets/centerpoint_ros2_tracking_public.mp4)

## CenterPoint implementation

- Deterministic C++ voxelization and CUDA pillar feature encoding/scatter.
- Mixed-FP16 TensorRT inference with native CenterPoint decoding and circle NMS.
- Numerical validation from intermediate tensors through final 3D boxes.
- ROS 2 messages, replay nodes, RViz visualization and launch workflows.
- Global-frame Kalman tracking with class-wise Hungarian assignment.
- Full nuScenes mini-val detection/tracking evaluation, latency percentiles and
  process-level GPU-memory measurement.

## CenterPoint verified results

| Result | Measured value | Scope |
|---|---:|---|
| TensorRT C++ detection | 0.412667 mAP / 0.480336 NDS | All 81 mini-val frames |
| PyTorch detection baseline | 0.410953 mAP / 0.482046 NDS | Same checkpoint and protocol |
| Detector latency | 14.996 ms P50 / 16.024 ms P95 | 81 mini-val frames |
| Deployment GPU memory | 693.91 MiB NVML peak delta | Includes TensorRT allocations |
| Multi-object tracking | 0.655 AMOTA / 0.568 mAMOTP | Official mini-val evaluation |

These are nuScenes `v1.0-mini` results, not full nuScenes validation results.

## Repository map

```text
configs/centerpoint/                       mini dataset configuration overlay
projects/CenterPointDeployment/cpp/       C++/CUDA/TensorRT detector
projects/CenterPointDeployment/ros2/      ROS 2 integration and tracker
projects/CenterPointDeployment/tools/     export, validation and evaluation tools
projects/CenterPointDeployment/results/   machine-readable benchmark summaries
projects/BEVFusion/deploy/native_cpp/      native fusion deployment and reproduction
```

See the [implementation and reproduction guide](projects/CenterPointDeployment/README.md),
[resume-ready project summary](projects/CenterPointDeployment/docs/RESUME_PROJECT_CN.md)
and [third-party attribution](projects/CenterPointDeployment/THIRD_PARTY_NOTICES.md).

## Integration with MMDetection3D

This is a project-specific integration layer rather than a vendored copy of
MMDetection3D. Use it with a separate MMDetection3D v1.4.0 checkout. The
existing tools expect the project directory and mini configuration to be
visible from that checkout:

```bash
export PORTFOLIO_ROOT=/absolute/path/to/robust-3d-perception-systems
export MMDET3D_ROOT=/absolute/path/to/mmdetection3d

ln -s "${PORTFOLIO_ROOT}/projects/CenterPointDeployment" \
  "${MMDET3D_ROOT}/projects/CenterPointDeployment"
ln -s "${PORTFOLIO_ROOT}/configs/centerpoint/centerpoint_pillar02_nuscenes_mini_1gpu.py" \
  "${MMDET3D_ROOT}/configs/centerpoint/centerpoint_pillar02_nuscenes_mini_1gpu.py"
```

Then follow the reproduction guide from `${MMDET3D_ROOT}`. Dataset files,
checkpoints, exported ONNX graphs and TensorRT engines are intentionally not
redistributed.

## Scope and attribution

CenterPoint, BEVFusion, MMDetection3D, official configurations/pretrained models
and NVIDIA CUDA-BEVFusion are upstream work. This repository covers checkpoint
export and interface adaptation, deployment/reproduction tooling, ROS2 adapters,
tracking implementation and systems evaluation. See the
[third-party notices](projects/CenterPointDeployment/THIRD_PARTY_NOTICES.md)
for the complete dependency scope.
