# Robust 3D Perception Systems

[![Portfolio CI](https://github.com/Fxyuh/robust-3d-perception-systems/actions/workflows/portfolio-ci.yml/badge.svg)](https://github.com/Fxyuh/robust-3d-perception-systems/actions/workflows/portfolio-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A portfolio project for deployable LiDAR 3D perception. The current milestone
takes an MMDetection3D CenterPoint checkpoint through a verified
C++17/CUDA/TensorRT inference path, ROS 2 visualization, global-frame
multi-object tracking and system-level benchmarking.

[![ROS 2 detection and tracking demo](projects/CenterPointDeployment/docs/assets/centerpoint_ros2_tracking.png)](projects/CenterPointDeployment/docs/assets/centerpoint_ros2_tracking_public.mp4)

## What I implemented

- Deterministic C++ voxelization and CUDA pillar feature encoding/scatter.
- Mixed-FP16 TensorRT inference with native CenterPoint decoding and circle NMS.
- Numerical validation from intermediate tensors through final 3D boxes.
- ROS 2 messages, replay nodes, RViz visualization and launch workflows.
- Global-frame Kalman tracking with class-wise Hungarian assignment.
- Full nuScenes mini-val detection/tracking evaluation, latency percentiles and
  process-level GPU-memory measurement.

## Verified results

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

CenterPoint, MMDetection3D, the official configuration and pretrained model are
upstream work. The code in this repository covers the deployment boundary,
validation tooling, ROS 2 integration, tracking implementation and systems
evaluation. See the
[third-party notices](projects/CenterPointDeployment/THIRD_PARTY_NOTICES.md)
for the complete dependency scope.
