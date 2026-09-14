# Third-party scope and attribution

This repository is a standalone integration layer used with MMDetection3D and
is released under the repository-level Apache-2.0 license. Its deployment,
validation, ROS 2 integration and tracking code is maintained as
project-specific work.

The project relies on the following upstream components:

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) provides the
  CenterPoint model definition, official configuration and pretrained
  checkpoint used as the PyTorch reference. Model weights are not redistributed.
- [CenterPoint](https://github.com/tianweiy/CenterPoint) is the original model
  publication and implementation. This project does not claim authorship of
  the CenterPoint architecture.
- [NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT) provides runtime APIs
  and C++ headers. The helper script fetches headers from a pinned official
  revision for reproducibility; TensorRT source and binaries are not vendored.
- [nuScenes](https://www.nuscenes.org/) and the nuScenes devkit provide the
  dataset and official detection/tracking evaluators. Dataset files are not
  redistributed.
- [ROS 2](https://www.ros.org/) provides middleware, message generation and
  visualization integration.

Each upstream project and dataset remains subject to its own license and terms.
Generated ONNX graphs, TensorRT engines, checkpoints, point clouds and complete
evaluation outputs are intentionally excluded from version control.
