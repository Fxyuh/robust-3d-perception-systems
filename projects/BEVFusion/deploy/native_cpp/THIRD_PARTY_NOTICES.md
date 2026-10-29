# Third-party components

- MMDetection3D v1.4.0 and its official BEVFusion implementation/configuration: Apache-2.0, https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0/projects/BEVFusion . This repository does not vendor the detector or claim authorship of BEVFusion/TransFusion.
- NVIDIA Lidar_AI_Solution / CUDA-BEVFusion, commit `7c1623f234d5a9794fd862c9456f68f87ae7b900`: MIT. https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution . Its native sparse-convolution runtime and CUDA voxelization, depth, geometry, pooling and decoder code execute the corresponding stages. The checkout and its license remain external dependencies.
- TensorRT, CUDA, ROS2 Humble, PyTorch, MMCV, MMEngine, MMDetection and nuScenes devkit/dataset retain their respective licenses. No datasets or pretrained weights are redistributed.

The work delivered here is the checkpoint export and interface adaptation, reproducibility tooling, C++ ROS2 adapter, input replay, precision/latency validation and reporting. The model architecture and underlying native inference algorithms are upstream work. TensorRT engines are compiled locally from a user-supplied checkpoint.
