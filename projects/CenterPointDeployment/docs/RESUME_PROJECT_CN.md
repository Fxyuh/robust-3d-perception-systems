# CenterPoint 部署项目简历表述

## 推荐项目名

**CenterPoint 三维目标检测部署与 ROS 2 多目标跟踪系统**

技术栈：C++17、CUDA C++、TensorRT、ONNX、ROS 2 Humble、Eigen、CMake、
PyTorch、MMDetection3D、nuScenes。

## 一页简历版本

- 基于 MMDetection3D 官方 CenterPoint 权重，完成从 ROS 2 `PointCloud2`
  输入到三维检测框输出的 C++ 部署链路，实现确定性体素化、CUDA
  PillarFeatureNet/Scatter、TensorRT 推理以及 C++ 解码与 Circle NMS，部署
  可执行程序不依赖 Python/PyTorch 运行时。

- 建立 PyTorch—TensorRT 数值一致性验证流程，针对真实 nuScenes mini
  样本逐级校验中间特征和最终检测框；在置信度阈值 0.2 下两端均输出
  66 个目标并实现 100% 匹配，框中心平均/最大误差为 0.127/0.591 mm。
  在 RTX 5070 Ti 上连续测试 50 次，点云到检测框平均耗时 12.17 ms、
  P95 13.45 ms，吞吐率 82.16 FPS。

- 使用 nuScenes 官方工具完成全部 81 帧 mini-val 检测评测，TensorRT C++
  取得 mAP 0.4127、NDS 0.4803，相对 PyTorch 基准分别变化 +0.0017/
  -0.0017；跨帧推理延迟 P50/P95 为 15.00/16.02 ms。以 1 ms 周期从
  CUDA 初始化前持续采样 NVML，实测包含 TensorRT 内部空间的设备显存峰值
  增量为 693.91 MiB。

- 搭建 ROS 2 感知节点、结构化检测/跟踪消息和 RViz 可视化，支持
  nuScenes 连续场景按时间戳回放、LiDAR 到全局坐标位姿同步，以及三维框、
  稳定 ID 和历史轨迹发布；提供 40/41 帧两个 mini-val 场景的可复现实验
  入口和一键演示脚本。

- 实现全局坐标系恒速 Kalman 多目标跟踪和类别感知 Hungarian 关联，并用
  nuScenes 官方工具评测全部 81 个 mini-val 帧，取得 AMOTA 0.655、AMOTP
  0.568 m、MOTA 0.629 和 58 次 ID Switch；通过按类别分块求解，在官方
  指标完全不变的情况下将平均关联/跟踪耗时由 4.728 ms 降至 2.973 ms，
  降低 37.1%。

## 30 秒口述版

这个项目不是重新训练一个 CenterPoint，而是把官方 PyTorch 模型做成可验证
的 C++ 感知系统。我实现了点云体素化、CUDA 柱特征编码、TensorRT 混合精度
推理、C++ 后处理和 ROS 2 接口，并通过逐级对齐保证部署框与 PyTorch 结果
一致。在此基础上，我又实现了全局坐标系 Kalman/Hungarian 跟踪，用官方
nuScenes mini tracking 指标验证，并通过类别分块在不损失指标的情况下把
跟踪耗时降低了 37.1%。

## 表述边界

- 使用的是官方 CenterPoint 架构与预训练权重，个人工作重点是部署、验证、
  系统集成和跟踪优化，不应表述为“自研 CenterPoint 算法”。
- 12.17 ms 和 82.16 FPS 来自 RTX 5070 Ti 上一个真实样本的 50 次稳态测试，
  81 帧 mini-val 的跨帧 P50/P95 则为 15.00/16.02 ms；两者均不包含点云
  文件读取时间。
- WSL2 未提供 NVML Linux 子进程显存记账，因此 693.91 MiB 是无并发负载时
  的设备基线到峰值差值，不应表述为原生 per-PID 独占显存。
- AMOTA/AMOTP 来自 nuScenes `v1.0-mini` 的全部 81 个验证帧，不能写成
  nuScenes 完整 validation 或 test 榜单成绩。
