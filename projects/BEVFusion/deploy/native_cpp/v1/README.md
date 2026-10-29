# Frozen Robust BEVFusion v1: native deployment and visualization

This is the deployed **frozen v1** method: the reliability-gated Camera + LiDAR model (`iter_242.pth`) is used when any camera is available; the official LiDAR-only model is selected when all six cameras fail. It is a different model from the earlier sensor-dropout Robust epoch-4 deployment in the parent folder.

The [frozen contract](frozen_contract.json) records the exact two weight hashes and historical PyTorch results. Export rejects other weights. Architecture, degradation seed and route policy are preserved. No training is performed by these tools.

## Watch or try the demo

[![Real native v1 inference replay](docs/assets/bevfusion_v1_demo.gif)](docs/assets/bevfusion_v1_demo.mp4)

[MP4 video](docs/assets/bevfusion_v1_demo.mp4) · [Interactive demo ZIP](docs/demo/bevfusion_v1_interactive.zip?raw=true) · [Single HTML](docs/demo/bevfusion_v1_interactive.html?raw=true)

The README GIF previews three actual deployment conditions; the MP4 is a 1280x720 H.264 clip of about 19 seconds. Download the ZIP, extract it, then double-click `bevfusion_v1_interactive.html` in your browser. It contains 12 selected frames per condition, supports orbit/playback/condition selection and works without a server or installed model. GitHub's repository view does not execute HTML; the file must be downloaded and opened locally. The full 81-frame local replay remains available through the generated viewer.

These are compact **derived visualization assets**, with additional display-only point/image compression. They are separate from the complete dataset/inference bundles and never enter evaluation. See [the demo manifest](docs/demo/demo_manifest.json) for provenance, video properties and hashes.

## Measured native results

Each row uses all 81 nuScenes v1.0-mini val frames and degradation seed 0.

| Condition | Frozen PyTorch v1 mAP / NDS | Native v1 mAP / NDS | Native model P50 / P95 |
| --- | ---: | ---: | ---: |
| Normal | 0.5680 / 0.5761 | 0.5697 / 0.5765 | 43.7 / 45.5 ms |
| 3 cameras off + 75% LiDAR dropout | 0.4669 / 0.5082 | 0.4628 / 0.5056 | 32.9 / 34.1 ms |
| 6 cameras off + 75% LiDAR dropout | 0.4361 / 0.4970 | 0.4327 / 0.4953 | 13.1 / 13.9 ms |

Under complete camera failure plus 75% LiDAR dropout, the frozen Plain reference is 0.3630 mAP. The historical PyTorch v1 gain is **+0.0731**; the measured native v1 gain against that same historical reference is **+0.0697**. These are descriptive comparisons across backend/training protocols, not a controlled architecture ablation. The complete-camera-failure gain comes from the official LiDAR-only fallback, not solely from the learned gate.

All 81 normal and 81 partial-failure frames used `gated_fusion`; all 81 complete-camera-failure frames used `lidar_fallback`. Native v1 holds both experts resident. NVML observed a **3.317 GiB whole-device increase above baseline** during startup/smoke/three-condition inference. WSL2 returned no per-process memory samples, so this is not a process peak-VRAM measurement.

Native model timings include CPU reliability cue computation, FP32-to-FP16 point conversion, calibration update, H2D transfer, CUDA/TensorRT execution and C++ decoding. They exclude offline image preprocessing and ROS2 transport. One smoke frame per condition preceded the 81-frame timing runs. The earlier epoch-4 deployment's timing must not be attributed to v1.

See [results/](results/) for exact metrics, class AP, weight/engine hashes, route counts, numerical checks and memory scope.

## What runs in C++

`V1Core` composes the public modules of the pinned NVIDIA native runtime; upstream source/library code is unchanged. It uses six TensorRT engines and two native sparse-convolution models:

- Fusion: camera/depth network, camera BEV downsample, gated fusion + SECOND/SECONDFPN, and TransFusion head.
- Fallback: LiDAR SECOND/SECONDFPN and its own TransFusion head, plus its independently weighted sparse encoder.
- CUDA modules: voxelization, sparse convolution, sparse depth, camera geometry, BEV pooling and box decoding.
- C++: camera-health routing and the gate's point-density, camera-coverage and distance cues.

The gate's camera/LiDAR energy cues, learned 5-to-16-to-2 network and feature scaling are inside the fusion TensorRT engine. Reliability cue inputs remain FP32. Dense engine boundaries otherwise use FP16; this does not mean every dense layer computes in FP16.

The ROS2 packet has magic `BFV1`, version 2 and a six-bit available-camera mask. It carries preprocessed camera images, **FP32** point coordinates for faithful density computation and calibration matrices. Points are converted to FP16 inside the native node for the existing CUDA runtime. Raw camera/PointCloud2 topic synchronization remains outside the current interface.

## Verification

- Gate and fallback checkpoint SHA-256 values match the frozen record.
- The original evaluation transform is retained as `reference_degradation.py`.
- Input preparation uses the same two workers and seed 0 as the frozen evaluation. This matters because multi-sweep selection also uses worker RNG state.
- The aggregate point counts and camera-drop frequencies match the historical frozen run: 19,756,228 points before dropout and 4,940,303 after dropout in each degraded condition.
- Independently rebuilt loaders reproduced first/last-frame point arrays, normalized FP16 images and camera masks exactly.
- C++ camera-coverage maps matched PyTorch exactly; maximum density/distance cue error was below 2.4e-7.
- The gated TensorRT trunk versus the frozen PyTorch gate on quantized test features had correlation 0.9999986 and relative L2 error 0.00162.
- Both route smoke checks and all three 81-frame native ROS2 evaluations completed.

The reference files are the project's frozen gate/evaluation implementation with imports and registry aliases adapted for the separate official MMDetection3D checkout. They are not used by the native inference node.

## Reproduce

Use the existing dependencies described in the [parent guide](../README.md). Copy [settings.example.json](settings.example.json) outside the repository and fill in absolute paths to the two frozen checkpoints, official MMDetection3D v1.4.0 checkout, dataset, pinned NVIDIA checkout and existing toolchains. Set `monitor_python` to an existing Python with pynvml if memory monitoring is required.

```bash
python3 /path/to/portfolio/projects/BEVFusion/deploy/native_cpp/v1/pipeline_v1.py \
  --settings /path/to/settings-v1.local.json \
  --output-dir /path/to/new-v1-run
```

The entry point composes the export, input, native build, parity, three-condition validation and viewer-generation commands. It supports `--resume` with unchanged settings; completed outputs are preserved and retries use new attempt directories. The recorded experiment ran these same component commands with existing compiled CUDA dependencies and a separate official source tree.

Start the generated viewer locally:

```bash
python3 -m http.server 8766 --bind 127.0.0.1 \
  --directory /path/to/new-v1-run/visualization
```

Open `http://localhost:8766/`. The viewer shows **saved actual native predictions**, not a live inference service. It supports three conditions, six camera images and projected boxes, 3D point-cloud orbit/zoom, frame playback and a display-only confidence threshold. Display point sampling uses the same baseline sampling rate, so sparse inputs remain visibly sparser. Display filtering never changes evaluation results.

Checkpoints, engines, binary inputs, full prediction files, raw dataset images and complete 81-frame viewer datasets remain outside Git. Source, small aggregate results and the explicitly published compact demo assets are included here.
