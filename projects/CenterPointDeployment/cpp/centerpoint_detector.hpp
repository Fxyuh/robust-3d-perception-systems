#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cpd {

struct Detection {
  // MMDetection3D LiDAR box convention:
  // [bottom_center_x, bottom_center_y, bottom_center_z,
  //  length_x, width_y, height_z, yaw, velocity_x, velocity_y].
  std::array<float, 9> box{};
  float score = 0.0F;
  int label = 0;
};

struct InferenceTiming {
  std::size_t point_count = 0;
  std::size_t voxel_count = 0;
  double cpu_voxelization_ms = 0.0;
  double gpu_pipeline_ms = 0.0;
  double decode_ms = 0.0;
  double end_to_end_ms = 0.0;
};

struct DetectorConfig {
  std::string parameter_directory;
  std::string engine_path;
  float score_threshold = 0.1F;
};

class CenterPointDetector {
 public:
  explicit CenterPointDetector(DetectorConfig config);
  ~CenterPointDetector();

  CenterPointDetector(const CenterPointDetector&) = delete;
  CenterPointDetector& operator=(const CenterPointDetector&) = delete;
  CenterPointDetector(CenterPointDetector&&) noexcept;
  CenterPointDetector& operator=(CenterPointDetector&&) noexcept;

  // points is a tightly packed N x 5 tensor containing
  // x, y, z, intensity and time_lag in the current LiDAR frame.
  std::vector<Detection> infer(const std::vector<float>& points,
                               InferenceTiming* timing = nullptr);

  std::size_t explicitCudaBufferBytes() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cpd
