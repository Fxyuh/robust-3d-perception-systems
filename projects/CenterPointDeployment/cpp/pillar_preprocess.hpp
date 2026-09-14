#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cpd {

inline constexpr int kPointChannels = 5;
inline constexpr int kDecoratedChannels = 11;
inline constexpr int kFeatureChannels = 64;
inline constexpr int kMaxPoints = 20;
inline constexpr int kMaxVoxels = 40000;
inline constexpr int kGridX = 512;
inline constexpr int kGridY = 512;
inline constexpr int kGridZ = 1;
inline constexpr float kVoxelX = 0.2F;
inline constexpr float kVoxelY = 0.2F;
inline constexpr float kVoxelZ = 8.0F;
inline constexpr float kMinX = -51.2F;
inline constexpr float kMinY = -51.2F;
inline constexpr float kMinZ = -5.0F;
inline constexpr float kMaxX = 51.2F;
inline constexpr float kMaxY = 51.2F;
inline constexpr float kMaxZ = 3.0F;

struct Voxels {
  std::vector<float> values;
  std::vector<std::int32_t> coordinates;
  std::vector<std::int32_t> counts;
};

Voxels voxelize(const std::vector<float>& points, std::size_t point_count);

void launchPillarFeatureScatter(
    const float* voxels, const std::int32_t* coordinates,
    const std::int32_t* counts, int voxel_count, const float* weight,
    const float* bias, float* pillar_features, float* pseudo_image,
    cudaStream_t stream = nullptr);

}  // namespace cpd
