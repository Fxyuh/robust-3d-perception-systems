#include "array_io.hpp"
#include "pillar_preprocess.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cpd {

#define CUDA_CHECK(expression)                                                \
  do {                                                                        \
    const cudaError_t error = (expression);                                   \
    if (error != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("CUDA failure: ") +               \
                               cudaGetErrorString(error));                    \
    }                                                                         \
  } while (false)

struct Options {
  std::string input_dir = "work_dirs/centerpoint_deployment/preprocess";
  std::string output =
      "work_dirs/centerpoint_deployment/preprocess/pseudo_image_cpp_f32.cpd";
  int warmup = 20;
  int iterations = 100;
};

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--input-dir" && index + 1 < argc) {
      options.input_dir = argv[++index];
    } else if (argument == "--output" && index + 1 < argc) {
      options.output = argv[++index];
    } else if (argument == "--warmup" && index + 1 < argc) {
      options.warmup = std::stoi(argv[++index]);
    } else if (argument == "--iterations" && index + 1 < argc) {
      options.iterations = std::stoi(argv[++index]);
    } else {
      throw std::runtime_error("unknown or incomplete option: " + argument);
    }
  }
  if (options.warmup < 0 || options.iterations <= 0) {
    throw std::runtime_error("warmup must be non-negative and iterations positive");
  }
  return options;
}

std::string joinPath(const std::string& directory, const std::string& name) {
  return directory + (directory.empty() || directory.back() == '/' ? "" : "/") +
         name;
}

Voxels voxelize(const std::vector<float>& points, std::size_t point_count) {
  Voxels result;
  result.values.assign(static_cast<std::size_t>(kMaxVoxels) * kMaxPoints *
                           kPointChannels,
                       0.0F);
  result.coordinates.reserve(static_cast<std::size_t>(kMaxVoxels) * 4);
  result.counts.reserve(kMaxVoxels);
  std::unordered_map<int, int> voxel_indices;
  voxel_indices.reserve(kMaxVoxels);

  for (std::size_t point_index = 0; point_index < point_count; ++point_index) {
    const float* point = points.data() + point_index * kPointChannels;
    if (!(point[0] >= kMinX && point[0] < kMaxX && point[1] >= kMinY &&
          point[1] < kMaxY && point[2] >= kMinZ && point[2] < kMaxZ)) {
      continue;
    }
    const int x = static_cast<int>(std::floor((point[0] - kMinX) / kVoxelX));
    const int y = static_cast<int>(std::floor((point[1] - kMinY) / kVoxelY));
    const int z = static_cast<int>(std::floor((point[2] - kMinZ) / kVoxelZ));
    if (x < 0 || x >= kGridX || y < 0 || y >= kGridY || z < 0 || z >= kGridZ) {
      continue;
    }
    const int key = (z * kGridY + y) * kGridX + x;
    auto found = voxel_indices.find(key);
    int voxel_index = 0;
    if (found == voxel_indices.end()) {
      if (result.counts.size() >= kMaxVoxels) {
        continue;
      }
      voxel_index = static_cast<int>(result.counts.size());
      voxel_indices.emplace(key, voxel_index);
      result.coordinates.insert(result.coordinates.end(), {0, z, y, x});
      result.counts.push_back(0);
    } else {
      voxel_index = found->second;
    }
    int& count = result.counts[voxel_index];
    if (count >= kMaxPoints) {
      continue;
    }
    float* destination =
        result.values.data() +
        (static_cast<std::size_t>(voxel_index) * kMaxPoints + count) *
            kPointChannels;
    std::copy(point, point + kPointChannels, destination);
    ++count;
  }
  result.values.resize(result.counts.size() * kMaxPoints * kPointChannels);
  return result;
}

__global__ void pillarFeatureScatter(
    const float* voxels, const std::int32_t* coordinates,
    const std::int32_t* counts, int voxel_count, const float* weight,
    const float* bias, float* pillar_features, float* pseudo_image) {
  const int voxel_index = blockIdx.x;
  const int output_channel = threadIdx.x;
  if (voxel_index >= voxel_count || output_channel >= kFeatureChannels) {
    return;
  }

  const int point_count = counts[voxel_index];
  float mean_x = 0.0F;
  float mean_y = 0.0F;
  float mean_z = 0.0F;
  for (int point_index = 0; point_index < point_count; ++point_index) {
    const float* point =
        voxels + (voxel_index * kMaxPoints + point_index) * kPointChannels;
    mean_x += point[0];
    mean_y += point[1];
    mean_z += point[2];
  }
  mean_x /= static_cast<float>(point_count);
  mean_y /= static_cast<float>(point_count);
  mean_z /= static_cast<float>(point_count);

  const int z = coordinates[voxel_index * 4 + 1];
  const int y = coordinates[voxel_index * 4 + 2];
  const int x = coordinates[voxel_index * 4 + 3];
  const float center_x = static_cast<float>(x) * kVoxelX + kMinX + kVoxelX / 2;
  const float center_y = static_cast<float>(y) * kVoxelY + kMinY + kVoxelY / 2;
  const float center_z = static_cast<float>(z) * kVoxelZ + kMinZ + kVoxelZ / 2;
  const float* channel_weight = weight + output_channel * kDecoratedChannels;
  float maximum = -FLT_MAX;

  // MMDetection3D masks padded decorated inputs to zero before the folded
  // Linear+BatchNorm. Consequently padding still evaluates to ReLU(bias),
  // so all 20 slots must participate in the max reduction.
  for (int point_index = 0; point_index < kMaxPoints; ++point_index) {
    float decorated[kDecoratedChannels] = {};
    if (point_index < point_count) {
      const float* point =
          voxels + (voxel_index * kMaxPoints + point_index) * kPointChannels;
#pragma unroll
      for (int channel = 0; channel < kPointChannels; ++channel) {
        decorated[channel] = point[channel];
      }
      decorated[5] = point[0] - mean_x;
      decorated[6] = point[1] - mean_y;
      decorated[7] = point[2] - mean_z;
      decorated[8] = point[0] - center_x;
      decorated[9] = point[1] - center_y;
      decorated[10] = point[2] - center_z;
    }
    float value = bias[output_channel];
#pragma unroll
    for (int channel = 0; channel < kDecoratedChannels; ++channel) {
      value = fmaf(channel_weight[channel], decorated[channel], value);
    }
    maximum = fmaxf(maximum, fmaxf(value, 0.0F));
  }

  pillar_features[voxel_index * kFeatureChannels + output_channel] = maximum;
  pseudo_image[(output_channel * kGridY + y) * kGridX + x] = maximum;
}

void launchPillarFeatureScatter(
    const float* voxels, const std::int32_t* coordinates,
    const std::int32_t* counts, int voxel_count, const float* weight,
    const float* bias, float* pillar_features, float* pseudo_image,
    cudaStream_t stream) {
  const std::size_t pseudo_elements =
      static_cast<std::size_t>(kFeatureChannels) * kGridY * kGridX;
  CUDA_CHECK(cudaMemsetAsync(pseudo_image, 0, pseudo_elements * sizeof(float),
                             stream));
  if (voxel_count > 0) {
    pillarFeatureScatter<<<voxel_count, kFeatureChannels, 0, stream>>>(
        voxels, coordinates, counts, voxel_count, weight, bias, pillar_features,
        pseudo_image);
    CUDA_CHECK(cudaGetLastError());
  }
}

std::pair<double, double> compare(const std::vector<float>& actual,
                                  const std::vector<float>& reference) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("comparison size mismatch");
  }
  double maximum = 0.0;
  double mean = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double difference =
        std::abs(static_cast<double>(actual[index]) - reference[index]);
    maximum = std::max(maximum, difference);
    mean += difference;
  }
  return {maximum, mean / static_cast<double>(actual.size())};
}

template <typename T>
std::size_t mismatchCount(const std::vector<T>& actual,
                          const std::vector<T>& reference) {
  if (actual.size() != reference.size()) {
    return std::max(actual.size(), reference.size());
  }
  std::size_t mismatches = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    mismatches += actual[index] != reference[index];
  }
  return mismatches;
}

template <typename T> class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t count() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace cpd

#ifndef CPD_NO_MAIN
int main(int argc, char** argv) {
  try {
    using namespace cpd;
    const Options options = parseOptions(argc, argv);
    const auto points =
        cpd::readArray<float>(joinPath(options.input_dir, "points_f32.cpd"));
    const auto weight = cpd::readArray<float>(
        joinPath(options.input_dir, "pfn_weight_f32.cpd"));
    const auto bias =
        cpd::readArray<float>(joinPath(options.input_dir, "pfn_bias_f32.cpd"));
    const auto reference_voxels =
        cpd::readArray<float>(joinPath(options.input_dir, "voxels_f32.cpd"));
    const auto reference_coordinates = cpd::readArray<std::int32_t>(
        joinPath(options.input_dir, "coors_i32.cpd"));
    const auto reference_counts = cpd::readArray<std::int32_t>(
        joinPath(options.input_dir, "num_points_i32.cpd"));
    const auto reference_features = cpd::readArray<float>(
        joinPath(options.input_dir, "pfn_features_f32.cpd"));
    const auto reference_pseudo = cpd::readArray<float>(
        joinPath(options.input_dir, "pseudo_image_f32.cpd"));

    if (points.shape.size() != 2 || points.shape[1] != kPointChannels ||
        weight.shape != std::vector<std::uint64_t>{kFeatureChannels,
                                                   kDecoratedChannels} ||
        bias.shape != std::vector<std::uint64_t>{kFeatureChannels}) {
      throw std::runtime_error("unexpected input or PFN weight shape");
    }

    const int cpu_iterations = std::min(options.iterations, 20);
    Voxels voxels;
    const auto cpu_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < cpu_iterations; ++iteration) {
      voxels = voxelize(points.values, points.shape[0]);
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double voxelization_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count() /
        cpu_iterations;
    const int voxel_count = static_cast<int>(voxels.counts.size());

    const std::size_t pseudo_elements =
        static_cast<std::size_t>(kFeatureChannels) * kGridY * kGridX;
    DeviceBuffer<float> device_voxels(voxels.values.size());
    DeviceBuffer<std::int32_t> device_coordinates(voxels.coordinates.size());
    DeviceBuffer<std::int32_t> device_counts(voxels.counts.size());
    DeviceBuffer<float> device_weight(weight.values.size());
    DeviceBuffer<float> device_bias(bias.values.size());
    DeviceBuffer<float> device_features(static_cast<std::size_t>(voxel_count) *
                                        kFeatureChannels);
    DeviceBuffer<float> device_pseudo(pseudo_elements);
    CUDA_CHECK(cudaMemcpy(device_voxels.get(), voxels.values.data(),
                          voxels.values.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_coordinates.get(), voxels.coordinates.data(),
                          voxels.coordinates.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_counts.get(), voxels.counts.data(),
                          voxels.counts.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_weight.get(), weight.values.data(),
                          weight.values.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_bias.get(), bias.values.data(),
                          bias.values.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto launch = [&]() {
      launchPillarFeatureScatter(
          device_voxels.get(), device_coordinates.get(), device_counts.get(),
          voxel_count, device_weight.get(), device_bias.get(),
          device_features.get(), device_pseudo.get());
    };
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<float> features(device_features.count());
    std::vector<float> pseudo(pseudo_elements);
    CUDA_CHECK(cudaMemcpy(features.data(), device_features.get(),
                          features.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pseudo.data(), device_pseudo.get(),
                          pseudo.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const auto voxel_error = compare(voxels.values, reference_voxels.values);
    const auto feature_error = compare(features, reference_features.values);
    const auto pseudo_error = compare(pseudo, reference_pseudo.values);
    const std::size_t coordinate_mismatches = mismatchCount(
        voxels.coordinates, reference_coordinates.values);
    const std::size_t count_mismatches =
        mismatchCount(voxels.counts, reference_counts.values);

    cpd::writeArray<float>(options.output, {1, kFeatureChannels, kGridY, kGridX},
                           pseudo);
    std::cout << std::fixed << std::setprecision(8)
              << "points=" << points.shape[0] << '\n'
              << "voxels_cpp=" << voxel_count
              << " voxels_reference=" << reference_counts.values.size() << '\n'
              << "coordinate_mismatches=" << coordinate_mismatches << '\n'
              << "count_mismatches=" << count_mismatches << '\n'
              << "voxel_max_abs=" << voxel_error.first
              << " voxel_mean_abs=" << voxel_error.second << '\n'
              << "pfn_max_abs=" << feature_error.first
              << " pfn_mean_abs=" << feature_error.second << '\n'
              << "pseudo_max_abs=" << pseudo_error.first
              << " pseudo_mean_abs=" << pseudo_error.second << '\n'
              << "cpu_voxelization_ms=" << voxelization_ms << '\n'
              << "cuda_pfn_scatter_ms="
              << gpu_elapsed_ms / static_cast<float>(options.iterations) << '\n'
              << "output=" << options.output << '\n';

    const bool passed =
        voxel_count == static_cast<int>(reference_counts.values.size()) &&
        coordinate_mismatches == 0 && count_mismatches == 0 &&
        voxel_error.first <= 1.0e-6 && feature_error.first <= 2.0e-4 &&
        pseudo_error.first <= 2.0e-4;
    if (!passed) {
      std::cerr << "C++/CUDA preprocessing parity failed\n";
      return 2;
    }
    std::cout << "C++/CUDA preprocessing parity passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
#endif
