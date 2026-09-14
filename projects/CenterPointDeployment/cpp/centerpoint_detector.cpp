#include "centerpoint_detector.hpp"

#include "NvInferRuntime.h"
#include "array_io.hpp"
#include "pillar_preprocess.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cpd {
namespace {

#define CUDA_CHECK(expression)                                                \
  do {                                                                        \
    const cudaError_t error = (expression);                                   \
    if (error != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("CUDA failure: ") +               \
                               cudaGetErrorString(error));                    \
    }                                                                         \
  } while (false)

constexpr int kHeatmapHeight = 128;
constexpr int kHeatmapWidth = 128;
constexpr int kTopK = 500;
constexpr int kPostMaxSize = 83;
constexpr std::array<int, 6> kTaskClasses = {1, 2, 2, 1, 2, 2};
constexpr std::array<float, 6> kMinRadius = {4.0F, 12.0F, 10.0F,
                                              1.0F, 0.85F, 0.175F};

std::string joinPath(const std::string& directory, const std::string& name) {
  return directory + (directory.empty() || directory.back() == '/' ? "" : "/") +
         name;
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << message << '\n';
    }
  }
};

template <typename T> class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      cudaFree(data_);
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }
  void allocate(std::size_t count) {
    if (data_ != nullptr) {
      throw std::runtime_error("device buffer already allocated");
    }
    count_ = count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
  }
  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

std::vector<char> readBytes(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open engine " + path);
  }
  const auto size = stream.tellg();
  if (size <= 0) {
    throw std::runtime_error("empty engine " + path);
  }
  std::vector<char> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(bytes.data(), size);
  if (!stream) {
    throw std::runtime_error("failed to read engine " + path);
  }
  return bytes;
}

std::size_t elementCount(const nvinfer1::Dims& dimensions) {
  std::size_t count = 1;
  for (int index = 0; index < dimensions.nbDims; ++index) {
    if (dimensions.d[index] <= 0) {
      throw std::runtime_error("dynamic or invalid TensorRT tensor shape");
    }
    count *= static_cast<std::size_t>(dimensions.d[index]);
  }
  return count;
}

struct OutputBuffer {
  std::string name;
  std::vector<float> host;
  DeviceBuffer<float> device;
};

class TensorRtRunner {
 public:
  explicit TensorRtRunner(const std::string& engine_path) {
    const std::vector<char> engine_bytes = readBytes(engine_path);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      throw std::runtime_error("failed to create TensorRT runtime");
    }
    engine_.reset(runtime_->deserializeCudaEngine(engine_bytes.data(),
                                                  engine_bytes.size()));
    if (!engine_) {
      throw std::runtime_error("failed to deserialize TensorRT engine");
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw std::runtime_error("failed to create TensorRT execution context");
    }

    for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
      const std::string name = engine_->getIOTensorName(index);
      if (engine_->getTensorDataType(name.c_str()) !=
          nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("only float TensorRT I/O is supported: " + name);
      }
      if (engine_->getTensorIOMode(name.c_str()) ==
          nvinfer1::TensorIOMode::kINPUT) {
        if (!input_name_.empty() || name != "pseudo_image") {
          throw std::runtime_error("unexpected TensorRT input " + name);
        }
        input_name_ = name;
        const auto dimensions = engine_->getTensorShape(name.c_str());
        if (elementCount(dimensions) !=
            static_cast<std::size_t>(kFeatureChannels) * kGridY * kGridX) {
          throw std::runtime_error("unexpected TensorRT pseudo image shape");
        }
      } else {
        OutputBuffer output;
        output.name = name;
        const std::size_t count =
            elementCount(engine_->getTensorShape(name.c_str()));
        output.host.resize(count);
        output.device.allocate(count);
        output_index_.emplace(output.name, outputs_.size());
        outputs_.push_back(std::move(output));
      }
    }
    if (input_name_.empty() || outputs_.size() != 36) {
      throw std::runtime_error("unexpected TensorRT I/O tensor count");
    }
    for (auto& output : outputs_) {
      if (!context_->setTensorAddress(output.name.c_str(), output.device.get())) {
        throw std::runtime_error("failed to bind TensorRT output " + output.name);
      }
    }
  }

  void bindInput(float* pseudo_image) {
    if (!context_->setTensorAddress(input_name_.c_str(), pseudo_image)) {
      throw std::runtime_error("failed to bind TensorRT pseudo image");
    }
  }

  void enqueue(cudaStream_t stream) {
    if (!context_->enqueueV3(stream)) {
      throw std::runtime_error("TensorRT enqueueV3 failed");
    }
  }

  void copyOutputsToHost(cudaStream_t stream) {
    for (auto& output : outputs_) {
      CUDA_CHECK(cudaMemcpyAsync(output.host.data(), output.device.get(),
                                 output.device.bytes(), cudaMemcpyDeviceToHost,
                                 stream));
    }
  }

  const std::vector<float>& output(const std::string& name) const {
    const auto found = output_index_.find(name);
    if (found == output_index_.end()) {
      throw std::runtime_error("missing TensorRT output " + name);
    }
    return outputs_[found->second].host;
  }

  std::size_t outputBytes() const {
    return std::accumulate(
        outputs_.begin(), outputs_.end(), std::size_t{0},
        [](std::size_t total, const OutputBuffer& output) {
          return total + output.device.bytes();
        });
  }

 private:
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::string input_name_;
  std::vector<OutputBuffer> outputs_;
  std::unordered_map<std::string, std::size_t> output_index_;
};

struct Peak {
  float score;
  int spatial_index;
  int class_index;
};

float sigmoid(float value) {
  return 1.0F / (1.0F + std::exp(-value));
}

std::vector<Peak> topKHeatmap(const std::vector<float>& heatmap,
                              int class_count) {
  constexpr int spatial_count = kHeatmapHeight * kHeatmapWidth;
  std::vector<Peak> merged;
  merged.reserve(static_cast<std::size_t>(class_count) * kTopK);
  const auto higher_score = [](const Peak& left, const Peak& right) {
    return left.score > right.score;
  };
  for (int class_index = 0; class_index < class_count; ++class_index) {
    std::vector<Peak> per_class;
    per_class.reserve(spatial_count);
    for (int spatial_index = 0; spatial_index < spatial_count; ++spatial_index) {
      per_class.push_back(
          {sigmoid(heatmap[class_index * spatial_count + spatial_index]),
           spatial_index, class_index});
    }
    std::partial_sort(per_class.begin(), per_class.begin() + kTopK,
                      per_class.end(), higher_score);
    merged.insert(merged.end(), per_class.begin(), per_class.begin() + kTopK);
  }
  std::partial_sort(merged.begin(), merged.begin() + kTopK, merged.end(),
                    higher_score);
  merged.resize(kTopK);
  return merged;
}

std::vector<Detection> circleNms(std::vector<Detection> detections,
                                 float minimum_radius) {
  std::vector<int> order(detections.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int left, int right) {
    return detections[left].score > detections[right].score;
  });
  std::vector<bool> suppressed(detections.size(), false);
  std::vector<Detection> kept;
  kept.reserve(std::min<std::size_t>(detections.size(), kPostMaxSize));
  for (std::size_t order_index = 0; order_index < order.size(); ++order_index) {
    const int current = order[order_index];
    if (suppressed[current]) {
      continue;
    }
    kept.push_back(detections[current]);
    if (kept.size() == kPostMaxSize) {
      break;
    }
    for (std::size_t next_index = order_index + 1; next_index < order.size();
         ++next_index) {
      const int next = order[next_index];
      if (suppressed[next]) {
        continue;
      }
      const float dx = detections[current].box[0] - detections[next].box[0];
      const float dy = detections[current].box[1] - detections[next].box[1];
      if (dx * dx + dy * dy <= minimum_radius) {
        suppressed[next] = true;
      }
    }
  }
  return kept;
}

float gather(const std::vector<float>& tensor, int channel, int index) {
  constexpr int spatial_count = kHeatmapHeight * kHeatmapWidth;
  return tensor[channel * spatial_count + index];
}

std::vector<Detection> decode(const TensorRtRunner& runner,
                              float score_threshold) {
  std::vector<Detection> merged;
  int label_offset = 0;
  for (int task = 0; task < static_cast<int>(kTaskClasses.size()); ++task) {
    const std::string prefix = "task" + std::to_string(task) + "_";
    const auto& heatmap = runner.output(prefix + "heatmap");
    const auto& regression = runner.output(prefix + "reg");
    const auto& height = runner.output(prefix + "height");
    const auto& dimensions = runner.output(prefix + "dim");
    const auto& rotation = runner.output(prefix + "rot");
    const auto& velocity = runner.output(prefix + "vel");
    const std::vector<Peak> peaks = topKHeatmap(heatmap, kTaskClasses[task]);
    std::vector<Detection> task_detections;
    task_detections.reserve(kTopK);
    for (const Peak& peak : peaks) {
      if (!(peak.score > score_threshold)) {
        continue;
      }
      const int grid_y = peak.spatial_index / kHeatmapWidth;
      const int grid_x = peak.spatial_index % kHeatmapWidth;
      Detection detection;
      detection.box[0] =
          (static_cast<float>(grid_x) + gather(regression, 0, peak.spatial_index)) *
              4.0F * kVoxelX +
          kMinX;
      detection.box[1] =
          (static_cast<float>(grid_y) + gather(regression, 1, peak.spatial_index)) *
              4.0F * kVoxelY +
          kMinY;
      detection.box[2] = gather(height, 0, peak.spatial_index);
      detection.box[3] = std::exp(gather(dimensions, 0, peak.spatial_index));
      detection.box[4] = std::exp(gather(dimensions, 1, peak.spatial_index));
      detection.box[5] = std::exp(gather(dimensions, 2, peak.spatial_index));
      detection.box[6] =
          std::atan2(gather(rotation, 0, peak.spatial_index),
                     gather(rotation, 1, peak.spatial_index));
      detection.box[7] = gather(velocity, 0, peak.spatial_index);
      detection.box[8] = gather(velocity, 1, peak.spatial_index);
      detection.score = peak.score;
      detection.label = label_offset + peak.class_index;
      if (detection.box[0] >= -61.2F && detection.box[0] <= 61.2F &&
          detection.box[1] >= -61.2F && detection.box[1] <= 61.2F &&
          detection.box[2] >= -10.0F && detection.box[2] <= 10.0F) {
        task_detections.push_back(detection);
      }
    }
    std::vector<Detection> kept =
        circleNms(std::move(task_detections), kMinRadius[task]);
    for (Detection& detection : kept) {
      detection.box[2] -= detection.box[5] * 0.5F;
      merged.push_back(detection);
    }
    label_offset += kTaskClasses[task];
  }
  return merged;
}

}  // namespace

class CenterPointDetector::Impl {
 public:
  explicit Impl(DetectorConfig config)
      : config_(std::move(config)),
        weight_(readArray<float>(joinPath(config_.parameter_directory,
                                          "pfn_weight_f32.cpd"))),
        bias_(readArray<float>(joinPath(config_.parameter_directory,
                                        "pfn_bias_f32.cpd"))),
        device_voxels_(static_cast<std::size_t>(kMaxVoxels) * kMaxPoints *
                       kPointChannels),
        device_coordinates_(static_cast<std::size_t>(kMaxVoxels) * 4),
        device_counts_(kMaxVoxels),
        device_weight_(weight_.values.size()),
        device_bias_(bias_.values.size()),
        device_features_(static_cast<std::size_t>(kMaxVoxels) *
                         kFeatureChannels),
        device_pseudo_(static_cast<std::size_t>(kFeatureChannels) * kGridY *
                       kGridX),
        runner_(config_.engine_path) {
    if (weight_.shape !=
            std::vector<std::uint64_t>{kFeatureChannels, kDecoratedChannels} ||
        bias_.shape != std::vector<std::uint64_t>{kFeatureChannels}) {
      throw std::runtime_error("unexpected PFN weight shape");
    }
    if (!(config_.score_threshold >= 0.0F &&
          config_.score_threshold <= 1.0F)) {
      throw std::runtime_error("score_threshold must be within [0, 1]");
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpy(device_weight_.get(), weight_.values.data(),
                          device_weight_.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_bias_.get(), bias_.values.data(),
                          device_bias_.bytes(), cudaMemcpyHostToDevice));
    runner_.bindInput(device_pseudo_.get());
  }

  ~Impl() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  std::vector<Detection> infer(const std::vector<float>& points,
                               InferenceTiming* timing) {
    if (points.empty() || points.size() % kPointChannels != 0) {
      throw std::runtime_error("points must be a non-empty, packed N x 5 tensor");
    }
    const auto end_to_end_start = std::chrono::steady_clock::now();
    const auto voxel_start = end_to_end_start;
    const std::size_t point_count = points.size() / kPointChannels;
    Voxels voxels = voxelize(points, point_count);
    const auto voxel_stop = std::chrono::steady_clock::now();

    const auto gpu_start = voxel_stop;
    const int voxel_count = static_cast<int>(voxels.counts.size());
    if (!voxels.values.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(
          device_voxels_.get(), voxels.values.data(),
          voxels.values.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(
          device_coordinates_.get(), voxels.coordinates.data(),
          voxels.coordinates.size() * sizeof(std::int32_t),
          cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(
          device_counts_.get(), voxels.counts.data(),
          voxels.counts.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
          stream_));
    }
    launchPillarFeatureScatter(
        device_voxels_.get(), device_coordinates_.get(), device_counts_.get(),
        voxel_count, device_weight_.get(), device_bias_.get(),
        device_features_.get(), device_pseudo_.get(), stream_);
    runner_.enqueue(stream_);
    runner_.copyOutputsToHost(stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const auto gpu_stop = std::chrono::steady_clock::now();

    const auto decode_start = gpu_stop;
    std::vector<Detection> detections =
        decode(runner_, config_.score_threshold);
    const auto decode_stop = std::chrono::steady_clock::now();

    if (timing != nullptr) {
      timing->point_count = point_count;
      timing->voxel_count = voxels.counts.size();
      timing->cpu_voxelization_ms =
          std::chrono::duration<double, std::milli>(voxel_stop - voxel_start)
              .count();
      timing->gpu_pipeline_ms =
          std::chrono::duration<double, std::milli>(gpu_stop - gpu_start)
              .count();
      timing->decode_ms =
          std::chrono::duration<double, std::milli>(decode_stop - decode_start)
              .count();
      timing->end_to_end_ms =
          std::chrono::duration<double, std::milli>(decode_stop -
                                                    end_to_end_start)
              .count();
    }
    return detections;
  }

  std::size_t explicitCudaBufferBytes() const {
    return device_voxels_.bytes() + device_coordinates_.bytes() +
           device_counts_.bytes() + device_weight_.bytes() +
           device_bias_.bytes() + device_features_.bytes() +
           device_pseudo_.bytes() + runner_.outputBytes();
  }

 private:
  DetectorConfig config_;
  Array<float> weight_;
  Array<float> bias_;
  DeviceBuffer<float> device_voxels_;
  DeviceBuffer<std::int32_t> device_coordinates_;
  DeviceBuffer<std::int32_t> device_counts_;
  DeviceBuffer<float> device_weight_;
  DeviceBuffer<float> device_bias_;
  DeviceBuffer<float> device_features_;
  DeviceBuffer<float> device_pseudo_;
  TensorRtRunner runner_;
  cudaStream_t stream_ = nullptr;
};

CenterPointDetector::CenterPointDetector(DetectorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

CenterPointDetector::~CenterPointDetector() = default;
CenterPointDetector::CenterPointDetector(CenterPointDetector&&) noexcept =
    default;
CenterPointDetector& CenterPointDetector::operator=(
    CenterPointDetector&&) noexcept = default;

std::vector<Detection> CenterPointDetector::infer(
    const std::vector<float>& points, InferenceTiming* timing) {
  return impl_->infer(points, timing);
}

std::size_t CenterPointDetector::explicitCudaBufferBytes() const {
  return impl_->explicitCudaBufferBytes();
}

}  // namespace cpd
