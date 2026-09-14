#include "array_io.hpp"
#include "centerpoint_detector.hpp"
#include "pillar_preprocess.hpp"

#include <cuda_runtime_api.h>
#include <nvml.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::array<const char*, 10> kClassNames = {
    "car",          "truck",   "construction_vehicle", "bus",
    "trailer",      "barrier", "motorcycle",            "bicycle",
    "pedestrian",   "traffic_cone"};

constexpr std::array<const char*, 10> kDefaultAttributes = {
    "vehicle.parked",     "vehicle.parked",      "vehicle.parked",
    "vehicle.moving",     "vehicle.parked",      "",
    "cycle.without_rider", "cycle.without_rider", "pedestrian.moving",
    ""};

struct Options {
  std::string parameter_directory =
      "work_dirs/centerpoint_deployment/preprocess";
  std::string engine = "work_dirs/centerpoint_deployment/tensorrt/"
                       "centerpoint_dense_mixed_fp16_deep.engine";
  std::string scene_root =
      "work_dirs/centerpoint_deployment/ros2-scenes";
  std::string result = "work_dirs/centerpoint_deployment/detection_eval/"
                       "results_nusc.json";
  std::string timings = "work_dirs/centerpoint_deployment/detection_eval/"
                        "frame_timings.csv";
  std::string report = "work_dirs/centerpoint_deployment/detection_eval/"
                       "benchmark.json";
  float score_threshold = 0.1F;
  int gpu_index = 0;
  int nvml_interval_ms = 1;
  std::size_t max_frames = 0;
};

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--parameter-directory" && index + 1 < argc) {
      options.parameter_directory = argv[++index];
    } else if (argument == "--engine" && index + 1 < argc) {
      options.engine = argv[++index];
    } else if (argument == "--scene-root" && index + 1 < argc) {
      options.scene_root = argv[++index];
    } else if (argument == "--result" && index + 1 < argc) {
      options.result = argv[++index];
    } else if (argument == "--timings" && index + 1 < argc) {
      options.timings = argv[++index];
    } else if (argument == "--report" && index + 1 < argc) {
      options.report = argv[++index];
    } else if (argument == "--score-threshold" && index + 1 < argc) {
      options.score_threshold = std::stof(argv[++index]);
    } else if (argument == "--gpu-index" && index + 1 < argc) {
      options.gpu_index = std::stoi(argv[++index]);
    } else if (argument == "--nvml-interval-ms" && index + 1 < argc) {
      options.nvml_interval_ms = std::stoi(argv[++index]);
    } else if (argument == "--max-frames" && index + 1 < argc) {
      options.max_frames = static_cast<std::size_t>(std::stoull(argv[++index]));
    } else {
      throw std::runtime_error("unknown or incomplete option: " + argument);
    }
  }
  if (!(options.score_threshold >= 0.0F &&
        options.score_threshold <= 1.0F)) {
    throw std::runtime_error("score threshold must be within [0, 1]");
  }
  if (options.gpu_index < 0 || options.nvml_interval_ms <= 0) {
    throw std::runtime_error(
        "gpu index must be non-negative and NVML interval must be positive");
  }
  return options;
}

std::string joinPath(const std::string& directory, const std::string& name) {
  return directory + (directory.empty() || directory.back() == '/' ? "" : "/") +
         name;
}

std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

struct Quaternion {
  double w = 1.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

Quaternion normalize(Quaternion quaternion) {
  const double norm = std::sqrt(
      quaternion.w * quaternion.w + quaternion.x * quaternion.x +
      quaternion.y * quaternion.y + quaternion.z * quaternion.z);
  if (!(norm > 1e-12)) {
    throw std::runtime_error("encountered a zero quaternion");
  }
  quaternion.w /= norm;
  quaternion.x /= norm;
  quaternion.y /= norm;
  quaternion.z /= norm;
  return quaternion;
}

Quaternion multiply(const Quaternion& left, const Quaternion& right) {
  return normalize({
      left.w * right.w - left.x * right.x - left.y * right.y -
          left.z * right.z,
      left.w * right.x + left.x * right.w + left.y * right.z -
          left.z * right.y,
      left.w * right.y - left.x * right.z + left.y * right.w +
          left.z * right.x,
      left.w * right.z + left.x * right.y - left.y * right.x +
          left.z * right.w,
  });
}

std::array<double, 3> rotate(const Quaternion& quaternion,
                             const std::array<double, 3>& vector) {
  const std::array<double, 3> axis = {quaternion.x, quaternion.y,
                                      quaternion.z};
  const std::array<double, 3> cross = {
      axis[1] * vector[2] - axis[2] * vector[1],
      axis[2] * vector[0] - axis[0] * vector[2],
      axis[0] * vector[1] - axis[1] * vector[0],
  };
  const std::array<double, 3> twice_cross = {
      2.0 * cross[0], 2.0 * cross[1], 2.0 * cross[2]};
  const std::array<double, 3> axis_cross = {
      axis[1] * twice_cross[2] - axis[2] * twice_cross[1],
      axis[2] * twice_cross[0] - axis[0] * twice_cross[2],
      axis[0] * twice_cross[1] - axis[1] * twice_cross[0],
  };
  return {
      vector[0] + quaternion.w * twice_cross[0] + axis_cross[0],
      vector[1] + quaternion.w * twice_cross[1] + axis_cross[1],
      vector[2] + quaternion.w * twice_cross[2] + axis_cross[2],
  };
}

struct Frame {
  std::string scene_name;
  std::size_t sequence_index = 0;
  std::size_t dataset_index = 0;
  std::string sample_token;
  std::int64_t timestamp_ns = 0;
  std::size_t point_count = 0;
  std::string point_path;
  std::array<double, 3> translation{};
  Quaternion rotation;
};

std::vector<Frame> readManifest(const std::string& scene_root,
                                const std::string& scene_name) {
  const std::string scene_directory = joinPath(scene_root, scene_name);
  const std::string manifest_path = joinPath(scene_directory, "frames.csv");
  std::ifstream stream(manifest_path);
  if (!stream) {
    throw std::runtime_error("cannot open scene manifest " + manifest_path);
  }
  std::string line;
  if (!std::getline(stream, line)) {
    throw std::runtime_error("missing scene manifest header in " +
                             manifest_path);
  }
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  if (line != "sequence_index,dataset_index,sample_token,timestamp_ns,"
              "point_count,relative_path,tx,ty,tz,qx,qy,qz,qw") {
    throw std::runtime_error("unexpected scene manifest header in " +
                             manifest_path);
  }
  std::vector<Frame> frames;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> fields = splitCsvLine(line);
    if (fields.size() != 13) {
      throw std::runtime_error("unexpected field count in " + manifest_path);
    }
    Frame frame;
    frame.scene_name = scene_name;
    frame.sequence_index = static_cast<std::size_t>(std::stoull(fields[0]));
    frame.dataset_index = static_cast<std::size_t>(std::stoull(fields[1]));
    frame.sample_token = fields[2];
    frame.timestamp_ns = std::stoll(fields[3]);
    frame.point_count = static_cast<std::size_t>(std::stoull(fields[4]));
    frame.point_path = joinPath(scene_directory, fields[5]);
    frame.translation = {std::stod(fields[6]), std::stod(fields[7]),
                         std::stod(fields[8])};
    frame.rotation = normalize({std::stod(fields[12]), std::stod(fields[9]),
                                std::stod(fields[10]), std::stod(fields[11])});
    frames.push_back(std::move(frame));
  }
  if (frames.empty()) {
    throw std::runtime_error("empty scene manifest " + manifest_path);
  }
  return frames;
}

std::vector<Frame> readFrames(const Options& options) {
  std::vector<Frame> frames;
  for (const std::string scene_name : {"scene-0103", "scene-0916"}) {
    std::vector<Frame> scene_frames =
        readManifest(options.scene_root, scene_name);
    frames.insert(frames.end(), std::make_move_iterator(scene_frames.begin()),
                  std::make_move_iterator(scene_frames.end()));
  }
  std::set<std::string> sample_tokens;
  for (std::size_t index = 0; index < frames.size(); ++index) {
    if (frames[index].dataset_index != index ||
        !sample_tokens.insert(frames[index].sample_token).second) {
      throw std::runtime_error(
          "scene manifests must contain unique contiguous dataset indices");
    }
  }
  if (frames.size() != 81) {
    throw std::runtime_error("expected exactly 81 mini-val frames");
  }
  if (options.max_frames != 0) {
    frames.resize(std::min(options.max_frames, frames.size()));
  }
  return frames;
}

std::string nvmlError(nvmlReturn_t result) {
  return std::string(nvmlErrorString(result));
}

class ProcessMemoryMonitor {
 public:
  struct Stats {
    std::uint64_t peak_process_bytes = 0;
    std::uint64_t device_baseline_bytes = 0;
    std::uint64_t peak_device_bytes = 0;
    std::uint64_t query_count = 0;
    std::uint64_t device_memory_sample_count = 0;
    std::uint64_t process_memory_sample_count = 0;
    std::string device_name;
    std::string driver_version;
  };

  ProcessMemoryMonitor(unsigned int gpu_index, int interval_ms)
      : process_id_(static_cast<unsigned int>(getpid())),
        interval_ms_(interval_ms) {
    nvmlReturn_t result = nvmlInit_v2();
    if (result != NVML_SUCCESS) {
      throw std::runtime_error("nvmlInit_v2 failed: " + nvmlError(result));
    }
    initialized_ = true;
    result = nvmlDeviceGetHandleByIndex_v2(gpu_index, &device_);
    if (result != NVML_SUCCESS) {
      nvmlShutdown();
      initialized_ = false;
      throw std::runtime_error("cannot get NVML device: " +
                               nvmlError(result));
    }
    char device_name[128] = {};
    result = nvmlDeviceGetName(device_, device_name, sizeof(device_name));
    if (result == NVML_SUCCESS) {
      device_name_ = device_name;
    }
    char driver_version[128] = {};
    result = nvmlSystemGetDriverVersion(driver_version,
                                        sizeof(driver_version));
    if (result == NVML_SUCCESS) {
      driver_version_ = driver_version;
    }
    nvmlMemory_t memory{};
    result = nvmlDeviceGetMemoryInfo(device_, &memory);
    if (result != NVML_SUCCESS) {
      nvmlShutdown();
      initialized_ = false;
      throw std::runtime_error("cannot read NVML device memory: " +
                               nvmlError(result));
    }
    device_baseline_bytes_ = memory.used;
    peak_device_bytes_.store(memory.used);
    device_memory_sample_count_.store(1);
    thread_ = std::thread([this] { run(); });
  }

  ~ProcessMemoryMonitor() {
    try {
      stop();
    } catch (...) {
    }
  }

  ProcessMemoryMonitor(const ProcessMemoryMonitor&) = delete;
  ProcessMemoryMonitor& operator=(const ProcessMemoryMonitor&) = delete;

  Stats stop() {
    if (!stopped_) {
      stop_requested_.store(true);
      if (thread_.joinable()) {
        thread_.join();
      }
      if (initialized_) {
        nvmlShutdown();
        initialized_ = false;
      }
      stopped_ = true;
    }
    std::string error;
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error = error_;
    }
    if (!error.empty()) {
      throw std::runtime_error("NVML monitor failed: " + error);
    }
    if (device_memory_sample_count_.load() == 0) {
      throw std::runtime_error(
          "NVML did not report device memory during inference");
    }
    return {peak_process_bytes_.load(), device_baseline_bytes_,
            peak_device_bytes_.load(), query_count_.load(),
            device_memory_sample_count_.load(),
            process_memory_sample_count_.load(), device_name_,
            driver_version_};
  }

 private:
  std::optional<std::uint64_t> queryCurrentProcessBytes() {
    unsigned int count = 0;
    nvmlReturn_t result =
        nvmlDeviceGetComputeRunningProcesses(device_, &count, nullptr);
    if (result == NVML_SUCCESS && count == 0) {
      return std::nullopt;
    }
    if (result != NVML_ERROR_INSUFFICIENT_SIZE) {
      throw std::runtime_error("process query failed: " + nvmlError(result));
    }
    for (int attempt = 0; attempt < 2; ++attempt) {
      std::vector<nvmlProcessInfo_t> processes(count + 8);
      unsigned int capacity = static_cast<unsigned int>(processes.size());
      result = nvmlDeviceGetComputeRunningProcesses(device_, &capacity,
                                                     processes.data());
      if (result == NVML_ERROR_INSUFFICIENT_SIZE) {
        count = capacity;
        continue;
      }
      if (result != NVML_SUCCESS) {
        throw std::runtime_error("process query failed: " +
                                 nvmlError(result));
      }
      for (unsigned int index = 0; index < capacity; ++index) {
        if (processes[index].pid == process_id_ &&
            processes[index].usedGpuMemory !=
                static_cast<unsigned long long>(NVML_VALUE_NOT_AVAILABLE)) {
          return processes[index].usedGpuMemory;
        }
      }
      return std::nullopt;
    }
    throw std::runtime_error("NVML process list changed repeatedly");
  }

  void run() noexcept {
    try {
      while (!stop_requested_.load()) {
        ++query_count_;
        nvmlMemory_t memory{};
        const nvmlReturn_t memory_result =
            nvmlDeviceGetMemoryInfo(device_, &memory);
        if (memory_result != NVML_SUCCESS) {
          throw std::runtime_error("device memory query failed: " +
                                   nvmlError(memory_result));
        }
        ++device_memory_sample_count_;
        std::uint64_t previous_device = peak_device_bytes_.load();
        while (previous_device < memory.used &&
               !peak_device_bytes_.compare_exchange_weak(previous_device,
                                                         memory.used)) {
        }
        const std::optional<std::uint64_t> process_bytes =
            queryCurrentProcessBytes();
        if (process_bytes.has_value()) {
          ++process_memory_sample_count_;
          std::uint64_t previous_process = peak_process_bytes_.load();
          while (previous_process < *process_bytes &&
                 !peak_process_bytes_.compare_exchange_weak(
                     previous_process, *process_bytes)) {
          }
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(interval_ms_));
      }
    } catch (const std::exception& exception) {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error_ = exception.what();
    }
  }

  unsigned int process_id_ = 0;
  int interval_ms_ = 1;
  nvmlDevice_t device_ = nullptr;
  bool initialized_ = false;
  bool stopped_ = false;
  std::string device_name_;
  std::string driver_version_;
  std::uint64_t device_baseline_bytes_ = 0;
  std::atomic<bool> stop_requested_{false};
  std::atomic<std::uint64_t> peak_process_bytes_{0};
  std::atomic<std::uint64_t> peak_device_bytes_{0};
  std::atomic<std::uint64_t> query_count_{0};
  std::atomic<std::uint64_t> device_memory_sample_count_{0};
  std::atomic<std::uint64_t> process_memory_sample_count_{0};
  std::thread thread_;
  std::mutex error_mutex_;
  std::string error_;
};

std::string attributeFor(int label, double speed) {
  const std::string name = kClassNames.at(static_cast<std::size_t>(label));
  if (speed > 0.2) {
    if (name == "car" || name == "construction_vehicle" || name == "bus" ||
        name == "truck" || name == "trailer") {
      return "vehicle.moving";
    }
    if (name == "bicycle" || name == "motorcycle") {
      return "cycle.with_rider";
    }
  } else {
    if (name == "pedestrian") {
      return "pedestrian.standing";
    }
    if (name == "bus") {
      return "vehicle.stopped";
    }
  }
  return kDefaultAttributes.at(static_cast<std::size_t>(label));
}

struct GlobalDetection {
  std::array<double, 3> translation{};
  std::array<double, 3> size{};
  Quaternion rotation;
  std::array<double, 2> velocity{};
  std::string name;
  double score = 0.0;
  std::string attribute;
};

GlobalDetection transformDetection(const Frame& frame,
                                   const cpd::Detection& detection) {
  if (detection.label < 0 ||
      detection.label >= static_cast<int>(kClassNames.size())) {
    throw std::runtime_error("detection has an invalid class label");
  }
  const std::array<double, 3> local_center = {
      detection.box[0], detection.box[1],
      detection.box[2] + detection.box[5] * 0.5};
  const std::array<double, 3> rotated_center =
      rotate(frame.rotation, local_center);
  const std::array<double, 3> global_velocity =
      rotate(frame.rotation,
             {detection.box[7], detection.box[8], 0.0});
  const double half_yaw = detection.box[6] * 0.5;
  const Quaternion orientation = multiply(
      frame.rotation, {std::cos(half_yaw), 0.0, 0.0, std::sin(half_yaw)});
  const double speed =
      std::hypot(static_cast<double>(detection.box[7]),
                 static_cast<double>(detection.box[8]));
  GlobalDetection result;
  result.translation = {rotated_center[0] + frame.translation[0],
                        rotated_center[1] + frame.translation[1],
                        rotated_center[2] + frame.translation[2]};
  result.size = {detection.box[4], detection.box[3], detection.box[5]};
  result.rotation = orientation;
  result.velocity = {global_velocity[0], global_velocity[1]};
  result.name = kClassNames.at(static_cast<std::size_t>(detection.label));
  result.score = detection.score;
  result.attribute = attributeFor(detection.label, speed);
  return result;
}

struct FrameResult {
  Frame frame;
  double point_load_ms = 0.0;
  cpd::InferenceTiming timing;
  std::vector<GlobalDetection> detections;
};

std::filesystem::path prepareOutput(const std::string& path) {
  const std::filesystem::path output =
      std::filesystem::absolute(path).lexically_normal();
  const std::filesystem::path temporary = output.string() + ".tmp";
  if (std::filesystem::exists(output) || std::filesystem::exists(temporary)) {
    throw std::runtime_error("refusing to overwrite output " +
                             output.string());
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  return output;
}

std::ofstream openTemporary(const std::filesystem::path& output) {
  std::ofstream stream(output.string() + ".tmp");
  if (!stream) {
    throw std::runtime_error("cannot create temporary output for " +
                             output.string());
  }
  return stream;
}

void finishOutput(std::ofstream& stream, const std::filesystem::path& output) {
  stream.close();
  if (!stream) {
    std::filesystem::remove(output.string() + ".tmp");
    throw std::runtime_error("failed while writing " + output.string());
  }
  std::filesystem::rename(output.string() + ".tmp", output);
}

std::string escapeJson(const std::string& value) {
  std::string escaped;
  for (const char character : value) {
    if (character == '"' || character == '\\') {
      escaped.push_back('\\');
      escaped.push_back(character);
    } else if (character == '\n') {
      escaped += "\\n";
    } else {
      escaped.push_back(character);
    }
  }
  return escaped;
}

void writeDetectionResult(const std::filesystem::path& output,
                          const std::vector<FrameResult>& frames) {
  std::ofstream stream = openTemporary(output);
  stream << std::setprecision(10)
         << "{\n  \"meta\": {\"use_camera\": false, \"use_lidar\": true, "
            "\"use_radar\": false, \"use_map\": false, "
            "\"use_external\": false},\n  \"results\": {\n";
  for (std::size_t frame_index = 0; frame_index < frames.size();
       ++frame_index) {
    const FrameResult& frame = frames[frame_index];
    stream << "    \"" << frame.frame.sample_token << "\": [";
    for (std::size_t detection_index = 0;
         detection_index < frame.detections.size(); ++detection_index) {
      const GlobalDetection& detection = frame.detections[detection_index];
      if (detection_index != 0) {
        stream << ',';
      }
      stream << "\n      {\"sample_token\": \""
             << frame.frame.sample_token << "\", \"translation\": ["
             << detection.translation[0] << ", " << detection.translation[1]
             << ", " << detection.translation[2] << "], \"size\": ["
             << detection.size[0] << ", " << detection.size[1] << ", "
             << detection.size[2] << "], \"rotation\": ["
             << detection.rotation.w << ", " << detection.rotation.x << ", "
             << detection.rotation.y << ", " << detection.rotation.z
             << "], \"velocity\": [" << detection.velocity[0] << ", "
             << detection.velocity[1] << "], \"detection_name\": \""
             << detection.name << "\", \"detection_score\": "
             << detection.score << ", \"attribute_name\": \""
             << detection.attribute << "\"}";
    }
    if (!frame.detections.empty()) {
      stream << '\n' << "    ";
    }
    stream << ']';
    if (frame_index + 1 != frames.size()) {
      stream << ',';
    }
    stream << '\n';
  }
  stream << "  }\n}\n";
  finishOutput(stream, output);
}

void writeFrameTimings(const std::filesystem::path& output,
                       const std::vector<FrameResult>& frames) {
  std::ofstream stream = openTemporary(output);
  stream << "dataset_index,scene_name,sample_token,point_count,voxel_count,"
            "detection_count,point_load_ms,cpu_voxelization_ms,gpu_pipeline_ms,"
            "decode_ms,end_to_end_ms\n";
  stream << std::fixed << std::setprecision(6);
  for (const FrameResult& frame : frames) {
    stream << frame.frame.dataset_index << ',' << frame.frame.scene_name << ','
           << frame.frame.sample_token << ',' << frame.timing.point_count << ','
           << frame.timing.voxel_count << ',' << frame.detections.size() << ','
           << frame.point_load_ms << ',' << frame.timing.cpu_voxelization_ms
           << ',' << frame.timing.gpu_pipeline_ms << ','
           << frame.timing.decode_ms << ',' << frame.timing.end_to_end_ms
           << '\n';
  }
  finishOutput(stream, output);
}

double mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::round(fraction * static_cast<double>(values.size() - 1)));
  return values[index];
}

struct TimingSummary {
  double mean = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double maximum = 0.0;
};

TimingSummary summarize(const std::vector<double>& values) {
  return {mean(values), percentile(values, 0.50), percentile(values, 0.95),
          *std::max_element(values.begin(), values.end())};
}

void writeTimingSummary(std::ofstream& stream, const std::string& name,
                        const TimingSummary& summary, bool comma) {
  stream << "    \"" << name << "\": {\"mean\": " << summary.mean
         << ", \"p50\": " << summary.p50 << ", \"p95\": "
         << summary.p95 << ", \"max\": " << summary.maximum << '}';
  if (comma) {
    stream << ',';
  }
  stream << '\n';
}

void writeBenchmarkReport(
    const std::filesystem::path& output, const Options& options,
    const std::vector<FrameResult>& frames,
    const ProcessMemoryMonitor::Stats& memory_stats,
    std::size_t explicit_buffer_bytes, double total_wall_ms,
    const std::filesystem::path& result_path,
    const std::filesystem::path& timing_path) {
  std::vector<double> load_ms;
  std::vector<double> voxel_ms;
  std::vector<double> gpu_ms;
  std::vector<double> decode_ms;
  std::vector<double> end_to_end_ms;
  std::uint64_t point_count = 0;
  std::uint64_t detection_count = 0;
  for (const FrameResult& frame : frames) {
    load_ms.push_back(frame.point_load_ms);
    voxel_ms.push_back(frame.timing.cpu_voxelization_ms);
    gpu_ms.push_back(frame.timing.gpu_pipeline_ms);
    decode_ms.push_back(frame.timing.decode_ms);
    end_to_end_ms.push_back(frame.timing.end_to_end_ms);
    point_count += frame.timing.point_count;
    detection_count += frame.detections.size();
  }
  std::ofstream stream = openTemporary(output);
  const bool process_accounting_available =
      memory_stats.process_memory_sample_count != 0;
  const std::uint64_t peak_device_delta_bytes =
      memory_stats.peak_device_bytes > memory_stats.device_baseline_bytes
          ? memory_stats.peak_device_bytes - memory_stats.device_baseline_bytes
          : 0;
  stream << std::fixed << std::setprecision(8)
         << "{\n"
         << "  \"runtime\": \"C++/CUDA/TensorRT\",\n"
         << "  \"dataset\": \"nuScenes v1.0-mini mini_val\",\n"
         << "  \"frame_count\": " << frames.size() << ",\n"
         << "  \"point_count\": " << point_count << ",\n"
         << "  \"detection_count\": " << detection_count << ",\n"
         << "  \"score_threshold\": " << options.score_threshold << ",\n"
         << "  \"timing_scope\": \"per-frame point tensor to decoded 3D "
            "boxes; file I/O reported separately\",\n"
         << "  \"timing_ms\": {\n";
  writeTimingSummary(stream, "point_load", summarize(load_ms), true);
  writeTimingSummary(stream, "cpu_voxelization", summarize(voxel_ms), true);
  writeTimingSummary(stream, "gpu_pipeline", summarize(gpu_ms), true);
  writeTimingSummary(stream, "cpp_decode_circle_nms", summarize(decode_ms),
                     true);
  writeTimingSummary(stream, "end_to_end", summarize(end_to_end_ms), false);
  stream << "  },\n"
         << "  \"full_sequence_wall_ms\": " << total_wall_ms << ",\n"
         << "  \"explicit_cuda_buffer_mib\": "
         << explicit_buffer_bytes / static_cast<double>(1 << 20) << ",\n"
         << "  \"nvml\": {\n"
         << "    \"scope\": \"device used-memory sampled from before CUDA "
            "initialization through all inference frames; delta includes CUDA "
            "context, TensorRT engine/context/workspace and explicit "
            "buffers when no concurrent GPU workload changes\",\n"
         << "    \"gpu_index\": " << options.gpu_index << ",\n"
         << "    \"device_name\": \""
         << escapeJson(memory_stats.device_name) << "\",\n"
         << "    \"driver_version\": \""
         << escapeJson(memory_stats.driver_version) << "\",\n"
         << "    \"sample_interval_ms\": " << options.nvml_interval_ms
         << ",\n"
         << "    \"query_count\": " << memory_stats.query_count << ",\n"
         << "    \"device_memory_sample_count\": "
         << memory_stats.device_memory_sample_count << ",\n"
         << "    \"process_memory_sample_count\": "
         << memory_stats.process_memory_sample_count << ",\n"
         << "    \"process_accounting_available\": "
         << (process_accounting_available ? "true" : "false") << ",\n"
         << "    \"peak_process_memory_mib\": ";
  if (process_accounting_available) {
    stream << memory_stats.peak_process_bytes / static_cast<double>(1 << 20);
  } else {
    stream << "null";
  }
  stream << ",\n"
         << "    \"device_baseline_used_mib\": "
         << memory_stats.device_baseline_bytes / static_cast<double>(1 << 20)
         << ",\n"
         << "    \"peak_device_used_mib\": "
         << memory_stats.peak_device_bytes / static_cast<double>(1 << 20)
         << ",\n"
         << "    \"peak_device_memory_delta_mib\": "
         << peak_device_delta_bytes / static_cast<double>(1 << 20) << "\n"
         << "  },\n"
         << "  \"engine\": \"" << escapeJson(options.engine) << "\",\n"
         << "  \"result\": \"" << escapeJson(result_path.string())
         << "\",\n"
         << "  \"frame_timings\": \"" << escapeJson(timing_path.string())
         << "\"\n"
         << "}\n";
  finishOutput(stream, output);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseOptions(argc, argv);
    const std::filesystem::path result_path = prepareOutput(options.result);
    const std::filesystem::path timing_path = prepareOutput(options.timings);
    const std::filesystem::path report_path = prepareOutput(options.report);
    const std::vector<Frame> frames = readFrames(options);

    ProcessMemoryMonitor memory_monitor(
        static_cast<unsigned int>(options.gpu_index), options.nvml_interval_ms);
    if (cudaSetDevice(options.gpu_index) != cudaSuccess) {
      throw std::runtime_error("cudaSetDevice failed");
    }
    cpd::CenterPointDetector detector(
        {options.parameter_directory, options.engine, options.score_threshold});

    std::vector<FrameResult> results;
    results.reserve(frames.size());
    const auto sequence_start = std::chrono::steady_clock::now();
    for (const Frame& frame : frames) {
      const auto load_start = std::chrono::steady_clock::now();
      const cpd::Array<float> points = cpd::readArray<float>(frame.point_path);
      const auto load_stop = std::chrono::steady_clock::now();
      if (points.shape.size() != 2 || points.shape[1] != cpd::kPointChannels ||
          points.shape[0] != frame.point_count) {
        throw std::runtime_error("unexpected point tensor in " +
                                 frame.point_path);
      }
      FrameResult result;
      result.frame = frame;
      result.point_load_ms =
          std::chrono::duration<double, std::milli>(load_stop - load_start)
              .count();
      const std::vector<cpd::Detection> local_detections =
          detector.infer(points.values, &result.timing);
      result.detections.reserve(local_detections.size());
      for (const cpd::Detection& detection : local_detections) {
        result.detections.push_back(transformDetection(frame, detection));
      }
      std::cout << "frame=" << (results.size() + 1) << '/' << frames.size()
                << " dataset_index=" << frame.dataset_index
                << " points=" << result.timing.point_count
                << " detections=" << result.detections.size()
                << " end_to_end_ms=" << std::fixed << std::setprecision(3)
                << result.timing.end_to_end_ms << '\n';
      results.push_back(std::move(result));
    }
    const auto sequence_stop = std::chrono::steady_clock::now();
    const ProcessMemoryMonitor::Stats memory_stats = memory_monitor.stop();
    const double total_wall_ms =
        std::chrono::duration<double, std::milli>(sequence_stop - sequence_start)
            .count();

    writeDetectionResult(result_path, results);
    writeFrameTimings(timing_path, results);
    writeBenchmarkReport(report_path, options, results, memory_stats,
                         detector.explicitCudaBufferBytes(), total_wall_ms,
                         result_path, timing_path);

    std::vector<double> end_to_end_ms;
    for (const FrameResult& result : results) {
      end_to_end_ms.push_back(result.timing.end_to_end_ms);
    }
    const TimingSummary summary = summarize(end_to_end_ms);
    const std::uint64_t peak_device_delta_bytes =
        memory_stats.peak_device_bytes > memory_stats.device_baseline_bytes
            ? memory_stats.peak_device_bytes -
                  memory_stats.device_baseline_bytes
            : 0;
    std::cout << "summary frames=" << results.size()
              << " mean_ms=" << summary.mean << " p50_ms=" << summary.p50
              << " p95_ms=" << summary.p95
              << " nvml_peak_device_delta_mib="
              << peak_device_delta_bytes / static_cast<double>(1 << 20)
              << " process_accounting_available="
              << (memory_stats.process_memory_sample_count != 0 ? "true"
                                                                 : "false")
              << '\n'
              << "result=" << result_path << '\n'
              << "timings=" << timing_path << '\n'
              << "report=" << report_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
