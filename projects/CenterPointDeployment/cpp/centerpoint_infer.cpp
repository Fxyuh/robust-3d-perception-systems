#include "array_io.hpp"
#include "centerpoint_detector.hpp"
#include "pillar_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string input_dir = "work_dirs/centerpoint_deployment/preprocess";
  std::string engine = "work_dirs/centerpoint_deployment/tensorrt/"
                       "centerpoint_dense_mixed_fp16_deep.engine";
  std::string detections = "work_dirs/centerpoint_deployment/tensorrt/"
                           "detections_cpp_f32.cpd";
  std::string report = "work_dirs/centerpoint_deployment/tensorrt/"
                       "centerpoint_cpp_end_to_end.json";
  int warmup = 20;
  int iterations = 50;
};

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--input-dir" && index + 1 < argc) {
      options.input_dir = argv[++index];
    } else if (argument == "--engine" && index + 1 < argc) {
      options.engine = argv[++index];
    } else if (argument == "--detections" && index + 1 < argc) {
      options.detections = argv[++index];
    } else if (argument == "--report" && index + 1 < argc) {
      options.report = argv[++index];
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

void writeDetections(const std::string& path,
                     const std::vector<cpd::Detection>& detections) {
  std::vector<float> values;
  values.reserve(detections.size() * 11);
  for (const cpd::Detection& detection : detections) {
    values.insert(values.end(), detection.box.begin(), detection.box.end());
    values.push_back(detection.score);
    values.push_back(static_cast<float>(detection.label));
  }
  cpd::writeArray<float>(path, {detections.size(), 11}, values);
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseOptions(argc, argv);
    const auto points =
        cpd::readArray<float>(joinPath(options.input_dir, "points_f32.cpd"));
    if (points.shape.size() != 2 || points.shape[1] != cpd::kPointChannels) {
      throw std::runtime_error("unexpected point tensor shape");
    }

    cpd::CenterPointDetector detector(
        {options.input_dir, options.engine, 0.1F});
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      detector.infer(points.values);
    }

    std::vector<double> voxel_ms;
    std::vector<double> gpu_ms;
    std::vector<double> decode_ms;
    std::vector<double> end_to_end_ms;
    voxel_ms.reserve(options.iterations);
    gpu_ms.reserve(options.iterations);
    decode_ms.reserve(options.iterations);
    end_to_end_ms.reserve(options.iterations);
    cpd::InferenceTiming timing;
    std::vector<cpd::Detection> detections;
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      detections = detector.infer(points.values, &timing);
      voxel_ms.push_back(timing.cpu_voxelization_ms);
      gpu_ms.push_back(timing.gpu_pipeline_ms);
      decode_ms.push_back(timing.decode_ms);
      end_to_end_ms.push_back(timing.end_to_end_ms);
    }
    writeDetections(options.detections, detections);

    const double mean_end_to_end = mean(end_to_end_ms);
    const std::size_t explicit_buffer_bytes =
        detector.explicitCudaBufferBytes();
    std::ofstream report(options.report);
    if (!report) {
      throw std::runtime_error("cannot create report " + options.report);
    }
    report << std::fixed << std::setprecision(8)
           << "{\n"
           << "  \"runtime\": \"C++/CUDA/TensorRT\",\n"
           << "  \"point_count\": " << timing.point_count << ",\n"
           << "  \"voxel_count\": " << timing.voxel_count << ",\n"
           << "  \"detection_count_score_gt_0_1\": " << detections.size()
           << ",\n"
           << "  \"iterations\": " << options.iterations << ",\n"
           << "  \"mean_cpu_voxelization_ms\": " << mean(voxel_ms) << ",\n"
           << "  \"mean_gpu_transfer_preprocess_tensorrt_d2h_ms\": "
           << mean(gpu_ms) << ",\n"
           << "  \"mean_cpp_decode_circle_nms_ms\": " << mean(decode_ms)
           << ",\n"
           << "  \"mean_end_to_end_ms\": " << mean_end_to_end << ",\n"
           << "  \"p50_end_to_end_ms\": "
           << percentile(end_to_end_ms, 0.50) << ",\n"
           << "  \"p95_end_to_end_ms\": "
           << percentile(end_to_end_ms, 0.95) << ",\n"
           << "  \"fps_end_to_end\": " << 1000.0 / mean_end_to_end << ",\n"
           << "  \"explicit_cuda_buffer_mib\": "
           << explicit_buffer_bytes / static_cast<double>(1 << 20) << ",\n"
           << "  \"engine\": \"" << options.engine << "\",\n"
           << "  \"detections\": \"" << options.detections << "\"\n"
           << "}\n";

    std::cout << std::fixed << std::setprecision(4)
              << "points=" << timing.point_count
              << " voxels=" << timing.voxel_count
              << " detections=" << detections.size() << '\n'
              << "cpu_voxelization_ms=" << mean(voxel_ms) << '\n'
              << "gpu_transfer_preprocess_tensorrt_d2h_ms=" << mean(gpu_ms)
              << '\n'
              << "cpp_decode_circle_nms_ms=" << mean(decode_ms) << '\n'
              << "end_to_end_mean_ms=" << mean_end_to_end
              << " p50_ms=" << percentile(end_to_end_ms, 0.50)
              << " p95_ms=" << percentile(end_to_end_ms, 0.95) << '\n'
              << "end_to_end_fps=" << 1000.0 / mean_end_to_end << '\n'
              << "detections=" << options.detections << '\n'
              << "report=" << options.report << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
