#include "centerpoint_detector.hpp"
#include "centerpoint_ros/msg/detection3_d.hpp"
#include "centerpoint_ros/msg/detection3_d_array.hpp"

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace centerpoint_ros {
namespace {

constexpr std::array<const char*, 10> kClassNames = {
    "car",          "truck",   "construction_vehicle", "bus",
    "trailer",      "barrier", "motorcycle",            "bicycle",
    "pedestrian",   "traffic_cone"};

constexpr std::array<std::array<float, 3>, 10> kClassColors = {{
    {0.15F, 0.75F, 1.00F}, {1.00F, 0.45F, 0.10F}, {0.75F, 0.55F, 0.20F},
    {1.00F, 0.80F, 0.10F}, {0.75F, 0.25F, 0.85F}, {0.55F, 0.55F, 0.55F},
    {1.00F, 0.15F, 0.35F}, {0.25F, 0.95F, 0.35F}, {0.20F, 1.00F, 0.80F},
    {1.00F, 0.35F, 0.75F},
}};

const sensor_msgs::msg::PointField* findFloatField(
    const sensor_msgs::msg::PointCloud2& message, const std::string& name,
    bool required) {
  for (const auto& field : message.fields) {
    if (field.name == name) {
      if (field.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
          field.count != 1) {
        throw std::runtime_error("PointCloud2 field '" + name +
                                 "' must be scalar FLOAT32");
      }
      if (field.offset + sizeof(float) > message.point_step) {
        throw std::runtime_error("PointCloud2 field '" + name +
                                 "' exceeds point_step");
      }
      return &field;
    }
  }
  if (required) {
    throw std::runtime_error("PointCloud2 is missing required field '" + name +
                             "'");
  }
  return nullptr;
}

float readFloat(const std::uint8_t* point, std::uint32_t offset) {
  float value = 0.0F;
  std::memcpy(&value, point + offset, sizeof(value));
  return value;
}

std::vector<float> unpackPoints(
    const sensor_msgs::msg::PointCloud2& message) {
  if (message.is_bigendian) {
    throw std::runtime_error("big-endian PointCloud2 is not supported");
  }
  const auto* x = findFloatField(message, "x", true);
  const auto* y = findFloatField(message, "y", true);
  const auto* z = findFloatField(message, "z", true);
  const auto* intensity = findFloatField(message, "intensity", true);
  const auto* time_lag = findFloatField(message, "time_lag", false);
  if (message.row_step < message.point_step * message.width ||
      message.data.size() <
          static_cast<std::size_t>(message.row_step) * message.height) {
    throw std::runtime_error("PointCloud2 data layout is inconsistent");
  }

  std::vector<float> points;
  points.reserve(static_cast<std::size_t>(message.width) * message.height * 5);
  for (std::uint32_t row = 0; row < message.height; ++row) {
    const std::uint8_t* row_data =
        message.data.data() + static_cast<std::size_t>(row) * message.row_step;
    for (std::uint32_t column = 0; column < message.width; ++column) {
      const std::uint8_t* point =
          row_data + static_cast<std::size_t>(column) * message.point_step;
      const float values[5] = {
          readFloat(point, x->offset), readFloat(point, y->offset),
          readFloat(point, z->offset), readFloat(point, intensity->offset),
          time_lag == nullptr ? 0.0F : readFloat(point, time_lag->offset)};
      if (std::isfinite(values[0]) && std::isfinite(values[1]) &&
          std::isfinite(values[2]) && std::isfinite(values[3]) &&
          std::isfinite(values[4])) {
        points.insert(points.end(), std::begin(values), std::end(values));
      }
    }
  }
  if (points.empty()) {
    throw std::runtime_error("PointCloud2 contains no finite points");
  }
  return points;
}

geometry_msgs::msg::Point makePoint(double x, double y, double z) {
  geometry_msgs::msg::Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  return point;
}

visualization_msgs::msg::Marker makeBoxMarker(
    const std_msgs::msg::Header& header, const cpd::Detection& detection,
    int marker_id) {
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.ns = "centerpoint_boxes";
  marker.id = marker_id;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.08;

  const auto& color = kClassColors.at(static_cast<std::size_t>(detection.label));
  marker.color.r = color[0];
  marker.color.g = color[1];
  marker.color.b = color[2];
  marker.color.a = 0.95F;

  const double half_x = detection.box[3] * 0.5;
  const double half_y = detection.box[4] * 0.5;
  const double bottom_z = detection.box[2];
  const double top_z = bottom_z + detection.box[5];
  const double cosine = std::cos(detection.box[6]);
  const double sine = std::sin(detection.box[6]);
  const std::array<std::array<double, 2>, 4> local = {{
      {half_x, half_y}, {half_x, -half_y}, {-half_x, -half_y},
      {-half_x, half_y},
  }};
  std::array<geometry_msgs::msg::Point, 8> corners;
  for (std::size_t index = 0; index < local.size(); ++index) {
    const double rotated_x = cosine * local[index][0] - sine * local[index][1];
    const double rotated_y = sine * local[index][0] + cosine * local[index][1];
    corners[index] = makePoint(detection.box[0] + rotated_x,
                               detection.box[1] + rotated_y, bottom_z);
    corners[index + 4] = makePoint(detection.box[0] + rotated_x,
                                   detection.box[1] + rotated_y, top_z);
  }
  constexpr std::array<std::array<int, 2>, 12> kEdges = {{
      {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
      {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
  }};
  for (const auto& edge : kEdges) {
    marker.points.push_back(corners[edge[0]]);
    marker.points.push_back(corners[edge[1]]);
  }
  return marker;
}

visualization_msgs::msg::Marker makeTextMarker(
    const std_msgs::msg::Header& header, const cpd::Detection& detection,
    int marker_id) {
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.ns = "centerpoint_labels";
  marker.id = marker_id;
  marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = detection.box[0];
  marker.pose.position.y = detection.box[1];
  marker.pose.position.z = detection.box[2] + detection.box[5] + 0.35F;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.42;
  marker.color.r = 1.0F;
  marker.color.g = 1.0F;
  marker.color.b = 1.0F;
  marker.color.a = 0.95F;
  std::ostringstream text;
  text << kClassNames.at(static_cast<std::size_t>(detection.label)) << ' '
       << std::fixed << std::setprecision(2) << detection.score;
  marker.text = text.str();
  return marker;
}

}  // namespace

class CenterPointNode final : public rclcpp::Node {
 public:
  CenterPointNode() : Node("centerpoint_node") {
    const std::string parameter_directory = declare_parameter<std::string>(
        "parameter_directory", "");
    const std::string engine_path =
        declare_parameter<std::string>("engine_path", "");
    const std::string point_topic =
        declare_parameter<std::string>("point_topic", "/lidar_points");
    const std::string detection_topic = declare_parameter<std::string>(
        "detection_topic", "/centerpoint/detections");
    const std::string marker_topic = declare_parameter<std::string>(
        "marker_topic", "/centerpoint/markers");
    const double score_threshold =
        declare_parameter<double>("score_threshold", 0.1);
    marker_score_threshold_ =
        declare_parameter<double>("marker_score_threshold", 0.2);
    const bool reliable_point_qos =
        declare_parameter<bool>("reliable_point_qos", false);
    if (parameter_directory.empty() || engine_path.empty()) {
      throw std::runtime_error(
          "parameter_directory and engine_path parameters are required");
    }

    detector_ = std::make_unique<cpd::CenterPointDetector>(cpd::DetectorConfig{
        parameter_directory, engine_path, static_cast<float>(score_threshold)});
    detection_publisher_ =
        create_publisher<msg::Detection3DArray>(detection_topic, 10);
    marker_publisher_ =
        create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic, 10);
    auto point_qos = rclcpp::SensorDataQoS();
    if (reliable_point_qos) {
      point_qos.reliable();
    }
    point_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        point_topic, point_qos,
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
          onPointCloud(std::move(message));
        });
    RCLCPP_INFO(get_logger(),
                "ready: input=%s detections=%s markers=%s point_qos=%s",
                point_topic.c_str(), detection_topic.c_str(),
                marker_topic.c_str(),
                reliable_point_qos ? "reliable" : "best_effort");
  }

 private:
  void onPointCloud(sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
    try {
      const std::vector<float> points = unpackPoints(*message);
      cpd::InferenceTiming timing;
      const std::vector<cpd::Detection> detections =
          detector_->infer(points, &timing);

      msg::Detection3DArray output;
      output.header = message->header;
      output.point_count = static_cast<std::uint32_t>(timing.point_count);
      output.voxel_count = static_cast<std::uint32_t>(timing.voxel_count);
      output.inference_ms = static_cast<float>(timing.end_to_end_ms);
      output.detections.reserve(detections.size());
      for (const auto& detection : detections) {
        if (detection.label < 0 ||
            detection.label >= static_cast<int>(kClassNames.size())) {
          continue;
        }
        msg::Detection3D item;
        item.box = detection.box;
        item.score = detection.score;
        item.label = detection.label;
        item.class_name = kClassNames[static_cast<std::size_t>(detection.label)];
        output.detections.push_back(std::move(item));
      }
      detection_publisher_->publish(output);

      visualization_msgs::msg::MarkerArray markers;
      visualization_msgs::msg::Marker clear;
      clear.header = message->header;
      clear.action = visualization_msgs::msg::Marker::DELETEALL;
      markers.markers.push_back(clear);
      int marker_id = 0;
      for (const auto& detection : detections) {
        if (detection.score < marker_score_threshold_ || detection.label < 0 ||
            detection.label >= static_cast<int>(kClassNames.size())) {
          continue;
        }
        markers.markers.push_back(
            makeBoxMarker(message->header, detection, marker_id));
        markers.markers.push_back(
            makeTextMarker(message->header, detection, marker_id));
        ++marker_id;
      }
      marker_publisher_->publish(markers);

      RCLCPP_INFO(get_logger(),
                  "frame=%s points=%zu voxels=%zu detections=%zu "
                  "inference=%.3f ms",
                  message->header.frame_id.c_str(), timing.point_count,
                  timing.voxel_count, output.detections.size(),
                  timing.end_to_end_ms);
    } catch (const std::exception& error) {
      RCLCPP_ERROR(get_logger(), "point-cloud inference failed: %s",
                   error.what());
    }
  }

  std::unique_ptr<cpd::CenterPointDetector> detector_;
  double marker_score_threshold_ = 0.2;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      point_subscription_;
  rclcpp::Publisher<msg::Detection3DArray>::SharedPtr detection_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_publisher_;
};

}  // namespace centerpoint_ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<centerpoint_ros::CenterPointNode>());
  } catch (const std::exception& error) {
    std::cerr << "centerpoint_node error: " << error.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
