#include "array_io.hpp"
#include "centerpoint_ros/msg/frame_metadata.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace centerpoint_ros {
namespace {

constexpr std::size_t kPointChannels = 5;
constexpr std::size_t kSequenceColumns = 13;

struct FrameRecord {
  std::size_t sequence_index = 0;
  std::size_t dataset_index = 0;
  std::string sample_token;
  std::int64_t timestamp_ns = 0;
  std::size_t point_count = 0;
  std::string points_path;
  geometry_msgs::msg::Pose lidar_pose_global;
};

std::vector<std::string> splitCsvRow(const std::string& row) {
  std::vector<std::string> columns;
  std::istringstream stream(row);
  std::string column;
  while (std::getline(stream, column, ',')) {
    if (!column.empty() && column.back() == '\r') {
      column.pop_back();
    }
    columns.push_back(column);
  }
  return columns;
}

std::vector<FrameRecord> loadSequence(const std::string& manifest_path) {
  std::ifstream stream(manifest_path);
  if (!stream) {
    throw std::runtime_error("cannot open sequence manifest " + manifest_path);
  }
  std::string row;
  if (!std::getline(stream, row)) {
    throw std::runtime_error("empty sequence manifest " + manifest_path);
  }
  const std::vector<std::string> header = splitCsvRow(row);
  if (header.size() != kSequenceColumns || header[0] != "sequence_index" ||
      header[2] != "sample_token" || header[3] != "timestamp_ns" ||
      header[5] != "relative_path" || header[12] != "qw") {
    throw std::runtime_error("unexpected sequence manifest schema");
  }

  const std::filesystem::path base =
      std::filesystem::absolute(manifest_path).parent_path();
  std::vector<FrameRecord> frames;
  while (std::getline(stream, row)) {
    if (row.empty()) {
      continue;
    }
    const std::vector<std::string> columns = splitCsvRow(row);
    if (columns.size() != kSequenceColumns) {
      throw std::runtime_error("invalid sequence manifest row");
    }
    FrameRecord frame;
    frame.sequence_index = std::stoull(columns[0]);
    frame.dataset_index = std::stoull(columns[1]);
    frame.sample_token = columns[2];
    frame.timestamp_ns = std::stoll(columns[3]);
    frame.point_count = std::stoull(columns[4]);
    frame.points_path = (base / columns[5]).lexically_normal().string();
    frame.lidar_pose_global.position.x = std::stod(columns[6]);
    frame.lidar_pose_global.position.y = std::stod(columns[7]);
    frame.lidar_pose_global.position.z = std::stod(columns[8]);
    frame.lidar_pose_global.orientation.x = std::stod(columns[9]);
    frame.lidar_pose_global.orientation.y = std::stod(columns[10]);
    frame.lidar_pose_global.orientation.z = std::stod(columns[11]);
    frame.lidar_pose_global.orientation.w = std::stod(columns[12]);
    if (frame.sequence_index != frames.size() || frame.sample_token.empty() ||
        frame.timestamp_ns <= 0 || frame.point_count == 0) {
      throw std::runtime_error("invalid sequence frame metadata");
    }
    if (!frames.empty() && frame.timestamp_ns <= frames.back().timestamp_ns) {
      throw std::runtime_error("sequence timestamps must be strictly increasing");
    }
    const auto& q = frame.lidar_pose_global.orientation;
    const double quaternion_norm =
        std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (std::abs(quaternion_norm - 1.0) > 1e-5) {
      throw std::runtime_error("sequence pose quaternion is not normalized");
    }
    frames.push_back(std::move(frame));
  }
  if (frames.empty()) {
    throw std::runtime_error("sequence manifest contains no frames");
  }
  return frames;
}

sensor_msgs::msg::PointCloud2 makePointCloud(
    const cpd::Array<float>& points, const std::string& frame_id,
    const builtin_interfaces::msg::Time& stamp) {
  if (points.shape.size() != 2 || points.shape[1] != kPointChannels ||
      points.shape[0] == 0) {
    throw std::runtime_error("points file must contain a non-empty N x 5 array");
  }
  sensor_msgs::msg::PointCloud2 message;
  message.header.frame_id = frame_id;
  message.header.stamp = stamp;
  message.height = 1;
  message.width = static_cast<std::uint32_t>(points.shape[0]);
  message.is_bigendian = false;
  message.is_dense = true;
  sensor_msgs::PointCloud2Modifier modifier(message);
  modifier.setPointCloud2Fields(
      5, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
      sensor_msgs::msg::PointField::FLOAT32, "z", 1,
      sensor_msgs::msg::PointField::FLOAT32, "intensity", 1,
      sensor_msgs::msg::PointField::FLOAT32, "time_lag", 1,
      sensor_msgs::msg::PointField::FLOAT32);
  modifier.resize(points.shape[0]);

  sensor_msgs::PointCloud2Iterator<float> x(message, "x");
  sensor_msgs::PointCloud2Iterator<float> y(message, "y");
  sensor_msgs::PointCloud2Iterator<float> z(message, "z");
  sensor_msgs::PointCloud2Iterator<float> intensity(message, "intensity");
  sensor_msgs::PointCloud2Iterator<float> time_lag(message, "time_lag");
  for (std::size_t index = 0; index < points.shape[0];
       ++index, ++x, ++y, ++z, ++intensity, ++time_lag) {
    const float* point = points.values.data() + index * kPointChannels;
    *x = point[0];
    *y = point[1];
    *z = point[2];
    *intensity = point[3];
    *time_lag = point[4];
  }
  return message;
}

}  // namespace

class PointCloudReplayNode final : public rclcpp::Node {
 public:
  PointCloudReplayNode() : Node("pointcloud_replay_node") {
    const std::string points_path =
        declare_parameter<std::string>("points_path", "");
    const std::string sequence_manifest =
        declare_parameter<std::string>("sequence_manifest", "");
    const std::string point_topic =
        declare_parameter<std::string>("point_topic", "/lidar_points");
    const std::string metadata_topic = declare_parameter<std::string>(
        "metadata_topic", "/centerpoint/frame_metadata");
    sensor_frame_ = declare_parameter<std::string>("frame_id", "lidar");
    global_frame_ = declare_parameter<std::string>("global_frame", "map");
    scene_name_ = declare_parameter<std::string>("scene_name", "single_sample");
    loop_ = declare_parameter<bool>("loop", false);
    const double publish_rate_hz =
        declare_parameter<double>("publish_rate_hz", 2.0);
    const double startup_delay_s =
        declare_parameter<double>("startup_delay_s", 0.0);
    if (points_path.empty() == sequence_manifest.empty()) {
      throw std::runtime_error(
          "set exactly one of points_path or sequence_manifest");
    }
    if (!(publish_rate_hz > 0.0)) {
      throw std::runtime_error("publish_rate_hz must be positive");
    }
    if (startup_delay_s < 0.0) {
      throw std::runtime_error("startup_delay_s must be non-negative");
    }

    if (!sequence_manifest.empty()) {
      frames_ = loadSequence(sequence_manifest);
      sequence_mode_ = true;
    } else {
      FrameRecord frame;
      frame.points_path = points_path;
      frame.lidar_pose_global.orientation.w = 1.0;
      frames_.push_back(std::move(frame));
    }

    point_publisher_ =
        create_publisher<sensor_msgs::msg::PointCloud2>(point_topic, 10);
    metadata_publisher_ =
        create_publisher<msg::FrameMetadata>(metadata_topic, 10);
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / publish_rate_hz));
    startup_deadline_ = std::chrono::steady_clock::now() +
                        std::chrono::duration_cast<
                            std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(startup_delay_s));
    timer_ = create_wall_timer(period, [this]() { publishNext(); });
    RCLCPP_INFO(get_logger(),
                "loaded %zu frame(s); publishing %.3f Hz on %s "
                "(loop=%s startup_delay=%.2f s)",
                frames_.size(), publish_rate_hz, point_topic.c_str(),
                loop_ ? "true" : "false", startup_delay_s);
  }

 private:
  void publishNext() {
    try {
      if (std::chrono::steady_clock::now() < startup_deadline_ ||
          point_publisher_->get_subscription_count() == 0) {
        return;
      }
      if (next_frame_ == frames_.size()) {
        if (loop_) {
          next_frame_ = 0;
        } else {
          timer_->cancel();
          RCLCPP_INFO(get_logger(), "scene playback complete: %zu frame(s)",
                      frames_.size());
          return;
        }
      }
      const FrameRecord& frame = frames_[next_frame_];
      const cpd::Array<float> points =
          cpd::readArray<float>(frame.points_path);
      if (sequence_mode_ && points.shape[0] != frame.point_count) {
        throw std::runtime_error("point count differs from sequence manifest");
      }
      const rclcpp::Time ros_time =
          sequence_mode_ ? rclcpp::Time(frame.timestamp_ns) : now();
      const builtin_interfaces::msg::Time stamp = ros_time;
      sensor_msgs::msg::PointCloud2 point_message =
          makePointCloud(points, sensor_frame_, stamp);

      msg::FrameMetadata metadata;
      metadata.header.stamp = stamp;
      metadata.header.frame_id = global_frame_;
      metadata.sensor_frame = sensor_frame_;
      metadata.sequence_index =
          static_cast<std::uint32_t>(frame.sequence_index);
      metadata.frame_count = static_cast<std::uint32_t>(frames_.size());
      metadata.scene_name = scene_name_;
      metadata.sample_token = frame.sample_token;
      metadata.lidar_pose_global = frame.lidar_pose_global;
      metadata_publisher_->publish(metadata);
      point_publisher_->publish(point_message);

      RCLCPP_INFO(get_logger(),
                  "frame=%zu/%zu dataset_index=%zu points=%zu token=%s",
                  next_frame_ + 1, frames_.size(), frame.dataset_index,
                  points.shape[0],
                  frame.sample_token.empty() ? "n/a" : frame.sample_token.c_str());
      ++next_frame_;
    } catch (const std::exception& error) {
      timer_->cancel();
      RCLCPP_ERROR(get_logger(), "playback stopped: %s", error.what());
    }
  }

  bool sequence_mode_ = false;
  bool loop_ = false;
  std::string sensor_frame_;
  std::string global_frame_;
  std::string scene_name_;
  std::size_t next_frame_ = 0;
  std::chrono::steady_clock::time_point startup_deadline_;
  std::vector<FrameRecord> frames_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_publisher_;
  rclcpp::Publisher<msg::FrameMetadata>::SharedPtr metadata_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace centerpoint_ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<centerpoint_ros::PointCloudReplayNode>());
  } catch (const std::exception& error) {
    std::cerr << "pointcloud_replay_node error: " << error.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
