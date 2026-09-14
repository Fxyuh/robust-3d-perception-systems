#include "centerpoint_ros/multi_object_tracker.hpp"
#include "centerpoint_ros/msg/detection3_d_array.hpp"
#include "centerpoint_ros/msg/frame_metadata.hpp"
#include "centerpoint_ros/msg/tracked_object3_d.hpp"
#include "centerpoint_ros/msg/tracked_object3_d_array.hpp"

#include <Eigen/Geometry>

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
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

std::int64_t stampNanoseconds(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

double wrapAngle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

Eigen::Quaterniond poseQuaternion(const geometry_msgs::msg::Pose& pose) {
  Eigen::Quaterniond quaternion(pose.orientation.w, pose.orientation.x,
                                pose.orientation.y, pose.orientation.z);
  if (quaternion.norm() < 1e-8) {
    throw std::runtime_error("LiDAR pose contains a zero quaternion");
  }
  quaternion.normalize();
  return quaternion;
}

Eigen::Vector3d poseTranslation(const geometry_msgs::msg::Pose& pose) {
  return {pose.position.x, pose.position.y, pose.position.z};
}

struct EvaluationTrack {
  std::string tracking_id;
  std::string tracking_name;
  std::array<double, 3> translation{};
  std::array<double, 3> size{};
  std::array<double, 4> rotation{};
  std::array<double, 2> velocity{};
  double tracking_score = 0.0;
};

bool isOfficialTrackingClass(int label) {
  return label == 0 || label == 1 || label == 3 || label == 4 || label == 6 ||
         label == 7 || label == 8;
}

void writeEvaluationResult(
    const std::string& result_path,
    const std::map<std::string, std::vector<EvaluationTrack>>& results) {
  const std::filesystem::path output =
      std::filesystem::absolute(result_path).lexically_normal();
  const std::filesystem::path temporary = output.string() + ".tmp";
  if (std::filesystem::exists(output) || std::filesystem::exists(temporary)) {
    throw std::runtime_error("refusing to overwrite tracking result " +
                             output.string());
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  std::ofstream stream(temporary);
  if (!stream) {
    throw std::runtime_error("cannot create tracking result " +
                             temporary.string());
  }
  stream << std::setprecision(10);
  stream << "{\n  \"meta\": {\"use_camera\": false, \"use_lidar\": true, "
            "\"use_radar\": false, \"use_map\": false, "
            "\"use_external\": false},\n  \"results\": {\n";
  std::size_t sample_index = 0;
  for (const auto& [sample_token, tracks] : results) {
    stream << "    \"" << sample_token << "\": [";
    for (std::size_t index = 0; index < tracks.size(); ++index) {
      const auto& track = tracks[index];
      if (index != 0) {
        stream << ',';
      }
      stream << "\n      {\"sample_token\": \"" << sample_token
             << "\", \"translation\": [" << track.translation[0] << ", "
             << track.translation[1] << ", " << track.translation[2]
             << "], \"size\": [" << track.size[0] << ", " << track.size[1]
             << ", " << track.size[2] << "], \"rotation\": ["
             << track.rotation[0] << ", " << track.rotation[1] << ", "
             << track.rotation[2] << ", " << track.rotation[3]
             << "], \"velocity\": [" << track.velocity[0] << ", "
             << track.velocity[1] << "], \"tracking_id\": \""
             << track.tracking_id << "\", \"tracking_name\": \""
             << track.tracking_name << "\", \"tracking_score\": "
             << track.tracking_score << '}';
    }
    if (!tracks.empty()) {
      stream << '\n' << "    ";
    }
    stream << ']';
    if (++sample_index != results.size()) {
      stream << ',';
    }
    stream << '\n';
  }
  stream << "  }\n}\n";
  stream.close();
  if (!stream) {
    std::filesystem::remove(temporary);
    throw std::runtime_error("failed while writing tracking result " +
                             temporary.string());
  }
  std::filesystem::rename(temporary, output);
}

geometry_msgs::msg::Point makePoint(double x, double y, double z) {
  geometry_msgs::msg::Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  return point;
}

visualization_msgs::msg::Marker makeBoxMarker(
    const std_msgs::msg::Header& header, const msg::TrackedObject3D& track) {
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.ns = "tracked_boxes";
  marker.id = static_cast<int>(track.track_id);
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.10;
  const auto& color = kClassColors.at(static_cast<std::size_t>(track.label));
  marker.color.r = color[0];
  marker.color.g = color[1];
  marker.color.b = color[2];
  marker.color.a = track.time_since_update == 0 ? 1.0F : 0.45F;

  const double half_x = track.box[3] * 0.5;
  const double half_y = track.box[4] * 0.5;
  const double bottom_z = track.box[2];
  const double top_z = bottom_z + track.box[5];
  const double cosine = std::cos(track.box[6]);
  const double sine = std::sin(track.box[6]);
  const std::array<std::array<double, 2>, 4> local = {{
      {half_x, half_y}, {half_x, -half_y}, {-half_x, -half_y},
      {-half_x, half_y},
  }};
  std::array<geometry_msgs::msg::Point, 8> corners;
  for (std::size_t index = 0; index < local.size(); ++index) {
    const double x = cosine * local[index][0] - sine * local[index][1];
    const double y = sine * local[index][0] + cosine * local[index][1];
    corners[index] = makePoint(track.box[0] + x, track.box[1] + y, bottom_z);
    corners[index + 4] =
        makePoint(track.box[0] + x, track.box[1] + y, top_z);
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
    const std_msgs::msg::Header& header, const msg::TrackedObject3D& track) {
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.ns = "track_ids";
  marker.id = static_cast<int>(track.track_id);
  marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = track.box[0];
  marker.pose.position.y = track.box[1];
  marker.pose.position.z = track.box[2] + track.box[5] + 0.35F;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.45;
  marker.color.r = 1.0F;
  marker.color.g = 1.0F;
  marker.color.b = 1.0F;
  marker.color.a = track.time_since_update == 0 ? 1.0F : 0.45F;
  std::ostringstream text;
  text << track.class_name << " #" << track.track_id << ' ' << std::fixed
       << std::setprecision(2) << track.score;
  marker.text = text.str();
  return marker;
}

visualization_msgs::msg::Marker makeTrailMarker(
    const std_msgs::msg::Header& header, const tracking::TrackOutput& track,
    const Eigen::Quaterniond& lidar_to_global,
    const Eigen::Vector3d& translation) {
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.ns = "track_trails";
  marker.id = static_cast<int>(track.id);
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.12;
  const auto& color =
      kClassColors.at(static_cast<std::size_t>(track.object.label));
  marker.color.r = color[0];
  marker.color.g = color[1];
  marker.color.b = color[2];
  marker.color.a = 0.85F;
  for (const auto& global_xy : track.history) {
    const Eigen::Vector3d global_point(global_xy.x(), global_xy.y(),
                                       track.object.center_z);
    const Eigen::Vector3d local_point =
        lidar_to_global.conjugate() * (global_point - translation);
    marker.points.push_back(
        makePoint(local_point.x(), local_point.y(), local_point.z()));
  }
  return marker;
}

}  // namespace

class CenterPointTrackerNode final : public rclcpp::Node {
 public:
  CenterPointTrackerNode() : Node("centerpoint_tracker_node") {
    const std::string detection_topic = declare_parameter<std::string>(
        "detection_topic", "/centerpoint/detections");
    const std::string metadata_topic = declare_parameter<std::string>(
        "metadata_topic", "/centerpoint/frame_metadata");
    const std::string track_topic = declare_parameter<std::string>(
        "track_topic", "/centerpoint/tracked_objects");
    const std::string marker_topic = declare_parameter<std::string>(
        "marker_topic", "/centerpoint/track_markers");
    minimum_score_ = declare_parameter<double>("minimum_score", 0.2);
    publish_tentative_ =
        declare_parameter<bool>("publish_tentative", false);
    result_path_ = declare_parameter<std::string>("result_path", "");
    const std::string association_method =
        declare_parameter<std::string>("association_method",
                                       "hungarian_classwise");
    const std::string motion_model =
        declare_parameter<std::string>("motion_model", "kalman");

    tracking::TrackerConfig config;
    if (association_method == "greedy") {
      config.association_method = tracking::AssociationMethod::kGreedy;
    } else if (association_method == "hungarian") {
      config.association_method = tracking::AssociationMethod::kHungarian;
    } else if (association_method == "hungarian_classwise") {
      config.association_method =
          tracking::AssociationMethod::kHungarianClasswise;
    } else {
      throw std::runtime_error(
          "association_method must be greedy, hungarian, or "
          "hungarian_classwise");
    }
    if (motion_model == "none") {
      config.motion_model = tracking::MotionModel::kNone;
    } else if (motion_model == "kalman") {
      config.motion_model = tracking::MotionModel::kKalman;
    } else {
      throw std::runtime_error("motion_model must be none or kalman");
    }
    config.max_age = static_cast<std::uint32_t>(
        declare_parameter<int>("max_age", static_cast<int>(config.max_age)));
    config.min_hits = static_cast<std::uint32_t>(
        declare_parameter<int>("min_hits", static_cast<int>(config.min_hits)));
    config.max_history = static_cast<std::size_t>(declare_parameter<int>(
        "max_history", static_cast<int>(config.max_history)));
    config.process_noise =
        declare_parameter<double>("process_noise", config.process_noise);
    config.position_measurement_noise = declare_parameter<double>(
        "position_measurement_noise", config.position_measurement_noise);
    config.velocity_measurement_noise = declare_parameter<double>(
        "velocity_measurement_noise", config.velocity_measurement_noise);
    const std::vector<double> gates = declare_parameter<std::vector<double>>(
        "class_distance_gates",
        std::vector<double>(config.class_distance_gates.begin(),
                            config.class_distance_gates.end()));
    if (gates.size() != config.class_distance_gates.size()) {
      throw std::runtime_error("class_distance_gates must contain 10 values");
    }
    std::copy(gates.begin(), gates.end(), config.class_distance_gates.begin());
    if (minimum_score_ < 0.0 || minimum_score_ > 1.0) {
      throw std::runtime_error("minimum_score must be in [0, 1]");
    }
    if (!result_path_.empty()) {
      const std::filesystem::path output =
          std::filesystem::absolute(result_path_).lexically_normal();
      const std::filesystem::path temporary = output.string() + ".tmp";
      if (std::filesystem::exists(output) ||
          std::filesystem::exists(temporary)) {
        throw std::runtime_error("refusing to overwrite tracking result " +
                                 output.string());
      }
    }
    tracker_ = std::make_unique<tracking::MultiObjectTracker>(config);

    track_publisher_ =
        create_publisher<msg::TrackedObject3DArray>(track_topic, 10);
    marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        marker_topic, 10);
    const auto qos = rclcpp::QoS(10).reliable();
    detection_subscription_ = create_subscription<msg::Detection3DArray>(
        detection_topic, qos,
        [this](msg::Detection3DArray::ConstSharedPtr message) {
          const auto stamp = stampNanoseconds(message->header.stamp);
          if (processed_stamps_.count(stamp) != 0) {
            return;
          }
          detections_[stamp] = std::move(message);
          tryProcess(stamp);
          trimCaches();
        });
    metadata_subscription_ = create_subscription<msg::FrameMetadata>(
        metadata_topic, qos,
        [this](msg::FrameMetadata::ConstSharedPtr message) {
          const auto stamp = stampNanoseconds(message->header.stamp);
          if (message->sequence_index == 0 ||
              (!current_scene_.empty() &&
               message->scene_name != current_scene_)) {
            processed_stamps_.clear();
            detections_.clear();
            metadata_.clear();
          } else if (processed_stamps_.count(stamp) != 0) {
            return;
          }
          metadata_[stamp] = std::move(message);
          tryProcess(stamp);
          trimCaches();
        });
    RCLCPP_INFO(get_logger(),
                "ready: detections=%s metadata=%s tracks=%s markers=%s "
                "association=%s motion=%s",
                detection_topic.c_str(), metadata_topic.c_str(),
                track_topic.c_str(), marker_topic.c_str(),
                association_method.c_str(), motion_model.c_str());
  }

 private:
  void trimCaches() {
    constexpr std::size_t kMaximumPendingFrames = 32;
    while (detections_.size() > kMaximumPendingFrames) {
      RCLCPP_WARN(get_logger(), "dropping unmatched detection metadata stamp");
      detections_.erase(detections_.begin());
    }
    while (metadata_.size() > kMaximumPendingFrames) {
      RCLCPP_WARN(get_logger(), "dropping unmatched frame metadata stamp");
      metadata_.erase(metadata_.begin());
    }
  }

  void tryProcess(std::int64_t timestamp_ns) {
    const auto detection = detections_.find(timestamp_ns);
    const auto metadata = metadata_.find(timestamp_ns);
    if (detection == detections_.end() || metadata == metadata_.end()) {
      return;
    }
    const auto detection_message = detection->second;
    const auto metadata_message = metadata->second;
    detections_.erase(detection);
    metadata_.erase(metadata);
    try {
      processFrame(*detection_message, *metadata_message, timestamp_ns);
    } catch (const std::exception& error) {
      RCLCPP_ERROR(get_logger(), "tracking failed: %s", error.what());
    }
    processed_stamps_.insert(timestamp_ns);
    while (processed_stamps_.size() > 64) {
      processed_stamps_.erase(processed_stamps_.begin());
    }
  }

  void processFrame(const msg::Detection3DArray& detections,
                    const msg::FrameMetadata& metadata,
                    std::int64_t timestamp_ns) {
    if (metadata.scene_name.empty() || metadata.sample_token.empty()) {
      throw std::runtime_error("frame metadata is missing scene identity");
    }
    const bool reset_required =
        !initialized_ || metadata.scene_name != current_scene_ ||
        metadata.sequence_index == 0 || timestamp_ns <= last_timestamp_ns_;
    if (reset_required) {
      tracker_->reset();
      frame_timings_ms_.clear();
      evaluation_results_.clear();
      current_scene_ = metadata.scene_name;
      initialized_ = true;
      RCLCPP_INFO(get_logger(), "tracker reset for scene=%s",
                  current_scene_.c_str());
    }
    last_timestamp_ns_ = timestamp_ns;

    const auto start = std::chrono::steady_clock::now();
    const Eigen::Quaterniond lidar_to_global =
        poseQuaternion(metadata.lidar_pose_global);
    const Eigen::Vector3d translation =
        poseTranslation(metadata.lidar_pose_global);
    std::vector<tracking::Detection> global_detections;
    global_detections.reserve(detections.detections.size());
    for (const auto& detection : detections.detections) {
      if (detection.score < minimum_score_ || detection.label < 0 ||
          detection.label >= static_cast<int>(kClassNames.size()) ||
          detection.box[3] <= 0.0F || detection.box[4] <= 0.0F ||
          detection.box[5] <= 0.0F) {
        continue;
      }
      tracking::Detection transformed;
      const Eigen::Vector3d local_center(
          detection.box[0], detection.box[1],
          detection.box[2] + detection.box[5] * 0.5);
      const Eigen::Vector3d global_center =
          lidar_to_global * local_center + translation;
      const Eigen::Vector3d global_velocity =
          lidar_to_global * Eigen::Vector3d(detection.box[7], detection.box[8],
                                            0.0);
      const Eigen::Vector3d global_heading =
          lidar_to_global * Eigen::Vector3d(std::cos(detection.box[6]),
                                            std::sin(detection.box[6]), 0.0);
      transformed.center = global_center.head<2>();
      transformed.velocity = global_velocity.head<2>();
      transformed.center_z = global_center.z();
      transformed.dimensions = {detection.box[3], detection.box[4],
                                detection.box[5]};
      transformed.yaw = std::atan2(global_heading.y(), global_heading.x());
      transformed.score = detection.score;
      transformed.label = detection.label;
      global_detections.push_back(std::move(transformed));
    }

    const std::vector<tracking::TrackOutput> tracks =
        tracker_->update(global_detections, timestamp_ns);
    msg::TrackedObject3DArray output;
    output.header = detections.header;
    output.sequence_index = metadata.sequence_index;
    output.frame_count = metadata.frame_count;
    output.scene_name = metadata.scene_name;
    output.sample_token = metadata.sample_token;

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker clear;
    clear.header = detections.header;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear);
    for (const auto& track : tracks) {
      if (!track.confirmed && !publish_tentative_) {
        continue;
      }
      if (!result_path_.empty() &&
          isOfficialTrackingClass(track.object.label)) {
        EvaluationTrack result;
        result.tracking_id = metadata.scene_name + "-" +
                             std::to_string(track.id);
        result.tracking_name =
            kClassNames.at(static_cast<std::size_t>(track.object.label));
        result.translation = {track.object.center.x(), track.object.center.y(),
                              track.object.center_z};
        // CenterPoint stores length, width, height; nuScenes expects w, l, h.
        result.size = {track.object.dimensions.y(),
                       track.object.dimensions.x(),
                       track.object.dimensions.z()};
        result.rotation = {std::cos(track.object.yaw * 0.5), 0.0, 0.0,
                           std::sin(track.object.yaw * 0.5)};
        result.velocity = {track.object.velocity.x(),
                           track.object.velocity.y()};
        result.tracking_score = track.object.score *
            std::pow(0.9, static_cast<double>(track.time_since_update));
        evaluation_results_[metadata.sample_token].push_back(
            std::move(result));
      }
      const Eigen::Vector3d global_center(track.object.center.x(),
                                          track.object.center.y(),
                                          track.object.center_z);
      const Eigen::Vector3d local_center =
          lidar_to_global.conjugate() * (global_center - translation);
      const Eigen::Vector3d local_velocity = lidar_to_global.conjugate() *
          Eigen::Vector3d(track.object.velocity.x(),
                          track.object.velocity.y(), 0.0);
      const Eigen::Vector3d local_heading = lidar_to_global.conjugate() *
          Eigen::Vector3d(std::cos(track.object.yaw),
                          std::sin(track.object.yaw), 0.0);

      msg::TrackedObject3D item;
      item.track_id = track.id;
      item.confirmed = track.confirmed;
      item.box = {static_cast<float>(local_center.x()),
                  static_cast<float>(local_center.y()),
                  static_cast<float>(local_center.z() -
                                     track.object.dimensions.z() * 0.5),
                  static_cast<float>(track.object.dimensions.x()),
                  static_cast<float>(track.object.dimensions.y()),
                  static_cast<float>(track.object.dimensions.z()),
                  static_cast<float>(wrapAngle(
                      std::atan2(local_heading.y(), local_heading.x()))),
                  static_cast<float>(local_velocity.x()),
                  static_cast<float>(local_velocity.y())};
      item.score = track.object.score;
      item.label = track.object.label;
      item.class_name =
          kClassNames.at(static_cast<std::size_t>(track.object.label));
      item.age = track.age;
      item.hits = track.hits;
      item.time_since_update = track.time_since_update;
      output.tracks.push_back(item);
      markers.markers.push_back(makeBoxMarker(output.header, item));
      markers.markers.push_back(makeTextMarker(output.header, item));
      if (track.history.size() >= 2) {
        markers.markers.push_back(makeTrailMarker(
            output.header, track, lidar_to_global, translation));
      }
    }
    const auto end = std::chrono::steady_clock::now();
    output.tracking_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(end - start).count());
    frame_timings_ms_.push_back(output.tracking_ms);
    if (!result_path_.empty()) {
      // Required even when no official tracking class is emitted this frame.
      evaluation_results_.try_emplace(metadata.sample_token);
    }
    track_publisher_->publish(output);
    marker_publisher_->publish(markers);
    RCLCPP_INFO(get_logger(),
                "scene=%s frame=%u/%u detections=%zu active=%zu published=%zu "
                "tracking=%.3f ms",
                metadata.scene_name.c_str(), metadata.sequence_index + 1,
                metadata.frame_count, global_detections.size(), tracks.size(),
                output.tracks.size(), output.tracking_ms);
    if (metadata.sequence_index + 1 == metadata.frame_count) {
      std::vector<float> sorted_timings = frame_timings_ms_;
      std::sort(sorted_timings.begin(), sorted_timings.end());
      const double mean = std::accumulate(sorted_timings.begin(),
                                          sorted_timings.end(), 0.0) /
                          sorted_timings.size();
      const std::size_t p95_index = static_cast<std::size_t>(
          std::ceil(sorted_timings.size() * 0.95) - 1.0);
      RCLCPP_INFO(get_logger(),
                  "scene summary: frames=%zu tracking_mean=%.3f ms "
                  "tracking_p95=%.3f ms tracking_max=%.3f ms",
                  sorted_timings.size(), mean, sorted_timings[p95_index],
                  sorted_timings.back());
      if (!result_path_.empty()) {
        writeEvaluationResult(result_path_, evaluation_results_);
        RCLCPP_INFO(get_logger(), "wrote official tracking JSON: %s",
                    result_path_.c_str());
      }
    }
  }

  double minimum_score_ = 0.2;
  bool publish_tentative_ = false;
  bool initialized_ = false;
  std::string current_scene_;
  std::string result_path_;
  std::int64_t last_timestamp_ns_ = 0;
  std::vector<float> frame_timings_ms_;
  std::unique_ptr<tracking::MultiObjectTracker> tracker_;
  std::set<std::int64_t> processed_stamps_;
  std::map<std::string, std::vector<EvaluationTrack>> evaluation_results_;
  std::map<std::int64_t, msg::Detection3DArray::ConstSharedPtr> detections_;
  std::map<std::int64_t, msg::FrameMetadata::ConstSharedPtr> metadata_;
  rclcpp::Subscription<msg::Detection3DArray>::SharedPtr
      detection_subscription_;
  rclcpp::Subscription<msg::FrameMetadata>::SharedPtr metadata_subscription_;
  rclcpp::Publisher<msg::TrackedObject3DArray>::SharedPtr track_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_publisher_;
};

}  // namespace centerpoint_ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<centerpoint_ros::CenterPointTrackerNode>());
  } catch (const std::exception& error) {
    std::cerr << "centerpoint_tracker_node error: " << error.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
