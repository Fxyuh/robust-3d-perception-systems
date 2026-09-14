#pragma once

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace centerpoint_ros::tracking {

enum class AssociationMethod { kGreedy, kHungarian, kHungarianClasswise };
enum class MotionModel { kNone, kKalman };

struct Detection {
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
  double center_z = 0.0;
  Eigen::Vector3d dimensions = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  float score = 0.0F;
  int label = -1;
};

struct TrackOutput {
  std::uint64_t id = 0;
  Detection object;
  std::uint32_t age = 0;
  std::uint32_t hits = 0;
  std::uint32_t time_since_update = 0;
  bool confirmed = false;
  std::vector<Eigen::Vector2d> history;
};

struct TrackerConfig {
  AssociationMethod association_method = AssociationMethod::kHungarianClasswise;
  MotionModel motion_model = MotionModel::kKalman;
  std::uint32_t max_age = 3;
  std::uint32_t min_hits = 2;
  std::size_t max_history = 20;
  double process_noise = 4.0;
  double position_measurement_noise = 1.0;
  double velocity_measurement_noise = 4.0;
  std::array<double, 10> class_distance_gates = {
      4.0, 4.0, 3.0, 5.5, 3.0, 1.5, 4.0, 2.0, 1.0, 1.0};
};

class MultiObjectTracker {
 public:
  explicit MultiObjectTracker(TrackerConfig config = {});
  ~MultiObjectTracker();

  std::vector<TrackOutput> update(const std::vector<Detection>& detections,
                                  std::int64_t timestamp_ns);
  void reset();
  std::size_t activeTrackCount() const;

 private:
  struct TrackState;

  TrackerConfig config_;
  std::vector<TrackState> tracks_;
  std::uint64_t next_id_ = 1;
  std::int64_t last_timestamp_ns_ = 0;
};

}  // namespace centerpoint_ros::tracking
