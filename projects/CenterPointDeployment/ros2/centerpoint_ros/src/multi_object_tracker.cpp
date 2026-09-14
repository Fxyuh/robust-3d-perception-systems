#include "centerpoint_ros/multi_object_tracker.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace centerpoint_ros::tracking {
namespace {

constexpr double kInvalidCost = 1.0e6;

double wrapAngle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

// Solves a rectangular minimum-cost assignment. Each row is assigned at most
// once; -1 denotes an unassigned row when there are fewer columns than rows.
std::vector<int> assignRowsHungarian(
    const std::vector<std::vector<double>>& cost) {
  const std::size_t row_count = cost.size();
  if (row_count == 0) {
    return {};
  }
  const std::size_t column_count = cost.front().size();
  if (column_count == 0) {
    return std::vector<int>(row_count, -1);
  }
  for (const auto& row : cost) {
    if (row.size() != column_count) {
      throw std::invalid_argument("assignment cost matrix is not rectangular");
    }
  }

  if (row_count > column_count) {
    std::vector<std::vector<double>> transposed(
        column_count, std::vector<double>(row_count));
    for (std::size_t row = 0; row < row_count; ++row) {
      for (std::size_t column = 0; column < column_count; ++column) {
        transposed[column][row] = cost[row][column];
      }
    }
    const std::vector<int> reverse = assignRowsHungarian(transposed);
    std::vector<int> assignment(row_count, -1);
    for (std::size_t detection = 0; detection < reverse.size(); ++detection) {
      if (reverse[detection] >= 0) {
        assignment[static_cast<std::size_t>(reverse[detection])] =
            static_cast<int>(detection);
      }
    }
    return assignment;
  }

  // Hungarian algorithm, using 1-based indexing for potentials and links.
  std::vector<double> row_potential(row_count + 1, 0.0);
  std::vector<double> column_potential(column_count + 1, 0.0);
  std::vector<int> matched_row(column_count + 1, 0);
  std::vector<int> predecessor(column_count + 1, 0);
  for (std::size_t row = 1; row <= row_count; ++row) {
    matched_row[0] = static_cast<int>(row);
    int column0 = 0;
    std::vector<double> minimum(column_count + 1,
                                std::numeric_limits<double>::infinity());
    std::vector<bool> used(column_count + 1, false);
    do {
      used[static_cast<std::size_t>(column0)] = true;
      const int row0 = matched_row[static_cast<std::size_t>(column0)];
      double delta = std::numeric_limits<double>::infinity();
      int column1 = 0;
      for (std::size_t column = 1; column <= column_count; ++column) {
        if (used[column]) {
          continue;
        }
        const double reduced =
            cost[static_cast<std::size_t>(row0 - 1)][column - 1] -
            row_potential[static_cast<std::size_t>(row0)] -
            column_potential[column];
        if (reduced < minimum[column]) {
          minimum[column] = reduced;
          predecessor[column] = column0;
        }
        if (minimum[column] < delta) {
          delta = minimum[column];
          column1 = static_cast<int>(column);
        }
      }
      for (std::size_t column = 0; column <= column_count; ++column) {
        if (used[column]) {
          row_potential[static_cast<std::size_t>(matched_row[column])] += delta;
          column_potential[column] -= delta;
        } else {
          minimum[column] -= delta;
        }
      }
      column0 = column1;
    } while (matched_row[static_cast<std::size_t>(column0)] != 0);

    do {
      const int column1 = predecessor[static_cast<std::size_t>(column0)];
      matched_row[static_cast<std::size_t>(column0)] =
          matched_row[static_cast<std::size_t>(column1)];
      column0 = column1;
    } while (column0 != 0);
  }

  std::vector<int> assignment(row_count, -1);
  for (std::size_t column = 1; column <= column_count; ++column) {
    if (matched_row[column] != 0) {
      assignment[static_cast<std::size_t>(matched_row[column] - 1)] =
          static_cast<int>(column - 1);
    }
  }
  return assignment;
}

std::vector<int> assignRowsGreedy(
    const std::vector<std::vector<double>>& cost) {
  struct Candidate {
    double cost = 0.0;
    std::size_t row = 0;
    std::size_t column = 0;
  };
  std::vector<int> assignment(cost.size(), -1);
  if (cost.empty() || cost.front().empty()) {
    return assignment;
  }
  std::vector<Candidate> candidates;
  for (std::size_t row = 0; row < cost.size(); ++row) {
    for (std::size_t column = 0; column < cost[row].size(); ++column) {
      if (cost[row][column] < kInvalidCost * 0.5) {
        candidates.push_back(Candidate{cost[row][column], row, column});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& left, const Candidate& right) {
              if (left.cost != right.cost) {
                return left.cost < right.cost;
              }
              if (left.row != right.row) {
                return left.row < right.row;
              }
              return left.column < right.column;
            });
  std::vector<bool> used_columns(cost.front().size(), false);
  for (const auto& candidate : candidates) {
    if (assignment[candidate.row] == -1 &&
        !used_columns[candidate.column]) {
      assignment[candidate.row] = static_cast<int>(candidate.column);
      used_columns[candidate.column] = true;
    }
  }
  return assignment;
}

std::vector<int> assignRowsHungarianClasswise(
    const std::vector<std::vector<double>>& cost,
    const std::vector<int>& row_labels, const std::vector<int>& column_labels,
    std::size_t class_count) {
  if (cost.size() != row_labels.size()) {
    throw std::invalid_argument("assignment row labels have invalid size");
  }
  if (!cost.empty() && cost.front().size() != column_labels.size()) {
    throw std::invalid_argument("assignment column labels have invalid size");
  }
  std::vector<int> assignment(cost.size(), -1);
  constexpr double kUnmatchedCost = 1000.0;
  for (std::size_t label = 0; label < class_count; ++label) {
    std::vector<std::size_t> rows;
    std::vector<std::size_t> columns;
    for (std::size_t row = 0; row < row_labels.size(); ++row) {
      if (row_labels[row] == static_cast<int>(label)) {
        rows.push_back(row);
      }
    }
    for (std::size_t column = 0; column < column_labels.size(); ++column) {
      if (column_labels[column] == static_cast<int>(label)) {
        columns.push_back(column);
      }
    }
    if (rows.empty() || columns.empty()) {
      continue;
    }

    // Real-to-real entries carry gated distance. Dedicated dummy rows and
    // columns permit tracks and detections to remain unmatched instead of
    // forcing an invalid high-cost pair.
    const std::size_t size = rows.size() + columns.size();
    std::vector<std::vector<double>> block(
        size, std::vector<double>(size, kInvalidCost));
    for (std::size_t row = 0; row < rows.size(); ++row) {
      for (std::size_t column = 0; column < columns.size(); ++column) {
        block[row][column] = cost[rows[row]][columns[column]];
      }
      block[row][columns.size() + row] = kUnmatchedCost;
    }
    for (std::size_t column = 0; column < columns.size(); ++column) {
      block[rows.size() + column][column] = kUnmatchedCost;
      for (std::size_t dummy = 0; dummy < rows.size(); ++dummy) {
        block[rows.size() + column][columns.size() + dummy] = 0.0;
      }
    }

    const std::vector<int> block_assignment = assignRowsHungarian(block);
    for (std::size_t row = 0; row < rows.size(); ++row) {
      const int block_column = block_assignment[row];
      if (block_column >= 0 &&
          block_column < static_cast<int>(columns.size()) &&
          cost[rows[row]][columns[static_cast<std::size_t>(block_column)]] <
              kInvalidCost * 0.5) {
        assignment[rows[row]] = static_cast<int>(
            columns[static_cast<std::size_t>(block_column)]);
      }
    }
  }
  return assignment;
}

}  // namespace

struct MultiObjectTracker::TrackState {
  std::uint64_t id = 0;
  Eigen::Vector4d state = Eigen::Vector4d::Zero();
  Eigen::Matrix4d covariance = Eigen::Matrix4d::Identity();
  Detection object;
  std::uint32_t age = 1;
  std::uint32_t hits = 1;
  std::uint32_t time_since_update = 0;
  std::vector<Eigen::Vector2d> history;
};

MultiObjectTracker::MultiObjectTracker(TrackerConfig config)
    : config_(std::move(config)) {
  if (config_.max_age == 0 || config_.min_hits == 0 ||
      config_.max_history == 0 || config_.process_noise <= 0.0 ||
      config_.position_measurement_noise <= 0.0 ||
      config_.velocity_measurement_noise <= 0.0) {
    throw std::invalid_argument("tracker configuration values must be positive");
  }
  for (const double gate : config_.class_distance_gates) {
    if (gate <= 0.0) {
      throw std::invalid_argument("class distance gates must be positive");
    }
  }
}

MultiObjectTracker::~MultiObjectTracker() = default;

std::vector<TrackOutput> MultiObjectTracker::update(
    const std::vector<Detection>& detections, std::int64_t timestamp_ns) {
  if (timestamp_ns <= 0) {
    throw std::invalid_argument("tracker timestamp must be positive");
  }
  for (const auto& detection : detections) {
    if (detection.label < 0 ||
        detection.label >=
            static_cast<int>(config_.class_distance_gates.size()) ||
        !detection.center.allFinite() || !detection.velocity.allFinite() ||
        !detection.dimensions.allFinite() ||
        !(detection.dimensions.array() > 0.0).all() ||
        !std::isfinite(detection.center_z) || !std::isfinite(detection.yaw) ||
        !std::isfinite(detection.score)) {
      throw std::invalid_argument("tracker received an invalid detection");
    }
  }

  double delta_time = 0.0;
  if (last_timestamp_ns_ != 0) {
    if (timestamp_ns <= last_timestamp_ns_) {
      throw std::invalid_argument("tracker timestamps must increase");
    }
    delta_time = static_cast<double>(timestamp_ns - last_timestamp_ns_) * 1e-9;
    // A bounded time step prevents a paused replay from exploding covariance.
    delta_time = std::clamp(delta_time, 0.01, 1.5);
  }
  last_timestamp_ns_ = timestamp_ns;

  if (delta_time > 0.0) {
    for (auto& track : tracks_) {
      ++track.age;
      ++track.time_since_update;
    }
    if (config_.motion_model == MotionModel::kKalman) {
      Eigen::Matrix4d transition = Eigen::Matrix4d::Identity();
      transition(0, 2) = delta_time;
      transition(1, 3) = delta_time;
      const double dt2 = delta_time * delta_time;
      const double dt3 = dt2 * delta_time;
      const double dt4 = dt2 * dt2;
      Eigen::Matrix4d process = Eigen::Matrix4d::Zero();
      process(0, 0) = dt4 * 0.25;
      process(0, 2) = dt3 * 0.5;
      process(1, 1) = dt4 * 0.25;
      process(1, 3) = dt3 * 0.5;
      process(2, 0) = dt3 * 0.5;
      process(2, 2) = dt2;
      process(3, 1) = dt3 * 0.5;
      process(3, 3) = dt2;
      process *= config_.process_noise;
      for (auto& track : tracks_) {
        track.state = transition * track.state;
        track.covariance =
            transition * track.covariance * transition.transpose() + process;
      }
    }
  }

  std::vector<std::vector<double>> cost(
      tracks_.size(), std::vector<double>(detections.size(), kInvalidCost));
  for (std::size_t track = 0; track < tracks_.size(); ++track) {
    for (std::size_t detection = 0; detection < detections.size(); ++detection) {
      if (tracks_[track].object.label != detections[detection].label) {
        continue;
      }
      const double distance =
          (tracks_[track].state.head<2>() - detections[detection].center).norm();
      const double gate = config_.class_distance_gates.at(
          static_cast<std::size_t>(detections[detection].label));
      if (distance <= gate) {
        cost[track][detection] = distance;
      }
    }
  }
  std::vector<int> assignment;
  if (config_.association_method == AssociationMethod::kGreedy) {
    assignment = assignRowsGreedy(cost);
  } else if (config_.association_method ==
             AssociationMethod::kHungarianClasswise) {
    std::vector<int> track_labels;
    std::vector<int> detection_labels;
    track_labels.reserve(tracks_.size());
    detection_labels.reserve(detections.size());
    for (const auto& track : tracks_) {
      track_labels.push_back(track.object.label);
    }
    for (const auto& detection : detections) {
      detection_labels.push_back(detection.label);
    }
    assignment = assignRowsHungarianClasswise(
        cost, track_labels, detection_labels,
        config_.class_distance_gates.size());
  } else {
    assignment = assignRowsHungarian(cost);
  }
  std::vector<bool> matched_detection(detections.size(), false);

  Eigen::Matrix4d measurement_noise = Eigen::Matrix4d::Zero();
  measurement_noise(0, 0) = config_.position_measurement_noise;
  measurement_noise(1, 1) = config_.position_measurement_noise;
  measurement_noise(2, 2) = config_.velocity_measurement_noise;
  measurement_noise(3, 3) = config_.velocity_measurement_noise;

  for (std::size_t track_index = 0; track_index < tracks_.size();
       ++track_index) {
    TrackState& track = tracks_[track_index];
    const int detection_index =
        track_index < assignment.size() ? assignment[track_index] : -1;
    if (detection_index >= 0 &&
        cost[track_index][static_cast<std::size_t>(detection_index)] <
            kInvalidCost * 0.5) {
      const Detection& detection =
          detections[static_cast<std::size_t>(detection_index)];
      Eigen::Vector4d measurement;
      measurement << detection.center.x(), detection.center.y(),
          detection.velocity.x(), detection.velocity.y();
      if (config_.motion_model == MotionModel::kKalman) {
        const Eigen::Matrix4d innovation_covariance =
            track.covariance + measurement_noise;
        const Eigen::Matrix4d gain =
            innovation_covariance.ldlt().solve(track.covariance).transpose();
        track.state += gain * (measurement - track.state);
        const Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
        const Eigen::Matrix4d residual = identity - gain;
        track.covariance = residual * track.covariance * residual.transpose() +
                           gain * measurement_noise * gain.transpose();
      } else {
        track.state = measurement;
      }
      track.object = detection;
      track.object.center = track.state.head<2>();
      track.object.velocity = track.state.tail<2>();
      track.object.yaw = wrapAngle(track.object.yaw);
      ++track.hits;
      track.time_since_update = 0;
      matched_detection[static_cast<std::size_t>(detection_index)] = true;
    } else {
      track.object.center = track.state.head<2>();
      track.object.velocity = track.state.tail<2>();
    }
    track.history.push_back(track.state.head<2>());
    if (track.history.size() > config_.max_history) {
      track.history.erase(track.history.begin());
    }
  }

  for (std::size_t index = 0; index < detections.size(); ++index) {
    if (matched_detection[index]) {
      continue;
    }
    TrackState track;
    track.id = next_id_++;
    track.state << detections[index].center.x(), detections[index].center.y(),
        detections[index].velocity.x(), detections[index].velocity.y();
    track.covariance = Eigen::Vector4d(2.0, 2.0, 25.0, 25.0).asDiagonal();
    track.object = detections[index];
    track.object.yaw = wrapAngle(track.object.yaw);
    track.history.push_back(detections[index].center);
    tracks_.push_back(std::move(track));
  }

  tracks_.erase(
      std::remove_if(tracks_.begin(), tracks_.end(), [this](const auto& track) {
        return track.time_since_update > config_.max_age;
      }),
      tracks_.end());

  std::vector<TrackOutput> output;
  output.reserve(tracks_.size());
  for (const auto& track : tracks_) {
    output.push_back(TrackOutput{track.id,
                                 track.object,
                                 track.age,
                                 track.hits,
                                 track.time_since_update,
                                 track.hits >= config_.min_hits,
                                 track.history});
  }
  std::sort(output.begin(), output.end(),
            [](const auto& left, const auto& right) {
              return left.id < right.id;
            });
  return output;
}

void MultiObjectTracker::reset() {
  tracks_.clear();
  next_id_ = 1;
  last_timestamp_ns_ = 0;
}

std::size_t MultiObjectTracker::activeTrackCount() const {
  return tracks_.size();
}

}  // namespace centerpoint_ros::tracking
