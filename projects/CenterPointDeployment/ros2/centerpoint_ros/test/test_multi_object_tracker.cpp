#include "centerpoint_ros/multi_object_tracker.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tracking = centerpoint_ros::tracking;

tracking::Detection makeCar(double x, double y, double velocity_x) {
  tracking::Detection detection;
  detection.center = {x, y};
  detection.velocity = {velocity_x, 0.0};
  detection.center_z = 1.0;
  detection.dimensions = {4.2, 1.8, 1.6};
  detection.score = 0.9F;
  detection.label = 0;
  return detection;
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void testAssignmentDifference() {
  tracking::TrackerConfig greedy_config;
  greedy_config.min_hits = 1;
  greedy_config.max_age = 1;
  greedy_config.association_method = tracking::AssociationMethod::kGreedy;
  greedy_config.motion_model = tracking::MotionModel::kNone;
  tracking::TrackerConfig hungarian_config = greedy_config;
  hungarian_config.association_method =
      tracking::AssociationMethod::kHungarian;
  tracking::TrackerConfig classwise_config = greedy_config;
  classwise_config.association_method =
      tracking::AssociationMethod::kHungarianClasswise;
  tracking::MultiObjectTracker greedy(greedy_config);
  tracking::MultiObjectTracker hungarian(hungarian_config);
  tracking::MultiObjectTracker classwise(classwise_config);

  const std::int64_t t0 = 10'000'000'000;
  const std::vector<tracking::Detection> initial = {
      makeCar(0.0, 0.0, 0.0), makeCar(2.0, 0.0, 0.0)};
  greedy.update(initial, t0);
  hungarian.update(initial, t0);
  classwise.update(initial, t0);
  const std::vector<tracking::Detection> ambiguous = {
      makeCar(1.1, 0.0, 0.0), makeCar(4.0, 0.0, 0.0)};
  const auto greedy_tracks = greedy.update(ambiguous, t0 + 500'000'000);
  const auto hungarian_tracks =
      hungarian.update(ambiguous, t0 + 500'000'000);
  const auto classwise_tracks =
      classwise.update(ambiguous, t0 + 500'000'000);
  require(greedy_tracks.size() == 2 && hungarian_tracks.size() == 2,
          "both association methods must retain two tracks");
  require(std::abs(greedy_tracks[0].object.center.x() - 4.0) < 1e-9,
          "greedy must select its deterministic local optimum");
  require(std::abs(hungarian_tracks[0].object.center.x() - 1.1) < 1e-9,
          "Hungarian must select the lower global assignment cost");
  require(classwise_tracks.size() == 2 &&
              std::abs(classwise_tracks[0].object.center.x() - 1.1) < 1e-9,
          "classwise Hungarian must preserve the optimal within-class match");
}

int main() {
  try {
    tracking::TrackerConfig config;
    config.max_age = 2;
    config.min_hits = 2;
    tracking::MultiObjectTracker tracker(config);

    const std::int64_t t0 = 1'000'000'000;
    auto tracks = tracker.update({makeCar(0.0, 0.0, 2.0),
                                  makeCar(10.0, 0.0, -1.0)},
                                 t0);
    require(tracks.size() == 2, "first frame must create two tracks");
    require(tracks[0].id == 1 && tracks[1].id == 2,
            "track IDs must be deterministic");
    require(!tracks[0].confirmed && !tracks[1].confirmed,
            "one-hit tracks must remain tentative");

    // Reverse detection order to ensure association does not depend on input order.
    tracks = tracker.update({makeCar(9.5, 0.0, -1.0),
                             makeCar(1.0, 0.0, 2.0)},
                            t0 + 500'000'000);
    require(tracks.size() == 2, "second frame must retain both tracks");
    require(tracks[0].id == 1 && tracks[1].id == 2,
            "association must retain the original IDs");
    require(tracks[0].confirmed && tracks[1].confirmed,
            "two-hit tracks must be confirmed");
    require(tracks[0].object.center.x() < tracks[1].object.center.x(),
            "the two tracks must not swap identities");

    tracks = tracker.update({makeCar(2.0, 0.0, 2.0)},
                            t0 + 1'000'000'000);
    require(tracks.size() == 2, "one missed frame must keep the second track");
    require(tracks[1].time_since_update == 1,
            "missed track age must be observable");

    tracks = tracker.update({makeCar(3.0, 0.0, 2.0)},
                            t0 + 1'500'000'000);
    require(tracks.size() == 2, "max_age missed frames must still be kept");
    tracks = tracker.update({makeCar(4.0, 0.0, 2.0)},
                            t0 + 2'000'000'000);
    require(tracks.size() == 1 && tracks[0].id == 1,
            "track must be removed after max_age is exceeded");

    tracker.reset();
    tracks = tracker.update({makeCar(5.0, 0.0, 0.0)},
                            t0 + 3'000'000'000);
    require(tracks.size() == 1 && tracks[0].id == 1,
            "reset must restart deterministic track IDs");
    testAssignmentDifference();
    std::cout << "multi_object_tracker_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "multi_object_tracker_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
