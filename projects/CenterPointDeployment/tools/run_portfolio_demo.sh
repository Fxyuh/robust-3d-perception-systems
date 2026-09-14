#!/usr/bin/env bash

set -eo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(readlink -f "${script_directory}/../../..")"
scene_name="${1:-scene-0103}"
install_setup="${repository_root}/work_dirs/centerpoint_deployment/ros2-install/setup.bash"
scene_manifest="${repository_root}/work_dirs/centerpoint_deployment/ros2-scenes/${scene_name}/frames.csv"
engine_path="${repository_root}/work_dirs/centerpoint_deployment/tensorrt/centerpoint_dense_mixed_fp16_deep.engine"

for required_file in "${install_setup}" "${scene_manifest}" "${engine_path}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required demo artifact: ${required_file}" >&2
    exit 2
  fi
done

source /opt/ros/humble/setup.bash
source "${install_setup}"

set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-61}"

echo "Starting ${scene_name} portfolio demo on ROS_DOMAIN_ID=${ROS_DOMAIN_ID}."
echo "The 40-frame sequence repeats every 20 seconds; press Ctrl+C to stop."

exec env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 \
  ros2 launch centerpoint_ros centerpoint_tracking.launch.py \
  repository_root:="${repository_root}" \
  scene_name:="${scene_name}" \
  loop:=true \
  use_rviz:=true \
  publish_rate_hz:=2.0 \
  minimum_score:=0.2 \
  association_method:=hungarian_classwise \
  motion_model:=kalman
