import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    repository_root = LaunchConfiguration("repository_root")
    scene_name = LaunchConfiguration("scene_name")
    use_rviz = LaunchConfiguration("use_rviz")
    loop = LaunchConfiguration("loop")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")
    result_path = LaunchConfiguration("result_path")
    minimum_score = LaunchConfiguration("minimum_score")
    association_method = LaunchConfiguration("association_method")
    motion_model = LaunchConfiguration("motion_model")
    parameter_directory = PathJoinSubstitution(
        [repository_root, "work_dirs", "centerpoint_deployment", "preprocess"]
    )
    engine_path = PathJoinSubstitution(
        [
            repository_root,
            "work_dirs",
            "centerpoint_deployment",
            "tensorrt",
            "centerpoint_dense_mixed_fp16_deep.engine",
        ]
    )
    sequence_manifest = PathJoinSubstitution(
        [
            repository_root,
            "work_dirs",
            "centerpoint_deployment",
            "ros2-scenes",
            scene_name,
            "frames.csv",
        ]
    )
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("centerpoint_ros"), "rviz", "centerpoint_tracking.rviz"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "repository_root",
                default_value=os.environ.get("MMDET3D_ROOT", os.getcwd()),
                description="Absolute MMDetection3D repository path",
            ),
            DeclareLaunchArgument("scene_name", default_value="scene-0103"),
            DeclareLaunchArgument("loop", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("publish_rate_hz", default_value="2.0"),
            DeclareLaunchArgument("minimum_score", default_value="0.2"),
            DeclareLaunchArgument(
                "association_method",
                default_value="hungarian_classwise",
                description="greedy, hungarian, or hungarian_classwise",
            ),
            DeclareLaunchArgument("motion_model", default_value="kalman"),
            DeclareLaunchArgument(
                "result_path",
                default_value="",
                description="Optional nuScenes tracking JSON output path",
            ),
            Node(
                package="centerpoint_ros",
                executable="centerpoint_node",
                name="centerpoint_node",
                output="screen",
                parameters=[
                    {
                        "parameter_directory": parameter_directory,
                        "engine_path": engine_path,
                        "point_topic": "/lidar_points",
                        "detection_topic": "/centerpoint/detections",
                        "score_threshold": 0.1,
                        "marker_score_threshold": 0.2,
                        "reliable_point_qos": True,
                    }
                ],
            ),
            Node(
                package="centerpoint_ros",
                executable="centerpoint_tracker_node",
                name="centerpoint_tracker_node",
                output="screen",
                parameters=[
                    {
                        "detection_topic": "/centerpoint/detections",
                        "metadata_topic": "/centerpoint/frame_metadata",
                        "track_topic": "/centerpoint/tracked_objects",
                        "marker_topic": "/centerpoint/track_markers",
                        "minimum_score": minimum_score,
                        "max_age": 3,
                        "min_hits": 2,
                        "max_history": 20,
                        "publish_tentative": False,
                        "association_method": association_method,
                        "motion_model": motion_model,
                        "result_path": result_path,
                        "class_distance_gates": [
                            4.0,
                            4.0,
                            3.0,
                            5.5,
                            3.0,
                            1.5,
                            4.0,
                            2.0,
                            1.0,
                            1.0,
                        ],
                    }
                ],
            ),
            Node(
                package="centerpoint_ros",
                executable="pointcloud_replay_node",
                name="pointcloud_replay_node",
                output="screen",
                parameters=[
                    {
                        "sequence_manifest": sequence_manifest,
                        "scene_name": scene_name,
                        "point_topic": "/lidar_points",
                        "metadata_topic": "/centerpoint/frame_metadata",
                        "frame_id": "lidar",
                        "global_frame": "map",
                        "publish_rate_hz": publish_rate_hz,
                        "startup_delay_s": 2.0,
                        "loop": loop,
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                additional_env={
                    "QT_QPA_PLATFORM": "xcb",
                    "LIBGL_ALWAYS_SOFTWARE": "1",
                },
                output="screen",
                condition=IfCondition(use_rviz),
            ),
        ]
    )
