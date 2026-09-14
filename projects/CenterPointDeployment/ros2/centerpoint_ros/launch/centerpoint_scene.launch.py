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
        [FindPackageShare("centerpoint_ros"), "rviz", "centerpoint.rviz"]
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
                        "score_threshold": 0.1,
                        "marker_score_threshold": 0.2,
                        "reliable_point_qos": True,
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
                        "publish_rate_hz": 2.0,
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
