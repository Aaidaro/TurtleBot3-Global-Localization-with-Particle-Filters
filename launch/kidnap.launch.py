from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("map_yaml_path", default_value="/root/tb3_projects_ws/maps/explore_house.yaml"),
        DeclareLaunchArgument("model_name", default_value="burger"),
        DeclareLaunchArgument("truth_x_offset", default_value="2.229376"),
        DeclareLaunchArgument("truth_y_offset", default_value="0.415668"),
        DeclareLaunchArgument("truth_yaw_offset", default_value="-0.016549"),
        DeclareLaunchArgument("min_clearance_m", default_value="0.35"),

        Node(
            package="lidar_analysis_py",
            executable="randomize_robot_pose",
            name="kidnap_randomize_robot_pose",
            parameters=[{
                "map_yaml_path": "/root/tb3_projects_ws/maps/explore_house.yaml",
                "model_name": "burger",
                "delete_service": "/delete_entity",
                "spawn_service": "/spawn_entity",
                "reference_frame": "world",
                "truth_x_offset": 2.229376,
                "truth_y_offset": 0.415668,
                "truth_yaw_offset": -0.016549,
                "min_clearance_m": 0.35,
            }],
            output="screen",
        )
    ])
