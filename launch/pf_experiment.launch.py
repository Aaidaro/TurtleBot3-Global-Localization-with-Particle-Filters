from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch_ros.actions import Node


def as_bool(value):
    return str(value).lower() in ["true", "1", "yes", "y"]


def launch_setup(context, *args, **kwargs):
    def g(name):
        return context.launch_configurations[name]

    mode = g("mode")

    map_yaml_path = g("map_yaml_path")
    truth_x_offset = float(g("truth_x_offset"))
    truth_y_offset = float(g("truth_y_offset"))
    truth_yaw_offset = float(g("truth_yaw_offset"))

    n_particles = int(g("n_particles"))
    beam_step = int(g("beam_step"))
    max_beams = int(g("max_beams"))

    kidnap_detection_enabled = as_bool(g("kidnap_detection_enabled"))

    common_pf_params = {
        "map_yaml_path": map_yaml_path,
        "use_odom_as_truth": False,
        "true_odom_topic": "/ground_truth_odom",
        "truth_x_offset": truth_x_offset,
        "truth_y_offset": truth_y_offset,
        "truth_yaw_offset": truth_yaw_offset,
        "odom_topic": "/odom",
        "scan_topic": "/scan",
        "n_particles": n_particles,
        "beam_step": beam_step,
        "max_beams": max_beams,
        "debug_require_truth_error_for_convergence": True,
        "kidnap_detection_enabled": kidnap_detection_enabled,
        "kidnap_required_bad_scans": int(g("kidnap_required_bad_scans")),
        "kidnap_log_likelihood_drop": float(g("kidnap_log_likelihood_drop")),
        "kidnap_absolute_log_likelihood": float(g("kidnap_absolute_log_likelihood")),
        "kidnap_min_range_gate_m": float(g("kidnap_min_range_gate_m")),
        "kidnap_arm_after_converged_scans": int(g("kidnap_arm_after_converged_scans")),
    }

    if mode == "basic":
        pf_executable = "basic_pf_localization"
        converged_topic = "/basic_pf/converged"
        pf_params = {
            **common_pf_params,
            "resample_period": int(g("resample_period")),
        }
    elif mode == "adaptive":
        pf_executable = "pf_localization"
        converged_topic = "/pf/converged"
        pf_params = {
            **common_pf_params,
            "resample_neff_fraction": float(g("resample_neff_fraction")),
        }
    elif mode == "mcmc":
        pf_executable = "mcmc_pf_localization"
        converged_topic = "/mcmc_pf/converged"
        pf_params = {
            **common_pf_params,
            "resample_neff_fraction": float(g("resample_neff_fraction")),
            "mcmc_enabled": True,
            "mcmc_iterations": int(g("mcmc_iterations")),
            "mcmc_proposal_xy_std": float(g("mcmc_proposal_xy_std")),
            "mcmc_proposal_yaw_std": float(g("mcmc_proposal_yaw_std")),
            "mcmc_likelihood_scale": float(g("mcmc_likelihood_scale")),
        }
    else:
        raise RuntimeError("mode must be one of: basic, adaptive, mcmc")

    actions = []

    if as_bool(g("run_static_tf")):
        actions.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_map_to_odom",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "0",
                    "--roll", "0", "--pitch", "0", "--yaw", "0",
                    "--frame-id", "map",
                    "--child-frame-id", "odom",
                ],
                output="screen",
            )
        )

    if as_bool(g("randomize")):
        actions.append(
            Node(
                package="lidar_analysis_py",
                executable="randomize_robot_pose",
                name="randomize_robot_pose",
                parameters=[{
                    "map_yaml_path": map_yaml_path,
                    "model_name": g("model_name"),
                    "delete_service": "/delete_entity",
                    "spawn_service": "/spawn_entity",
                    "reference_frame": "world",
                    "truth_x_offset": truth_x_offset,
                    "truth_y_offset": truth_y_offset,
                    "truth_yaw_offset": truth_yaw_offset,
                    "min_clearance_m": float(g("min_clearance_m")),
                }],
                output="screen",
            )
        )

    pf_delay = float(g("pf_start_delay")) if as_bool(g("randomize")) else 0.0
    motion_delay = pf_delay + float(g("motion_start_delay"))

    pf_node = Node(
        package="lidar_analysis_py",
        executable=pf_executable,
        name=pf_executable,
        parameters=[pf_params],
        output="screen",
    )

    actions.append(
        TimerAction(
            period=pf_delay,
            actions=[pf_node],
        )
    )

    if as_bool(g("run_motion")):
        motion_node = Node(
            package="lidar_analysis_py",
            executable="pf_explore_motion",
            name="pf_explore_motion",
            parameters=[{
                "converged_topic": converged_topic,
                "linear_speed": float(g("linear_speed")),
                "angular_speed": float(g("angular_speed")),
                "front_clearance_m": float(g("front_clearance_m")),
            }],
            output="screen",
        )

        actions.append(
            TimerAction(
                period=motion_delay,
                actions=[motion_node],
            )
        )

    if as_bool(g("run_rviz")):
        actions.append(
            ExecuteProcess(
                cmd=["rviz2"],
                output="screen",
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="adaptive",
                              description="basic, adaptive, or mcmc"),

        DeclareLaunchArgument("map_yaml_path", default_value="/root/tb3_projects_ws/maps/explore_house.yaml"),

        DeclareLaunchArgument("randomize", default_value="true"),
        DeclareLaunchArgument("run_static_tf", default_value="true"),
        DeclareLaunchArgument("run_motion", default_value="true"),
        DeclareLaunchArgument("run_rviz", default_value="true"),

        DeclareLaunchArgument("pf_start_delay", default_value="3.0"),
        DeclareLaunchArgument("motion_start_delay", default_value="2.0"),

        DeclareLaunchArgument("model_name", default_value="burger"),
        DeclareLaunchArgument("min_clearance_m", default_value="0.35"),

        DeclareLaunchArgument("truth_x_offset", default_value="2.229376"),
        DeclareLaunchArgument("truth_y_offset", default_value="0.415668"),
        DeclareLaunchArgument("truth_yaw_offset", default_value="-0.016549"),

        DeclareLaunchArgument("n_particles", default_value="2000"),
        DeclareLaunchArgument("beam_step", default_value="8"),
        DeclareLaunchArgument("max_beams", default_value="150"),

        DeclareLaunchArgument("resample_period", default_value="5"),
        DeclareLaunchArgument("resample_neff_fraction", default_value="0.5"),

        DeclareLaunchArgument("mcmc_iterations", default_value="2"),
        DeclareLaunchArgument("mcmc_proposal_xy_std", default_value="0.04"),
        DeclareLaunchArgument("mcmc_proposal_yaw_std", default_value="0.08"),
        DeclareLaunchArgument("mcmc_likelihood_scale", default_value="10.0"),

        DeclareLaunchArgument("kidnap_detection_enabled", default_value="false"),
        DeclareLaunchArgument("kidnap_required_bad_scans", default_value="10"),
        DeclareLaunchArgument("kidnap_log_likelihood_drop", default_value="2.50"),
        DeclareLaunchArgument("kidnap_absolute_log_likelihood", default_value="-5.00"),
        DeclareLaunchArgument("kidnap_min_range_gate_m", default_value="0.22"),
        DeclareLaunchArgument("kidnap_arm_after_converged_scans", default_value="30"),

        DeclareLaunchArgument("linear_speed", default_value="0.08"),
        DeclareLaunchArgument("angular_speed", default_value="0.55"),
        DeclareLaunchArgument("front_clearance_m", default_value="0.45"),

        OpaqueFunction(function=launch_setup),
    ])
