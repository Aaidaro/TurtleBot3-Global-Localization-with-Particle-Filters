from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction


def launch_setup(context, *args, **kwargs):
    mode = context.launch_configurations["mode"]

    if mode == "basic":
        log_dir = "/root/tb3_projects_ws/log/pf_basic_runs"
        out_dir = "/root/tb3_projects_ws/log/pf_basic_plots"
    elif mode == "adaptive":
        log_dir = "/root/tb3_projects_ws/log/pf_runs"
        out_dir = "/root/tb3_projects_ws/log/pf_plots"
    elif mode == "mcmc":
        log_dir = "/root/tb3_projects_ws/log/pf_mcmc_runs"
        out_dir = "/root/tb3_projects_ws/log/pf_mcmc_plots"
    else:
        raise RuntimeError("mode must be one of: basic, adaptive, mcmc")

    return [
        ExecuteProcess(
            cmd=[
                "ros2", "run", "lidar_analysis_py", "pf_error_plotter",
                "--",
                "--latest",
                "--log-dir", log_dir,
                "--out-dir", out_dir,
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="adaptive",
                              description="basic, adaptive, or mcmc"),
        OpaqueFunction(function=launch_setup),
    ])
