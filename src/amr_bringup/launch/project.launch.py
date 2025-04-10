from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node

import math


def generate_launch_description():
    world = "project"
    start = (1, -1, math.radians(90))
    goal = (0.2, -0.6)
    

    particle_filter_node = LifecycleNode(
        package="amr_localization",
        executable="particle_filter",
        name="particle_filter",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "enable_plot": True,
                "global_localization": True,
                "particles": 500, # 2000
                "sigma_v": 0.05,
                "sigma_w": 0.1,
                "sigma_z": 0.2,
                "world": world,
            }
        ],
    )

    probabilistic_roadmap_node = LifecycleNode(
        package="amr_planning",
        executable="probabilistic_roadmap",
        name="probabilistic_roadmap",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "connection_distance": 0.15,
                "enable_plot": True,
                "goal": goal,
                "grid_size": 0.1,
                "node_count": 300,
                "obstacle_safety_distance": 0.15,
                "smoothing_additional_points": 1,
                "smoothing_data_weight": 0.1,
                "smoothing_smooth_weight": 0.1,
                "use_grid": True,
                "world": world,
            }
        ],
    )

    wall_follower_node = LifecycleNode(
        package="amr_control",
        executable="wall_follower",
        name="wall_follower",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[{"enable_localization": True}],
    )

    pure_pursuit_node = LifecycleNode(
        package="amr_control",
        executable="pure_pursuit",
        name="pure_pursuit",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[{"lookahead_distance": 0.3}],
    )

    coppeliasim_node = LifecycleNode(
        package="amr_simulation",
        executable="coppeliasim",
        name="coppeliasim",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "enable_localization": True,
                "goal": goal,
                "goal_tolerance": 0.1,
                "start": start,
            }
        ],
    )

    lifecycle_manager_node = Node(
        package="amr_bringup",
        executable="lifecycle_manager",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "node_startup_order": (
                    "particle_filter",
                    "probabilistic_roadmap",
                    "wall_follower",
                    "pure_pursuit",
                    "coppeliasim",  # Must be started last
                )
            }
        ],
    )

    return LaunchDescription(
        [
            particle_filter_node,
            probabilistic_roadmap_node,
            wall_follower_node,
            pure_pursuit_node,
            coppeliasim_node,
            lifecycle_manager_node,  # Must be launched last
        ]
    )
