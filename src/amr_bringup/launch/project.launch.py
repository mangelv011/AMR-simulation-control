from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node
import random
import math

def generate_launch_description():
    """
    Generate the launch description with all necessary nodes for the AMR system.
    
    Returns:
        LaunchDescription: Complete launch description for the project
    """
    # Define the world to be used
    world = "project"
    
    # Select start and goal positions
    start = (1.0, -1.0, math.pi)
    goal = (0.2, -0.6)
    
    # Print the selected combination for information
    print(f"Using random combination of points:")
    print(f"Start: {start}")
    print(f"Goal: {goal}")

    # Initialize particle filter node for localization
    particle_filter_node = LifecycleNode(
        package="amr_localization",
        executable="particle_filter",
        name="particle_filter",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "enable_plot": False,
                "global_localization": True,
                "particles": 2000,  # 2000 particles for robust localization
                "sigma_v": 0.05,
                "sigma_w": 0.1,
                "sigma_z": 0.2,
                "world": world,
                "use_ekf_when_localized": True,  # Enable for improved performance
            }
        ],
    )

    # Initialize probabilistic roadmap node for path planning
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
                "obstacle_safety_distance": 0.17,
                "smoothing_additional_points": 1,
                "smoothing_data_weight": 0.1,
                "smoothing_smooth_weight": 0.1,
                "use_grid": True,
                "world": world,
            }
        ],
    )

    # Initialize wall follower node for obstacle avoidance
    wall_follower_node = LifecycleNode(
        package="amr_control",
        executable="wall_follower",
        name="wall_follower",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[{"enable_localization": True}],
    )

    # Initialize pure pursuit node for path following
    pure_pursuit_node = LifecycleNode(
        package="amr_control",
        executable="pure_pursuit",
        name="pure_pursuit",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[{"lookahead_distance": 0.3}],
    )

    # Initialize simulation node for CoppeliaSim
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

    # Initialize lifecycle manager node to control node state transitions
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

    # Return the complete launch description with all nodes
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
