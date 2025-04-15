from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node
import random
import math


def select_random_start_goal():
    """
    Randomly selects a start and goal position combination and a random orientation.
    
    Returns:
        tuple: Pair of tuples (start, goal) with the selected coordinates
               where start includes random orientation (x, y, theta)
    """
    # Define the possible start and goal position pairs
    start_goal_positions = [
        ((-1.0, -1.0), (-1.0, 0.6)),
        ((-1.0, 0.6), (-1.0, -1.0)),
        ((-0.6, 1.0), (1.0, 1.0)),
        ((1.0, 1.0), (-0.6, 1.0))
    ]
    
    # Define possible orientations in radians (north, east, south, west)
    orientations = {
        "north": 0.0,                # 0 degrees
        "east": math.pi/2,           # 90 degrees
        "south": math.pi,            # 180 degrees
        "west": 3*math.pi/2          # 270 degrees
    }
    
    # Select random position pair and random orientation
    start_pos, goal = random.choice(start_goal_positions)
    orientation_name = random.choice(list(orientations.keys()))
    orientation_rad = orientations[orientation_name]
    
    # Create start tuple with position and orientation
    start = (start_pos[0], start_pos[1], orientation_rad)
    
    # Print selected orientation for information
    print(f"Selected orientation: {orientation_name} ({math.degrees(orientation_rad)} degrees)")
    
    return start, goal


def generate_launch_description():
    world = "project"
    
    # Seleccionar aleatoriamente el start y goal
    start, goal = select_random_start_goal()
    
    # Imprimir la combinación seleccionada para información
    print(f"Utilizando combinación aleatoria de puntos:")
    print(f"Start: {start}")
    print(f"Goal: {goal}")

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
                "particles": 2000, # 2000
                "sigma_v": 0.05,
                "sigma_w": 0.1,
                "sigma_z": 0.2,
                "world": world,
                "use_ekf_when_localized": True, # Activar para mejorar rendimiento
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
                "obstacle_safety_distance": 0.17,
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
