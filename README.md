# AMR Simulation Workspace

## Overview

This repository contains a ROS 2 workspace for Autonomous Mobile Robots (AMR) simulation and control, developed at Comillas ICAI. It provides a complete framework for simulating, controlling, and testing mobile robots in various environments.

## Repository Structure

The workspace is organized into the following ROS 2 packages:

- **amr_bringup**: Launch scripts and configuration files to start all necessary components of the system.
- **amr_control**: Control algorithms for mobile robots, including trajectory tracking and motion control.
- **amr_localization**: Algorithms for robot localization, including odometry processing and sensor fusion.
- **amr_msgs**: Custom message, service, and action definitions used across the workspace.
- **amr_planning**: Path and motion planning algorithms for autonomous navigation.
- **amr_simulation**: Simulation interfaces and models, primarily integrated with CoppeliaSim.

## Prerequisites

- ROS 2 (Humble or later recommended)
- CoppeliaSim simulation environment
- Python 3.8+

## Installation

Clone this repository into your workspace directory:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/your-username/sim_ws.git
cd ..
colcon build
source install/setup.bash  # or setup.ps1 on Windows
```

## Usage

To launch the basic simulation environment:

```bash
ros2 launch amr_bringup simulation.launch.py
```

For specific robot models (e.g., TurtleBot3 Burger):

```bash
ros2 launch amr_bringup robot_simulation.launch.py robot_type:=turtlebot3_burger
```

## Development

Each package follows the standard ROS 2 package structure:
- `include/` - C++ header files
- `src/` - Source code files
- `launch/` - Launch files
- `config/` - Configuration files
- `msg/`, `srv/`, `action/` - Message, service, and action definitions

## License

This project is licensed under the Apache License 2.0 - see individual package.xml files for details.

## Acknowledgments

Developed as part of the Autonomous Mobile Robots course at Comillas ICAI University.
 
