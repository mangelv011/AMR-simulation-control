# AMR-Sim-Control: Autonomous Mobile Robot Simulation & Control Workspace

## Overview

**AMR-Sim-Control** is a comprehensive ROS 2 workspace for simulating, controlling, and testing Autonomous Mobile Robots (AMRs) in diverse environments. Developed at Comillas ICAI, it integrates state-of-the-art algorithms and tools for robot control, localization, planning, and simulation, supporting both research and educational purposes.

## Repository Structure

The workspace is organized into modular ROS 2 packages:

- **amr_bringup**: Centralized launch scripts and configuration files to start all system components, enabling easy orchestration of simulation and real robot setups.
- **amr_control**: Advanced control algorithms for mobile robots, including trajectory tracking, motion control, and actuator interfaces.
- **amr_localization**: Robust localization algorithms, featuring odometry processing, sensor fusion (e.g., IMU, LIDAR), and state estimation.
- **amr_msgs**: Custom ROS 2 message, service, and action definitions for seamless inter-package communication.
- **amr_planning**: Path and motion planning algorithms for autonomous navigation, obstacle avoidance, and goal management.
- **amr_simulation**: Simulation interfaces and robot models, primarily integrated with CoppeliaSim, supporting realistic physics and sensor emulation.

## Prerequisites

- **ROS 2** (Humble or later recommended)
- **CoppeliaSim** simulation environment (for realistic robot simulation)
- **Python 3.8+**
- Recommended: Ubuntu 22.04 LTS or Windows 10/11

...existing code...

## Development Guidelines

Each package follows the standard ROS 2 structure:
- `include/` – C++ header files
- `src/` – Source code (C++/Python)
- `launch/` – Launch files for starting nodes and simulations
- `config/` – Configuration files (parameters, robot models, etc.)
- `msg/`, `srv/`, `action/` – Custom message, service, and action definitions

**Best Practices:**
- Use ROS 2 node composition for modularity and performance.
- Document code and configuration files for maintainability.
- Write unit and integration tests for new features.
- Follow ROS 2 and Python/C++ style guides.

## License

This project is licensed under the Apache License 2.0. See individual `package.xml` files for details.

## Contributors

- Francisco López-Alvarado
- Eugenio Ribón 
- Miguel Ángel Vallejo

## Acknowledgments

Developed as part of the Autonomous Mobile Robots course at Comillas ICAI University.
Special thanks to contributors and the ROS 2 community for their support and resources.
 
