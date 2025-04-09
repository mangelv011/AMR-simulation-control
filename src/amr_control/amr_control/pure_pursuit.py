import numpy as np



class PurePursuit:
    """Class to follow a path using a simple pure pursuit controller."""

    def __init__(self, dt: float, lookahead_distance: float = 0.5):
        """Pure pursuit class initializer.

        Args:
            dt: Sampling period [s].
            lookahead_distance: Distance to the next target point [m].

        """
        self._dt: float = dt
        self._lookahead_distance: float = lookahead_distance
        self._path: list[tuple[float, float]] = []


        self._aligned = False

    def compute_commands(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """Pure pursuit controller implementation.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].
            theta: Estimated robot heading [rad].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        # TODO: 4.11. Complete the function body with your code (i.e., compute v and w).
        # Initialize velocities to zero to maintain ROS node message flow
        v = 0.0
        w = 0.0
      

        # Check if we have a path to follow
        if not self._path:
            print("Path not found, robot not moving", flush=True)
            return v, w
        
        # Find the closest point on the path to the robot
        closest_point, closest_idx = self._find_closest_point(x, y)
        
        # Find target point at lookahead distance
        target_point = self._find_target_point((x, y), closest_idx)
        
        # Calculate vector from robot to target point in global frame
        dx = target_point[0] - x
        dy = target_point[1] - y
        
        
        # Calculate angle alpha (angle between robot's heading and target)
        alpha = np.arctan2(dy, dx) - theta

        alpha_norm = (alpha + np.pi) % (2 * np.pi) - np.pi # Normalize angle to [-pi, pi]
        angle_threshold = np.pi/6.5  # radians (about 45 degrees)
        # Make the robot rotate in place when the angle error is large
        if abs(alpha_norm) < angle_threshold: 
            self._aligned = True
        if not self._aligned:
            v = 0.0  # Stop forward movement
            w = 1 * np.sign(alpha_norm)  # Rotate in the direction of the error
            print(abs(alpha_norm), flush=True)
            return v, w
            
        
    
        # Pure pursuit formula: w = (2*v*sin(alpha))/(L)
        # Where L is the lookahead distance
        
        # Set a constant velocity for simplicity
        # Can be refined based on distance to goal or path curvature
        base_velocity = 0.15  # m/s
        
        # Slow down when approaching the end of the path
        if closest_idx > len(self._path) - 3:
            v = max(0.05, base_velocity * (1 - 0.8*(closest_idx / len(self._path))))
        else:
            v = base_velocity
        
        # Calculate angular velocity using pure pursuit formula
        # The curvature k = 2*sin(alpha)/L
        # And w = v*k
        w = (2.0 * v * np.sin(alpha)) / self._lookahead_distance
        
        # Limit maximum angular velocity to prevent jerky motion
        # max_angular_velocity = 1.0  # rad/s
        # w = np.clip(w, -max_angular_velocity, max_angular_velocity)
        
        return v, w
            
    
  

    @property
    def path(self) -> list[tuple[float, float]]:
        """Path getter."""
        return self._path

    @path.setter
    def path(self, value: list[tuple[float, float]]) -> None:
        """Path setter."""
        self._path = value

    def _find_closest_point(self, x: float, y: float) -> tuple[tuple[float, float], int]:
        """Find the closest path point to the current robot pose.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].

        Returns:
            tuple[float, float]: (x, y) coordinates of the closest path point [m].
            int: Index of the path point found.

        """
        # TODO: 4.9. Complete the function body (i.e., find closest_xy and closest_idx).
        # Check if path is empty
        if not self._path:
            return (0.0, 0.0), 0
        
        # Initialize variables with the first point
        closest_idx = 0
        closest_xy = self._path[0]
        min_distance = float('inf')
        
        # Iterate through all path points to find the closest one
        for i, point in enumerate(self._path):
            # Calculate squared Euclidean distance (more efficient than using sqrt)
            distance = (point[0] - x)**2 + (point[1] - y)**2
            
            # Update if this point is closer
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
                closest_xy = point
        
        return closest_xy, closest_idx
    def _find_target_point(
        self, origin_xy: tuple[float, float], origin_idx: int
    ) -> tuple[float, float]:
        """Find the destination path point based on the lookahead distance.

        Args:
            origin_xy: Current location of the robot (x, y) [m].
            origin_idx: Index of the current path point.

        Returns:
            tuple[float, float]: (x, y) coordinates of the target point [m].

        """
        # TODO: 4.10. Complete the function body with your code (i.e., determine target_xy).
        # Check if path is empty or we're at the end of the path
        if not self._path or origin_idx >= len(self._path) - 1:
            # Return last point if available, otherwise origin
            return self._path[-1] if self._path else origin_xy
        
        # Start looking from the closest point forward
        current_idx = origin_idx
        
        # Get robot's current position
        robot_x, robot_y = origin_xy
        
        # Keep track of accumulated distance along the path
        accumulated_distance = 0.0
        
        # Look for a point that's at least lookahead_distance away
        while current_idx < len(self._path) - 1:
            # Get current and next points on the path
            current_point = self._path[current_idx]
            next_point = self._path[current_idx + 1]
            

            # Calculate segment length
            segment_length = np.sqrt((next_point[0] - current_point[0])**2 + 
                                    (next_point[1] - current_point[1])**2)
            
            # Check if lookahead point is on this segment
            if accumulated_distance + segment_length >= self._lookahead_distance:
                # Calculate how far along this segment the lookahead point is
                remaining_distance = self._lookahead_distance - accumulated_distance
                ratio = remaining_distance / segment_length
                
                # Interpolate to find the exact lookahead point
                target_x = current_point[0] + ratio * (next_point[0] - current_point[0])
                target_y = current_point[1] + ratio * (next_point[1] - current_point[1])
                
                return (target_x, target_y)
            
            # If not on this segment, add segment length and move to next segment
            accumulated_distance += segment_length
            current_idx += 1
        
        # If we reach here, return the last point in the path
        return self._path[-1]
