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
        # Handle edge cases
        if not self._path:
            return origin_xy
        if origin_idx >= len(self._path) - 1:
            return self._path[-1]
        
        # Simple approach: Find the first point that's approximately at lookahead distance
        robot_x, robot_y = origin_xy
        
        for i in range(origin_idx, len(self._path)):
            point = self._path[i]
            # Calculate direct distance from robot to this point
            distance = np.sqrt((point[0] - robot_x)**2 + (point[1] - robot_y)**2)
            
            # If we found a point at approximately the lookahead distance, return it
            if distance >= self._lookahead_distance:
                return point
        
        # If no suitable point is found, return the last point in the path
        return self._path[-1]
