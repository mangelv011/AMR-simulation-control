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
        
        # Added: Precalculate constant values
        self._lookahead_squared = lookahead_distance ** 2
        self._angle_threshold = np.pi/4.0  # Modified: More permissive threshold (45 degrees)
        self._base_velocity = 0.4  # m/s  # Modified: Higher base velocity
        self._min_velocity = 0.15  # Added: Higher minimum velocity
        
        # Added: Cache for closest points
        self._last_closest_idx = 0
        
        # Added: Factors for velocity smoothing
        self._max_angular_vel = 1.5  # Maximum angular velocity
        self._slowdown_factor = 0.5  # Reduction factor at the end of the path

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
        
        # OPTIMIZED: Calculate differences once and reuse them
        dx = target_point[0] - x
        dy = target_point[1] - y
        
        # OPTIMIZED: Use arctan2 directly
        alpha = np.arctan2(dy, dx) - theta
        alpha_norm = (alpha + np.pi) % (2 * np.pi) - np.pi  # Normalize angle to [-pi, pi]
        
        # Make the robot rotate in place when the angle error is large
        if abs(alpha_norm) < self._angle_threshold: 
            self._aligned = True
        if not self._aligned:
            # MODIFIED: Faster rotation to align
            return 0.0, np.sign(alpha_norm) * self._max_angular_vel
            
        # MODIFIED: Improve linear velocity calculation
        path_len = len(self._path)
        # Calculate path progress factor (0-1)
        progress = min(1.0, max(0.0, closest_idx / max(1, path_len - 1)))
        
        # Determine velocity based on:
        # 1. Progress along the path (reduce speed at the end)
        # 2. Deviation angle (reduce speed in curves)
        # 3. Base velocity
        
        if closest_idx > path_len - 5:  # Last 5 points of the path
            # Gradual reduction at the end of the path
            v = max(self._min_velocity, self._base_velocity * (1.0 - progress * self._slowdown_factor))
        else:
            # Calculate reduction factor based on angle (1.0 in straight line, less in curves)
            angle_factor = max(0.7, 1.0 - abs(alpha_norm) / np.pi)
            v = self._base_velocity * angle_factor
        
        # MODIFIED: Calculate angular velocity with smoother limitation
        # Calculate w based on the pure pursuit formula but limit for smoother movements
        raw_w = 2.0 * v * np.sin(alpha_norm) / self._lookahead_distance
        
        # Limit w to avoid abrupt changes that would reduce speed
        w = np.clip(raw_w, -self._max_angular_vel, self._max_angular_vel)
        
        return v, w

    @property
    def path(self) -> list[tuple[float, float]]:
        """Path getter."""
        return self._path

    @path.setter
    def path(self, value: list[tuple[float, float]]) -> None:
        """Path setter."""
        self._path = value
        # Added: Reset cache index when changing the path
        self._last_closest_idx = 0
        self._aligned = False

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
        
        # OPTIMIZED: Search from the last known closest point
        start_idx = max(0, self._last_closest_idx - 2)
        end_idx = min(len(self._path), self._last_closest_idx + 10)
        
        # Initialize variables with the first point to check
        closest_idx = start_idx
        closest_xy = self._path[start_idx]
        min_distance = (self._path[start_idx][0] - x)**2 + (self._path[start_idx][1] - y)**2
        
        # OPTIMIZED: Traverse only a window of probable points
        for i in range(start_idx, end_idx):
            point = self._path[i]
            # OPTIMIZED: Use squared distance instead of square root
            distance = (point[0] - x)**2 + (point[1] - y)**2
            
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
                closest_xy = point
        
        # OPTIMIZED: If the closest point is at the edge of our window,
        #            expand the search to ensure we find the global minimum
        if closest_idx == end_idx - 1 and end_idx < len(self._path):
            # Search the rest of the path
            for i in range(end_idx, len(self._path)):
                point = self._path[i]
                distance = (point[0] - x)**2 + (point[1] - y)**2
                
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = i
                    closest_xy = point
                else:
                    # If distance starts increasing, we can stop
                    break
        
        # Added: Update cache with the newly found index
        self._last_closest_idx = closest_idx
        
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
        path_len = len(self._path)
        if not self._path or origin_idx >= path_len - 1:
            # Return last point if available, otherwise origin
            return self._path[-1] if self._path else origin_xy
        
        # OPTIMIZED: Start from the closest point + 1 to avoid searching backwards
        current_idx = max(origin_idx, 0)
        accumulated_distance = 0.0
        
        # OPTIMIZED: Precalculate values to avoid repeated calculations
        lookahead_distance = self._lookahead_distance
        
        # Look for a point that's at least lookahead_distance away
        while current_idx < path_len - 1:
            current_point = self._path[current_idx]
            next_point = self._path[current_idx + 1]
            
            # OPTIMIZED: Calculate distance using dx and dy only once
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            segment_length = np.sqrt(dx*dx + dy*dy)
            
            # Check if lookahead point is on this segment
            if accumulated_distance + segment_length >= lookahead_distance:
                # OPTIMIZED: More direct linear interpolation
                remaining_distance = lookahead_distance - accumulated_distance
                ratio = remaining_distance / segment_length
                
                # OPTIMIZED: More efficient vector calculation
                return (current_point[0] + ratio * dx, 
                        current_point[1] + ratio * dy)
            
            accumulated_distance += segment_length
            current_idx += 1
        
        # If we reach here, return the last point in the path
        return self._path[-1]
