import numpy as np
import math
from amr_localization.maps import Map

class ExtendedKalmanFilter:
    """
    Implementation of the Extended Kalman Filter (EKF) for robot localization.
    Used as a substitute for the particle filter once the robot is localized.
    """

    def __init__(
        self,
        dt: float,
        map_path: str,
        initial_pose: tuple[float, float, float],
        sigma_v: float = 0.05,
        sigma_w: float = 0.1,
        sigma_z: float = 0.2,
        sensor_range_max: float = 8.0,
        sensor_range_min: float = 0.16,
    ):
        """
        Initializes the extended Kalman filter.
        
        Args:
            dt: Sampling period [s].
            map_path: Path to the environment map.
            initial_pose: Initial pose of the robot (x, y, theta) in [m, m, rad].
            sigma_v: Standard deviation of linear velocity [m/s].
            sigma_w: Standard deviation of angular velocity [rad/s].
            sigma_z: Standard deviation of measurements [m].
            sensor_range_max: Maximum sensor range [m].
            sensor_range_min: Minimum sensor range [m].
        """
        self._dt = dt
        self._sigma_v = sigma_v
        self._sigma_w = sigma_w
        self._sigma_z = sigma_z
        self._sensor_range_max = sensor_range_max
        self._sensor_range_min = sensor_range_min
        self._interval = 20  # Same interval as the particle filter

        # Initialize the map
        self._map = Map(
            map_path,
            sensor_range_max,
            compiled_intersect=True,
            use_regions=False,
            safety_distance=0.08,
        )
        
        # Initial state [x, y, theta]
        self._mu = np.array([initial_pose[0], initial_pose[1], initial_pose[2]]).reshape(3, 1)
        
        # Initial covariance matrix (initial uncertainty)
        # Small values because we assume the initial pose is accurate
        self._sigma = np.diag([0.01, 0.01, 0.01])  
        
        # Process noise matrices for the motion model
        self._R = np.diag([self._sigma_v**2, self._sigma_w**2])
        
        # Iteration counter
        self._iteration = 0

    def predict(self, v: float, w: float) -> None:
        """
        EKF prediction update based on the motion model.
        
        Args:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].
        """
        self._iteration += 1
        
        # Current state
        x = self._mu[0, 0]
        y = self._mu[1, 0]
        theta = self._mu[2, 0]
        
        # If angular velocity is approximately zero, use linear model
        if abs(w) < 1e-6:
            # Linear model: the robot moves in a straight line
            x_new = x + v * self._dt * math.cos(theta)
            y_new = y + v * self._dt * math.sin(theta)
            theta_new = theta
            
            # Jacobian of the motion model with respect to the state
            G = np.array([
                [1.0, 0.0, -v * self._dt * math.sin(theta)],
                [0.0, 1.0, v * self._dt * math.cos(theta)],
                [0.0, 0.0, 1.0]
            ])
            
            # Jacobian of the motion model with respect to noise
            V = np.array([
                [self._dt * math.cos(theta), 0],
                [self._dt * math.sin(theta), 0],
                [0, 0]
            ])
        else:
            # Non-linear model: the robot turns and moves forward
            v_w_ratio = v / w
            theta_new = (theta + w * self._dt) % (2 * math.pi)
            x_new = x + v_w_ratio * (math.sin(theta_new) - math.sin(theta))
            y_new = y - v_w_ratio * (math.cos(theta_new) - math.cos(theta))
            
            # Jacobian of the motion model with respect to the state
            G = np.array([
                [1.0, 0.0, v_w_ratio * (math.cos(theta_new) - math.cos(theta))],
                [0.0, 1.0, v_w_ratio * (math.sin(theta_new) - math.sin(theta))],
                [0.0, 0.0, 1.0]
            ])
            
            # Jacobian of the motion model with respect to noise
            V = np.array([
                [(math.sin(theta_new) - math.sin(theta))/w, -v/(w**2)*(math.sin(theta_new) - math.sin(theta)) + v/w*self._dt*math.cos(theta_new)],
                [-(math.cos(theta_new) - math.cos(theta))/w, v/(w**2)*(math.cos(theta_new) - math.cos(theta)) + v/w*self._dt*math.sin(theta_new)],
                [0, self._dt]
            ])
        
        # Check for collision with the map and adjust if necessary
        segment = [(x, y), (x_new, y_new)]
        intersection_point, _ = self._map.check_collision(segment)
        
        if intersection_point:
            x_new, y_new = intersection_point
        
        # Update the state
        self._mu = np.array([x_new, y_new, theta_new]).reshape(3, 1)
        
        # Update the covariance
        self._sigma = G @ self._sigma @ G.T + V @ self._R @ V.T

    def update(self, measurements: list[float]) -> None:
        """
        EKF correction update based on sensor measurements.
        
        Args:
            measurements: LiDAR sensor measurements [m].
        """
        # Process measurements to use only those corresponding to the interval
        processed_measurements = np.array(measurements[0:240:self._interval])
        
        # Get the current pose
        x = self._mu[0, 0]
        y = self._mu[1, 0]
        theta = self._mu[2, 0]
        pose = (x, y, theta)
        
        # Get predicted measurements based on the current pose and map
        predicted_measurements = self._sense(pose)
        
        # For each measurement, perform an EKF update
        for i, (z_measured, z_predicted) in enumerate(zip(processed_measurements, predicted_measurements)):
            # Ignore invalid measurements (out of range or collision)
            if np.isnan(z_measured) or np.isnan(z_predicted):
                continue
                
            # Calculate the ray angle for this index
            ray_angle = math.radians(1.5 * i * self._interval)
            ray_angle_global = theta + ray_angle
            
            # Calculate the endpoint of the ray (coordinates of the detected obstacle)
            ray_endpoint_x = x + z_predicted * math.cos(ray_angle_global)
            ray_endpoint_y = y + z_predicted * math.sin(ray_angle_global)
            
            # Calculate the Jacobian of the measurement (derivative with respect to the state)
            # The measurement is the distance between the robot and the obstacle
            dx = ray_endpoint_x - x
            dy = ray_endpoint_y - y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 1e-6:  # Avoid division by zero
                continue
                
            H = np.zeros((1, 3))
            H[0, 0] = -dx / distance  # dx/dx
            H[0, 1] = -dy / distance  # dx/dy
            H[0, 2] = z_predicted * (-dx * math.sin(ray_angle_global) + dy * math.cos(ray_angle_global)) / distance
            
            # Measurement noise matrix
            Q = np.array([[self._sigma_z**2]])
            
            # Innovation (difference between measurement and prediction)
            innovation = z_measured - z_predicted
            
            # Innovation covariance
            S = H @ self._sigma @ H.T + Q
            
            # Kalman gain
            K = self._sigma @ H.T @ np.linalg.inv(S)
            
            # Update the state
            self._mu = self._mu + K @ np.array([[innovation]])
            
            # Update the covariance (Joseph form to ensure symmetry)
            I = np.eye(3)
            self._sigma = (I - K @ H) @ self._sigma @ (I - K @ H).T + K @ Q @ K.T
            
            # Normalize angle
            self._mu[2, 0] = self._mu[2, 0] % (2 * math.pi)

    def _sense(self, pose: tuple[float, float, float]) -> list[float]:
        """
        Gets the predicted measurements for each LiDAR ray for a given pose.
        
        Args:
            pose: Robot pose (x, y, theta) in [m, m, rad].
            
        Returns:
            List of predicted measurements; nan if a sensor is out of range.
        """
        z_hat = []
        
        # Get position and orientation
        x, y, theta = pose
        
        # Get the LiDAR rays from the robot's position
        indices = tuple(range(0, 240, self._interval))
        lidar_segments = self._lidar_rays(pose, indices)
        
        # For each ray, check for collision with the map
        for segment in lidar_segments:
            collision_result, distance = self._map.check_collision(segment, compute_distance=True)
            z_hat.append(distance)
            
        return z_hat
        
    def _lidar_rays(
        self, pose: tuple[float, float, float], indices: tuple[float], degree_increment: float = 1.5
    ) -> list[list[tuple[float, float]]]:
        """
        Determines the simulated LiDAR ray segments for a given pose.
        
        Args:
            pose: Robot pose (x, y, theta) in [m] and [rad].
            indices: Rays of interest in counterclockwise order (0 for the front ray).
            degree_increment: Angular difference of the sensor between contiguous rays [degrees].
            
        Returns:
            Ray segments. Format:
             [[(x0_start, y0_start), (x0_end, y0_end)],
              [(x1_start, y1_start), (x1_end, y1_end)],
              ...]
        """
        x, y, theta = pose
        
        # Convert the sensor origin to world coordinates
        x_start = x - 0.035 * math.cos(theta)
        y_start = y - 0.035 * math.sin(theta)
        
        rays = []
        
        for index in indices:
            ray_angle = math.radians(degree_increment * index)
            x_end = x_start + self._sensor_range_max * math.cos(theta + ray_angle)
            y_end = y_start + self._sensor_range_max * math.sin(theta + ray_angle)
            rays.append([(x_start, y_start), (x_end, y_end)])
            
        return rays
            
    def get_pose(self) -> tuple[float, float, float]:
        """
        Gets the current pose estimated by the filter.
        
        Returns:
            Estimated pose (x, y, theta) in [m, m, rad].
        """
        return (self._mu[0, 0], self._mu[1, 0], self._mu[2, 0])
        
    def get_covariance(self) -> np.ndarray:
        """
        Gets the current covariance matrix.
        
        Returns:
            3x3 covariance matrix.
        """
        return self._sigma
        
    def calculate_likelihood(self, measurements: list[float]) -> float:
        """
        Calculates the average likelihood of the measurements for the current pose.
        Useful for detecting localization issues.
        
        Args:
            measurements: LiDAR sensor measurements.
            
        Returns:
            Average likelihood of the measurements.
        """
        # Get the current pose
        pose = self.get_pose()
        
        # Get the predicted measurements
        predicted_measurements = self._sense(pose)
        
        # Preprocess the real measurements
        processed_measurements = np.array(measurements[0:240:self._interval])
        
        # Accumulate likelihood
        total_likelihood = 0.0
        valid_count = 0
        
        # Compare each real measurement with the predicted one
        for measured, predicted in zip(processed_measurements, predicted_measurements):
            if np.isnan(measured) and np.isnan(predicted):
                continue
            elif np.isnan(measured) or np.isnan(predicted):
                total_likelihood += 0.1
                valid_count += 1
                continue
                
            # Calculate likelihood using Gaussian
            diff = measured - predicted
            likelihood = np.exp(-0.5 * (diff / self._sigma_z)**2) / (self._sigma_z * math.sqrt(2.0 * math.pi))
            
            # Accumulate
            total_likelihood += likelihood
            valid_count += 1
            
        # Calculate average
        if valid_count > 0:
            avg_likelihood = total_likelihood / valid_count
        else:
            avg_likelihood = 0.0
            
        return avg_likelihood