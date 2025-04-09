import datetime
import math
import numpy as np
import os
import pytz
import random

from amr_localization.maps import Map
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


class ParticleFilter:
    """Particle filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        particle_count: int,
        sigma_v: float = 0.05,
        sigma_w: float = 0.1,
        sigma_z: float = 0.2,
        sensor_range_max: float = 8.0,
        sensor_range_min: float = 0.16,
        global_localization: bool = True,
        initial_pose: tuple[float, float, float] = (float("nan"), float("nan"), float("nan")),
        initial_pose_sigma: tuple[float, float, float] = (float("nan"), float("nan"), float("nan")),
    ):
        """Particle filter class initializer.

        Args:
            dt: Sampling period [s].
            map_path: Path to the map of the environment.
            particle_count: Initial number of particles.
            sigma_v: Standard deviation of the linear velocity [m/s].
            sigma_w: Standard deviation of the angular velocity [rad/s].
            sigma_z: Standard deviation of the measurements [m].
            sensor_range_max: Maximum sensor measurement range [m].
            sensor_range_min: Minimum sensor measurement range [m].
            global_localization: First localization if True, pose tracking otherwise.
            initial_pose: Approximate initial robot pose (x, y, theta) for tracking [m, m, rad].
            initial_pose_sigma: Standard deviation of the initial pose guess [m, m, rad].

        """
        self._dt: float = dt
        self._initial_particle_count: int = particle_count
        self._particle_count: int = particle_count
        self._sensor_range_max: float = sensor_range_max
        self._sensor_range_min: float = sensor_range_min
        self._sigma_v: float = sigma_v
        self._sigma_w: float = sigma_w
        self._sigma_z: float = sigma_z
        self._iteration: int = 0

        self._map = Map(
            map_path,
            sensor_range_max,
            compiled_intersect=True,
            use_regions=False,
            safety_distance=0.08,
        )
        self._particles = self._init_particles(
            particle_count, global_localization, initial_pose, initial_pose_sigma
        )
        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        self._interval = 20

    def compute_pose(self) -> tuple[bool, tuple[float, float, float]]:
        """Computes the pose estimate when the particles form a single DBSCAN cluster.

        Adapts the amount of particles depending on the number of clusters during localization.
        100 particles are kept for pose tracking.

        Returns:
            localized: True if the pose estimate is valid.
            pose: Robot pose estimate (x, y, theta) [m, m, rad].

        """
        # Optimized version of compute_pose for maximum speed
        
        # Early return for empty particles
        if len(self._particles) == 0:
            return False, (float("inf"), float("inf"), float("inf"))
            
        # Pre-allocate result variables
        localized = False
        pose = (float("inf"), float("inf"), float("inf"))
        
        # Use direct NumPy array access instead of list comprehension
        # This avoids creating intermediate lists
        positions = self._particles[:, :2]
        orientations = self._particles[:, 2].astype(np.float64)  # Ensure numeric type
        
        # Configure DBSCAN for best performance with current data
        # Use float32 for positions to speed up distance calculations
        pos_float32 = np.array(positions, dtype=np.float32)
        clustering = DBSCAN(
            eps=0.5, 
            min_samples=5,
            algorithm='kd_tree',  # KD-tree is fastest for low-dimensional Euclidean space
            leaf_size=30,  # Optimal for most cases
            n_jobs=-1  # Use all available processors
        ).fit(pos_float32)
        
        # Get unique labels efficiently
        labels = clustering.labels_
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters > 0:
            # Multi-cluster case - just adjust particle count and return
            if n_clusters > 1:
                target_particles = max(2000, n_clusters * 200)
                if len(self._particles) > target_particles:
                    self._particles = self._particles[:target_particles]
                return localized, pose
            
            # Single cluster case - compute pose
            # Get valid indices using fast NumPy boolean indexing
            mask = (labels == 0)
            if np.sum(mask) >= 5:  # At least 5 particles in cluster
                localized = True
                
                # Fast mean calculation with NumPy
                x_mean = np.mean(positions[mask, 0])
                y_mean = np.mean(positions[mask, 1])
                
                # Efficient circular mean for angles
                thetas = orientations[mask]
                sin_sum = np.sum(np.sin(thetas))
                cos_sum = np.sum(np.cos(thetas))
                count = len(thetas)
                
                # Vectorized calculation is faster than np.mean
                sin_mean = sin_sum / count
                cos_mean = cos_sum / count
                theta_mean = np.arctan2(sin_mean, cos_mean) % (2 * np.pi)
                
                pose = (x_mean, y_mean, theta_mean)
                
                # Reduce particles if localized with fast array slicing
                if len(self._particles) > 100:
                    # Get indices where mask is True
                    valid_indices = np.where(mask)[0][:100]
                    self._particles = self._particles[valid_indices]
        
        return localized, pose

    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        self._iteration += 1

        # TODO: 3.5. Complete the function body with your code.
        
        # Early return for empty particles to save computation
        if len(self._particles) == 0:
            return
            
        # Extract current orientations - convert to numpy array if needed
        # Use direct array access for better performance
        thetas = self._particles[:, 2].astype(np.float64)
        
        # Generate noise vectors for all particles at once
        particle_count = len(self._particles)
        # Use a single random call with larger array for better performance
        noise = np.random.normal(0, [self._sigma_v, self._sigma_w], (particle_count, 2))
        noisy_v = v + noise[:, 0]
        noisy_w = w + noise[:, 1]
        
        # Pre-compute trigonometric functions once
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)
        
        # Calculate movement steps for all particles (vectorized)
        dt = self._dt  # Cache this value for better performance
        x_steps = noisy_v * dt * cos_thetas
        y_steps = noisy_v * dt * sin_thetas
        theta_steps = noisy_w * dt
        
        # Store original positions and create arrays for new positions
        # Use direct array access for better performance
        old_x = self._particles[:, 0].astype(np.float64)
        old_y = self._particles[:, 1].astype(np.float64)
        new_x = old_x + x_steps
        new_y = old_y + y_steps
        
        # Create array to hold collision status
        collided = np.zeros(particle_count, dtype=bool)
        collision_points = np.zeros((particle_count, 2))
        
        # Check collisions for each particle
        # Use pre-allocated segments array for memory efficiency
        for i in range(particle_count):
            segment = [(old_x[i], old_y[i]), (new_x[i], new_y[i])]
            intersection_point, _ = self._map.check_collision(segment)
            
            if intersection_point:
                collided[i] = True
                collision_points[i] = intersection_point
        
        # Apply updates to each particle individually to avoid type issues
        for i in range(particle_count):
            if collided[i]:
                self._particles[i, 0] = collision_points[i, 0]
                self._particles[i, 1] = collision_points[i, 1]
            else:
                self._particles[i, 0] = new_x[i]
                self._particles[i, 1] = new_y[i]
            
            # Update orientations and keep in range [0, 2π)
            self._particles[i, 2] = (thetas[i] + theta_steps[i]) % (2 * np.pi)

    def resample(self, measurements: list[float]) -> None:
        """Samples a new set of particles.

        Args:
            measurements: Sensor measurements [m].

        """
        # TODO: 3.9. Complete the function body with your code (i.e., replace the pass statement).

        # Pre-process measurements once instead of repeatedly
        processed_measurements = np.array(measurements[0:240:self._interval])
        
        # Pre-allocate arrays for predicted measurements to avoid re-allocating in loop
        num_particles = len(self._particles)
        weights = np.zeros(num_particles)
        
        # Calculate all predicted measurements in one batch if possible
        # If not possible, calculate individually but efficiently
        for i in range(num_particles):
            particle = tuple(self._particles[i][:3])
            predicted = self._sense(particle)
            
            # Vectorized probability calculation
            prob = 1.0
            for j, (measured, pred) in enumerate(zip(processed_measurements, predicted)):
                if np.isnan(measured) and np.isnan(pred):
                    continue  # prob *= 1.0
                elif np.isnan(measured) or np.isnan(pred):
                    prob *= 0.1
                    continue
                
                # Fast Gaussian calculation
                diff = measured - pred
                prob *= np.exp(-0.5 * (diff / self._sigma_z)**2) / (self._sigma_z * 2.50662827463)
                
                # Avoid numerical underflow
                if prob < 1e-300:
                    prob = 1e-300
            
            weights[i] = prob
        
        # Fast normalization using NumPy
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            weights /= sum_weights
        else:
            weights.fill(1.0 / num_particles)
        
        # Optimized systematic resampling using NumPy
        # Pre-compute cumulative sum
        cumsum = np.cumsum(weights)
        
        # Generate sample points
        step = 1.0 / num_particles
        u = np.random.uniform(0, step)
        sample_points = u + np.arange(num_particles) * step
        
        # Use searchsorted for fastest sample selection
        indices = np.searchsorted(cumsum, sample_points)
        
        # Clip indices to valid range and create new particles
        indices = np.clip(indices, 0, num_particles - 1)
        
        # Use NumPy advanced indexing for fast copying
        self._particles = self._particles[indices].copy()

    def plot(self, axes, orientation: bool = True):
        """Draws particles.

        Args:
            axes: Figure axes.
            orientation: Draw particle orientation.

        Returns:
            axes: Modified axes.

        """
        if orientation:
            dx = [math.cos(particle[2]) for particle in self._particles]
            dy = [math.sin(particle[2]) for particle in self._particles]
            axes.quiver(
                self._particles[:, 0],
                self._particles[:, 1],
                dx,
                dy,
                color="b",
                scale=15,
                scale_units="inches",
            )
        else:
            axes.plot(self._particles[:, 0], self._particles[:, 1], "bo", markersize=1)

        return axes

    def show(
        self,
        title: str = "",
        orientation: bool = True,
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            orientation: Draw particle orientation.
            display: True to open a window to visualize the particle filter evolution in real-time.
                Time consuming. Does not work inside a container unless the screen is forwarded.
            block: True to stop program execution until the figure window is closed.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure = self._figure
        axes = self._axes
        axes.clear()

        axes = self._map.plot(axes)
        axes = self.plot(axes, orientation)

        axes.set_title(title + " (Iteration #" + str(self._iteration) + ")")
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait 1 ms or the figure won't be displayed

        if save_figure:
            save_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", save_dir, self._timestamp)
            )

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = str(self._iteration).zfill(4) + " " + title.lower() + ".png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def _init_particles(
        self,
        particle_count: int,
        global_localization: bool,
        initial_pose: tuple[float, float, float],
        initial_pose_sigma: tuple[float, float, float],
    ) -> np.ndarray:
        """Draws N random valid particles.

        The particles are guaranteed to be inside the map and
        can only have the following orientations [0, pi/2, pi, 3*pi/2].

        Args:
            particle_count: Number of particles.
            global_localization: First localization if True, pose tracking otherwise.
            initial_pose: Approximate initial robot pose (x, y, theta) for tracking [m, m, rad].
            initial_pose_sigma: Standard deviation of the initial pose guess [m, m, rad].

        Returns: A NumPy array of tuples (x, y, theta) [m, m, rad].

        """
        particles = np.empty((particle_count, 3), dtype=object)

        # TODO: 3.4. Complete the missing function body with your code.

        map_limits = self._map.bounds()
        valid_orientations = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

        num_particles = 0
        while num_particles < particle_count:
            if global_localization:
                theta = np.random.choice(valid_orientations)
                # Generar posiciones aleatorias dentro de los límites del mapa
                x = np.random.uniform(map_limits[0], map_limits[2])
                y = np.random.uniform(map_limits[1], map_limits[3])
            else:
                # Generar posiciones cerca de la pose inicial usando una distribución normal
                x = np.random.normal(initial_pose[0], initial_pose_sigma[0])
                y = np.random.normal(initial_pose[1], initial_pose_sigma[1])
                theta = np.random.normal(initial_pose[2], initial_pose_sigma[2])

            if self._map.contains((x, y)):  # Comprobar que la partícula está en una zona libre
                particles[num_particles] = (x, y, theta)
                num_particles += 1

        return particles

    def _sense(self, particle: tuple[float, float, float]) -> list[float]:
        """Obtains the predicted measurement of every LiDAR ray given the robot's pose.

        Args:
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns: List of predicted measurements; nan if a sensor is out of range.

        """
        z_hat: list[float] = []

        # TODO: 3.6. Complete the missing function body with your code.

        # Get particle position and orientation
        x, y, theta = particle

        # Get the LiDAR rays from the particle position
        indices = tuple(range(0, 240, self._interval)) 
        lidar_segments = self._lidar_rays((x, y, theta), indices)

        # For each ray, check if it collides with the map
        for segment in lidar_segments:
            # Check collision with the map
            collision_result, distance = self._map.check_collision(segment, compute_distance=True)

         
            z_hat.append(distance)

        return z_hat

    @staticmethod
    def _gaussian(mu: float, sigma: float, x: float) -> float:
        """Computes the value of a Gaussian.

        Args:
            mu: Mean.
            sigma: Standard deviation.
            x: Variable.

        Returns:
            float: Gaussian value.

        """
        # TODO: 3.7. Complete the function body (i.e., replace the code below).
        # Calculate the Gaussian probability density function
        # p(x) = (1/sqrt(2π*sigma²)) * e^(-(x-mu)²/(2*sigma²))

        # Handle division by zero
        if sigma == 0:
            return 1.0 if x == mu else 0.0

        # Calculate the exponent term
        exponent = -0.5 * ((x - mu) / sigma) ** 2

        # Calculate the normalization factor
        normalization = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

        # Calculate the full Gaussian
        return normalization * np.exp(exponent)

    def _lidar_rays(
        self, pose: tuple[float, float, float], indices: tuple[float], degree_increment: float = 1.5
    ) -> list[list[tuple[float, float]]]:
        """Determines the simulated LiDAR ray segments for a given robot pose.

        Args:
            pose: Robot pose (x, y, theta) in [m] and [rad].
            indices: Rays of interest in counterclockwise order (0 for to the forward-facing ray).
            degree_increment: Angle difference of the sensor between contiguous rays [degrees].

        Returns: Ray segments. Format:
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

    def _measurement_probability(
        self, measurements: list[float], particle: tuple[float, float, float]
    ) -> float:
        """Computes the probability of a set of measurements given a particle's pose."""
        probability = 1.0

        # Get predicted measurements for this particle
        predicted_measurements = self._sense(particle)
     
        measurements = [measurements[i] for i in range(0,240,self._interval)]

        # Compare each real measurement with the predicted one
        for i, (measured, predicted) in enumerate(zip(measurements, predicted_measurements)):
            # Handle NaN values (out of range measurements)
            if np.isnan(measured) and np.isnan(predicted):
                probability *= 1.0  # Buen match
                continue
            elif np.isnan(measured) or np.isnan(predicted):
                probability *= 0.1  # Penalización moderada por desajuste
                continue

            # Calculate probability for this measurement using Gaussian
            measurement_prob = self._gaussian(predicted, self._sigma_z, measured)
            
            # Add small constant to avoid zero probabilities
            measurement_prob = max(measurement_prob, 1e-300)
            
            # Update total probability
            probability *= measurement_prob

        return probability
