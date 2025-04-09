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
        # TODO: 3.10. Complete the missing function body with your code.

        localized: bool = False
        pose: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

        if len(self._particles) > 0:
            positions = np.array([(p[0], p[1]) for p in self._particles])
            orientations = np.array([p[2] for p in self._particles])

            # Usar DBSCAN más permisivo al inicio
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(positions)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters > 0:
                # Mantener más partículas cuando hay múltiples clusters
                if n_clusters > 1:
                    target_particles = max(2000, n_clusters * 200)
                    if len(self._particles) > target_particles:
                        self._particles = self._particles[:target_particles]
                    return localized, pose

                # Procesar solo cuando hay un cluster
                valid_indices = np.where(labels == 0)[0]
                if len(valid_indices) >= 5:
                    localized = True
                    x_mean = np.mean(positions[valid_indices, 0])
                    y_mean = np.mean(positions[valid_indices, 1])

                    thetas = orientations[valid_indices]
                    cos_mean = np.mean(np.cos(thetas))
                    sin_mean = np.mean(np.sin(thetas))
                    theta_mean = np.arctan2(sin_mean, cos_mean) % (2 * np.pi)

                    pose = (x_mean, y_mean, theta_mean)

                    # Reducir partículas solo cuando estamos seguros de la localización
                    if localized and len(self._particles) > 100:
                        self._particles = self._particles[valid_indices][:100]

        return localized, pose

    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        self._iteration += 1

        # TODO: 3.5. Complete the function body with your code.

        for particle in self._particles:
            # Current orientation
            theta = particle[2]

            # Add Gaussian noise to velocities
            noisy_v = v + np.random.normal(0, self._sigma_v)
            noisy_w = w + np.random.normal(0, self._sigma_w)

            # Calculate movement steps
            x_step = noisy_v * self._dt * np.cos(theta)
            y_step = noisy_v * self._dt * np.sin(theta)
            theta_step = noisy_w * self._dt

            # Store original position
            old_x, old_y = particle[0], particle[1]

            # Calculate new position
            new_x = old_x + x_step
            new_y = old_y + y_step

            # Define the movement segment
            segment = [(old_x, old_y), (new_x, new_y)]

            # Check if particle collided with a wall
            intersection_point, _ = self._map.check_collision(segment)

            if intersection_point:
                # If collision occurred, place the particle at the collision point
                particle[0], particle[1] = intersection_point
            else:
                # No collision, update to new position
                particle[0] = new_x
                particle[1] = new_y

            # Update orientation and keep in range [0, 2π)
            particle[2] = (particle[2] + theta_step) % (2 * np.pi)

    def resample(self, measurements: list[float]) -> None:
        """Samples a new set of particles.

        Args:
            measurements: Sensor measurements [m].

        """
        # TODO: 3.9. Complete the function body with your code (i.e., replace the pass statement).

        num_particles = len(self._particles)
        
        # Calculate weights for all particles using vectorization if possible
        weights = np.array([self._measurement_probability(measurements, tuple(particle[:3])) 
                            for particle in self._particles])
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # If all weights are zero, use uniform distribution
            weights = np.ones(num_particles) / num_particles
        
        # Prepare for Systematic Sampling
        # Create cumulative sum of weights
        cumulative_sum = np.cumsum(weights)
        
        # Generate a single random starting point between 0 and 1/N
        step_size = 1.0 / num_particles
        start_point = np.random.uniform(0, step_size)
        
        # Generate evenly spaced points for sampling
        points = start_point + np.arange(num_particles) * step_size
        
        # Find indices of particles to be resampled using searchsorted
        indices = np.searchsorted(cumulative_sum, points)
        
        # Ensure indices are within bounds
        indices = np.clip(indices, 0, num_particles - 1)
        
        # Select particles based on indices
        new_particles = self._particles[indices].copy()
        
        # Add small random noise to avoid particle depletion
        # new_particles[:, 0] += np.random.normal(0, 0.01, num_particles)  # Small noise in x
        # new_particles[:, 1] += np.random.normal(0, 0.01, num_particles)  # Small noise in y
        # new_particles[:, 2] = (new_particles[:, 2] + np.random.normal(0, 0.01, num_particles)) % (
        #     2 * np.pi
        # )  # Small noise in orientation
        
        # Replace old particles with new ones
        self._particles = new_particles

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
