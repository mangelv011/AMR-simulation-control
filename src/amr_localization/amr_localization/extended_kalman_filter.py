import numpy as np
import math
from amr_localization.maps import Map

class ExtendedKalmanFilter:
    """
    Implementación del Filtro de Kalman Extendido (EKF) para la localización del robot.
    Se usa como sustituto del filtro de partículas una vez que el robot está localizado.
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
        Inicializa el filtro de Kalman extendido.
        
        Args:
            dt: Período de muestreo [s].
            map_path: Ruta al mapa del entorno.
            initial_pose: Pose inicial del robot (x, y, theta) en [m, m, rad].
            sigma_v: Desviación estándar de la velocidad lineal [m/s].
            sigma_w: Desviación estándar de la velocidad angular [rad/s].
            sigma_z: Desviación estándar de las mediciones [m].
            sensor_range_max: Rango máximo del sensor [m].
            sensor_range_min: Rango mínimo del sensor [m].
        """
        self._dt = dt
        self._sigma_v = sigma_v
        self._sigma_w = sigma_w
        self._sigma_z = sigma_z
        self._sensor_range_max = sensor_range_max
        self._sensor_range_min = sensor_range_min
        self._interval = 20  # Mismo intervalo que el filtro de partículas

        # Inicializar el mapa
        self._map = Map(
            map_path,
            sensor_range_max,
            compiled_intersect=True,
            use_regions=False,
            safety_distance=0.08,
        )
        
        # Estado inicial [x, y, theta]
        self._mu = np.array([initial_pose[0], initial_pose[1], initial_pose[2]]).reshape(3, 1)
        
        # Matriz de covarianza inicial (incertidumbre inicial)
        # Valores pequeños porque asumimos que la pose inicial es precisa
        self._sigma = np.diag([0.01, 0.01, 0.01])  
        
        # Matrices de ruido del proceso para el modelo de movimiento
        self._R = np.diag([self._sigma_v**2, self._sigma_w**2])
        
        # Contador de iteraciones
        self._iteration = 0

    def predict(self, v: float, w: float) -> None:
        """
        Actualización de predicción del EKF basada en el modelo de movimiento.
        
        Args:
            v: Velocidad lineal [m/s].
            w: Velocidad angular [rad/s].
        """
        self._iteration += 1
        
        # Estado actual
        x = self._mu[0, 0]
        y = self._mu[1, 0]
        theta = self._mu[2, 0]
        
        # Si la velocidad angular es aproximadamente cero, usar modelo lineal
        if abs(w) < 1e-6:
            # Modelo lineal: el robot se mueve en línea recta
            x_new = x + v * self._dt * math.cos(theta)
            y_new = y + v * self._dt * math.sin(theta)
            theta_new = theta
            
            # Jacobiano del modelo de movimiento con respecto al estado
            G = np.array([
                [1.0, 0.0, -v * self._dt * math.sin(theta)],
                [0.0, 1.0, v * self._dt * math.cos(theta)],
                [0.0, 0.0, 1.0]
            ])
            
            # Jacobiano del modelo de movimiento con respecto al ruido
            V = np.array([
                [self._dt * math.cos(theta), 0],
                [self._dt * math.sin(theta), 0],
                [0, 0]
            ])
        else:
            # Modelo no lineal: el robot gira y avanza
            v_w_ratio = v / w
            theta_new = (theta + w * self._dt) % (2 * math.pi)
            x_new = x + v_w_ratio * (math.sin(theta_new) - math.sin(theta))
            y_new = y - v_w_ratio * (math.cos(theta_new) - math.cos(theta))
            
            # Jacobiano del modelo de movimiento con respecto al estado
            G = np.array([
                [1.0, 0.0, v_w_ratio * (math.cos(theta_new) - math.cos(theta))],
                [0.0, 1.0, v_w_ratio * (math.sin(theta_new) - math.sin(theta))],
                [0.0, 0.0, 1.0]
            ])
            
            # Jacobiano del modelo de movimiento con respecto al ruido
            V = np.array([
                [(math.sin(theta_new) - math.sin(theta))/w, -v/(w**2)*(math.sin(theta_new) - math.sin(theta)) + v/w*self._dt*math.cos(theta_new)],
                [-(math.cos(theta_new) - math.cos(theta))/w, v/(w**2)*(math.cos(theta_new) - math.cos(theta)) + v/w*self._dt*math.sin(theta_new)],
                [0, self._dt]
            ])
        
        # Verificar colisión con el mapa y ajustar si es necesario
        segment = [(x, y), (x_new, y_new)]
        intersection_point, _ = self._map.check_collision(segment)
        
        if intersection_point:
            x_new, y_new = intersection_point
        
        # Actualizar el estado
        self._mu = np.array([x_new, y_new, theta_new]).reshape(3, 1)
        
        # Actualizar la covarianza
        self._sigma = G @ self._sigma @ G.T + V @ self._R @ V.T

    def update(self, measurements: list[float]) -> None:
        """
        Actualización de corrección del EKF basada en las mediciones del sensor.
        
        Args:
            measurements: Mediciones del sensor LiDAR [m].
        """
        # Procesar mediciones para usar solo las que corresponden al intervalo
        processed_measurements = np.array(measurements[0:240:self._interval])
        
        # Obtener la pose actual
        x = self._mu[0, 0]
        y = self._mu[1, 0]
        theta = self._mu[2, 0]
        pose = (x, y, theta)
        
        # Obtener las mediciones predichas basadas en la pose actual y el mapa
        predicted_measurements = self._sense(pose)
        
        # Para cada medición, realizar una actualización del EKF
        for i, (z_measured, z_predicted) in enumerate(zip(processed_measurements, predicted_measurements)):
            # Ignorar mediciones inválidas (fuera de rango o colisión)
            if np.isnan(z_measured) or np.isnan(z_predicted):
                continue
                
            # Calcular el ángulo del rayo para este índice
            ray_angle = math.radians(1.5 * i * self._interval)
            ray_angle_global = theta + ray_angle
            
            # Calcular el punto final del rayo (coordenadas del obstáculo detectado)
            ray_endpoint_x = x + z_predicted * math.cos(ray_angle_global)
            ray_endpoint_y = y + z_predicted * math.sin(ray_angle_global)
            
            # Calcular el Jacobiano de la medición (derivada con respecto al estado)
            # La medición es la distancia entre el robot y el obstáculo
            dx = ray_endpoint_x - x
            dy = ray_endpoint_y - y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 1e-6:  # Evitar división por cero
                continue
                
            H = np.zeros((1, 3))
            H[0, 0] = -dx / distance  # dx/dx
            H[0, 1] = -dy / distance  # dx/dy
            H[0, 2] = z_predicted * (-dx * math.sin(ray_angle_global) + dy * math.cos(ray_angle_global)) / distance
            
            # Matriz de ruido de la medición
            Q = np.array([[self._sigma_z**2]])
            
            # Innovación (diferencia entre medición y predicción)
            innovation = z_measured - z_predicted
            
            # Covarianza de la innovación
            S = H @ self._sigma @ H.T + Q
            
            # Ganancia de Kalman
            K = self._sigma @ H.T @ np.linalg.inv(S)
            
            # Actualizar el estado
            self._mu = self._mu + K @ np.array([[innovation]])
            
            # Actualizar la covarianza (fórmula de Joseph para garantizar simetría)
            I = np.eye(3)
            self._sigma = (I - K @ H) @ self._sigma @ (I - K @ H).T + K @ Q @ K.T
            
            # Normalizar ángulo
            self._mu[2, 0] = self._mu[2, 0] % (2 * math.pi)

    def _sense(self, pose: tuple[float, float, float]) -> list[float]:
        """
        Obtiene las mediciones predichas de cada rayo LiDAR para una pose dada.
        
        Args:
            pose: Pose del robot (x, y, theta) en [m, m, rad].
            
        Returns:
            Lista de mediciones predichas; nan si un sensor está fuera de rango.
        """
        z_hat = []
        
        # Obtener posición y orientación
        x, y, theta = pose
        
        # Obtener los rayos LiDAR desde la posición del robot
        indices = tuple(range(0, 240, self._interval))
        lidar_segments = self._lidar_rays(pose, indices)
        
        # Para cada rayo, verificar colisión con el mapa
        for segment in lidar_segments:
            collision_result, distance = self._map.check_collision(segment, compute_distance=True)
            z_hat.append(distance)
            
        return z_hat
        
    def _lidar_rays(
        self, pose: tuple[float, float, float], indices: tuple[float], degree_increment: float = 1.5
    ) -> list[list[tuple[float, float]]]:
        """
        Determina los segmentos de rayos LiDAR simulados para una pose dada.
        
        Args:
            pose: Pose del robot (x, y, theta) en [m] y [rad].
            indices: Rayos de interés en orden antihorario (0 para el rayo frontal).
            degree_increment: Diferencia angular del sensor entre rayos contiguos [grados].
            
        Returns:
            Segmentos de rayos. Formato:
             [[(x0_start, y0_start), (x0_end, y0_end)],
              [(x1_start, y1_start), (x1_end, y1_end)],
              ...]
        """
        x, y, theta = pose
        
        # Convertir el origen del sensor a coordenadas mundiales
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
        Obtiene la pose actual estimada por el filtro.
        
        Returns:
            Pose estimada (x, y, theta) en [m, m, rad].
        """
        return (self._mu[0, 0], self._mu[1, 0], self._mu[2, 0])
        
    def get_covariance(self) -> np.ndarray:
        """
        Obtiene la matriz de covarianza actual.
        
        Returns:
            Matriz de covarianza 3x3.
        """
        return self._sigma
        
    def calculate_likelihood(self, measurements: list[float]) -> float:
        """
        Calcula la verosimilitud promedio de las mediciones para la pose actual.
        Útil para detectar problemas de localización.
        
        Args:
            measurements: Mediciones del sensor LiDAR.
            
        Returns:
            Verosimilitud promedio de las mediciones.
        """
        # Obtener la pose actual
        pose = self.get_pose()
        
        # Obtener las mediciones predichas
        predicted_measurements = self._sense(pose)
        
        # Preprocesar las mediciones reales
        processed_measurements = np.array(measurements[0:240:self._interval])
        
        # Acumular verosimilitud
        total_likelihood = 0.0
        valid_count = 0
        
        # Comparar cada medición real con la predicha
        for measured, predicted in zip(processed_measurements, predicted_measurements):
            if np.isnan(measured) and np.isnan(predicted):
                continue
            elif np.isnan(measured) or np.isnan(predicted):
                total_likelihood += 0.1
                valid_count += 1
                continue
                
            # Calcular verosimilitud usando Gaussiana
            diff = measured - predicted
            likelihood = np.exp(-0.5 * (diff / self._sigma_z)**2) / (self._sigma_z * math.sqrt(2.0 * math.pi))
            
            # Acumular
            total_likelihood += likelihood
            valid_count += 1
            
        # Calcular promedio
        if valid_count > 0:
            avg_likelihood = total_likelihood / valid_count
        else:
            avg_likelihood = 0.0
            
        return avg_likelihood