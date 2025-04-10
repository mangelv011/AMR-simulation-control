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
        
        # # ADDED: Precalcular valores constantes
        self._lookahead_squared = lookahead_distance ** 2
        self._angle_threshold = np.pi/4.0  # # MODIFIED: Umbral más permisivo (45 grados)
        self._base_velocity = 0.4  # m/s  # # MODIFIED: Velocidad base más alta
        self._min_velocity = 0.15  # # ADDED: Velocidad mínima más alta
        
        # # ADDED: Cache para puntos más cercanos
        self._last_closest_idx = 0
        
        # # ADDED: Factores para suavizado de velocidad
        self._max_angular_vel = 1.5  # Velocidad angular máxima
        self._slowdown_factor = 0.5  # Factor de reducción al final del camino

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
        
        # # OPTIMIZED: Calcular diferencias una vez y reutilizarlas
        dx = target_point[0] - x
        dy = target_point[1] - y
        
        # # OPTIMIZED: Usar arctan2 directamente 
        alpha = np.arctan2(dy, dx) - theta
        alpha_norm = (alpha + np.pi) % (2 * np.pi) - np.pi  # Normalize angle to [-pi, pi]
        
        # Make the robot rotate in place when the angle error is large
        if abs(alpha_norm) < self._angle_threshold: 
            self._aligned = True
        if not self._aligned:
            # # MODIFIED: Rotación más rápida para alinearse
            return 0.0, np.sign(alpha_norm) * self._max_angular_vel
            
        # # MODIFIED: Mejorar el cálculo de velocidad lineal
        path_len = len(self._path)
        # Calcular factor de progreso en el camino (0-1)
        progress = min(1.0, max(0.0, closest_idx / max(1, path_len - 1)))
        
        # Determinar velocidad basada en:
        # 1. Progreso en el camino (reducir velocidad al final)
        # 2. Ángulo de desviación (reducir velocidad en curvas)
        # 3. Velocidad base
        
        if closest_idx > path_len - 5:  # Últimos 5 puntos del camino
            # Reducción gradual al final del camino
            v = max(self._min_velocity, self._base_velocity * (1.0 - progress * self._slowdown_factor))
        else:
            # Calcular factor de reducción basado en el ángulo (1.0 en línea recta, menor en curvas)
            angle_factor = max(0.7, 1.0 - abs(alpha_norm) / np.pi)
            v = self._base_velocity * angle_factor
        
        # # MODIFIED: Calcular velocidad angular con limitación más suave
        # Calcular w basado en la fórmula de pure pursuit pero limitar para movimientos más suaves
        raw_w = 2.0 * v * np.sin(alpha_norm) / self._lookahead_distance
        
        # Limitar w para evitar cambios bruscos que reducirían la velocidad
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
        # # ADDED: Resetear el índice de caché al cambiar el path
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
        
        # # OPTIMIZED: Buscar desde el último punto más cercano conocido
        start_idx = max(0, self._last_closest_idx - 2)
        end_idx = min(len(self._path), self._last_closest_idx + 10)
        
        # Initialize variables with the first point to check
        closest_idx = start_idx
        closest_xy = self._path[start_idx]
        min_distance = (self._path[start_idx][0] - x)**2 + (self._path[start_idx][1] - y)**2
        
        # # OPTIMIZED: Recorrer solo una ventana de puntos probables
        for i in range(start_idx, end_idx):
            point = self._path[i]
            # # OPTIMIZED: Usar distancia al cuadrado en lugar de raíz cuadrada
            distance = (point[0] - x)**2 + (point[1] - y)**2
            
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
                closest_xy = point
        
        # # OPTIMIZED: Si el punto más cercano está al borde de nuestra ventana,
        # #           ampliar la búsqueda para asegurar que encontramos el mínimo global
        if closest_idx == end_idx - 1 and end_idx < len(self._path):
            # Buscar en el resto del camino
            for i in range(end_idx, len(self._path)):
                point = self._path[i]
                distance = (point[0] - x)**2 + (point[1] - y)**2
                
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = i
                    closest_xy = point
                else:
                    # Si la distancia empieza a aumentar, podemos parar
                    break
        
        # # ADDED: Actualizar caché con el nuevo índice encontrado
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
        
        # # OPTIMIZED: Iniciar desde el punto más cercano + 1 para evitar buscar atrás
        current_idx = max(origin_idx, 0)
        accumulated_distance = 0.0
        
        # # OPTIMIZED: Precalcular valores para evitar cálculos repetidos
        lookahead_distance = self._lookahead_distance
        
        # Look for a point that's at least lookahead_distance away
        while current_idx < path_len - 1:
            current_point = self._path[current_idx]
            next_point = self._path[current_idx + 1]
            
            # # OPTIMIZED: Cálculo de distancia usando dx y dy una sola vez
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            segment_length = np.sqrt(dx*dx + dy*dy)
            
            # Check if lookahead point is on this segment
            if accumulated_distance + segment_length >= lookahead_distance:
                # # OPTIMIZED: Interpolación lineal más directa
                remaining_distance = lookahead_distance - accumulated_distance
                ratio = remaining_distance / segment_length
                
                # # OPTIMIZED: Cálculo vectorial más eficiente
                return (current_point[0] + ratio * dx, 
                        current_point[1] + ratio * dy)
            
            accumulated_distance += segment_length
            current_idx += 1
        
        # If we reach here, return the last point in the path
        return self._path[-1]
