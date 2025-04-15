import math


class WallFollower:
    def __init__(self, dt: float) -> None:
        self._dt: float = dt  # para el real *2, para el simulado *1
        self._desired_distance = 0.2
        self.Kp = 2
        self.Kd = 1
        self.Ki = 0.005
        self.integral_error = 0.0
        self.last_error = 0.0
        self._safety_distance = 0.22

        self._turn_left_mode = False
        self._turn_right_mode = False

        # Variables para el modo callejón sin salida
        self._dead_end_mode = False
        self._rotation_completed = 0.0

        self.last_left = 0.15
        self.last_right = 0.15
        self.last_front = 0.15

        

    def compute_commands(self, z_scan: list[float], z_v: float, z_w: float) -> tuple[float, float]:
        front_distance = z_scan[0]
        left_distance = z_scan[60]
        right_distance = z_scan[-60]

        if math.isnan(front_distance):
            front_distance = self.last_front
        if math.isnan(left_distance):
            left_distance = self.last_left
        if math.isnan(right_distance):
            right_distance = self.last_right

        v = 0.22  # para el real 0.1
        w = 0.0

        if (
            front_distance <= self._safety_distance
            and left_distance <= 0.23
            and right_distance <= 0.23
        ):
            self._dead_end_mode = True
        if (front_distance <= self._safety_distance and not self._dead_end_mode) and self._rotation_completed == 0:
            if right_distance >= left_distance:
                self._turn_right_mode = True
            else:
                self._turn_left_mode = True
            

        if self._turn_right_mode and not self._dead_end_mode and not self._turn_left_mode:
            v = 0.0
            w = -1
            self._rotation_completed += abs(w) * self._dt
            if self._rotation_completed >= math.pi / 2:
                self._turn_right_mode = False
                self.last_error = 0
                self.integral_error = 0
                self._rotation_completed = 0.0

        elif self._turn_left_mode and not self._dead_end_mode and not self._turn_right_mode:
            v = 0.0
            w = 1
            self._rotation_completed += abs(w) * self._dt
            if self._rotation_completed >= math.pi / 2:
                self._turn_left_mode = False
                self.last_error = 0
                self.integral_error = 0
                self._rotation_completed = 0.0

        elif self._dead_end_mode and not self._turn_left_mode and not self._turn_right_mode:
            v = 0.0
            w = 1
            self._rotation_completed += abs(w) * self._dt
            if self._rotation_completed >= math.pi:
                self._dead_end_mode = False
                self.last_error = 0
                self.integral_error = 0
                self._rotation_completed = 0.0

        elif abs(left_distance - right_distance) < 0.2 and left_distance < 0.25 and right_distance < 0.25:
            if right_distance >= left_distance:
                error = left_distance - self._desired_distance
            else:
                error = self._desired_distance - right_distance
            derivative = (error - self.last_error) / self._dt
            self.integral_error += error * self._dt
            # Control PID completo
            w = self.Kp * error + self.Kd * derivative + self.Ki * self.integral_error
            self.last_error = error
        self.last_front = front_distance
        self.last_left = left_distance
        self.last_right = right_distance

 
        return v, w
