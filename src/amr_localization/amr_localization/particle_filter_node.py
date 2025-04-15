import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

import message_filters
from amr_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import math
import numpy as np  # # ADDED
import os
import time
import traceback
from transforms3d.euler import euler2quat

from amr_localization.particle_filter import ParticleFilter


class ParticleFilterNode(LifecycleNode):
    def __init__(self):
        """Particle filter node initializer."""
        super().__init__("particle_filter")

        # Parameters
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("enable_plot", False)
        self.declare_parameter("global_localization", True)
        self.declare_parameter("initial_pose", (0.0, 0.0, math.radians(0)))
        self.declare_parameter("initial_pose_sigma", (0.05, 0.05, math.radians(5)))
        self.declare_parameter("particles", 1000)
        self.declare_parameter("sigma_v", 0.1)
        self.declare_parameter("sigma_w", 0.1)
        self.declare_parameter("sigma_z", 0.1)
        self.declare_parameter("steps_btw_sense_updates", 25)
        self.declare_parameter("world", "lab03")
        self.declare_parameter("use_ekf_when_localized", True)  # Usar EKF cuando el robot está localizado

        self._pose_publisher = None
        
        # # ADDED: Variables para acumular la odometría
        self._accumulated_distance = 0.0  
        self._accumulated_rotation = 0.0
        self._odom_samples = 0
        self._dt = 0.05  # Valor predeterminado, se actualizará en on_configure

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Node configuration callback. Creates a particle filter object and initializes subscribers.

        Args:
            state: Current lifecycle state.

        Returns:
            Success if configuration is successful, failure otherwise.

        """
        try:
            # Parameters
            self._dt = self.get_parameter("dt").get_parameter_value().double_value
            self._enable_plot = self.get_parameter("enable_plot").get_parameter_value().bool_value
            global_localization = (
                self.get_parameter("global_localization").get_parameter_value().bool_value
            )
            initial_pose = tuple(
                self.get_parameter("initial_pose").get_parameter_value().double_array_value.tolist()
            )
            initial_pose_sigma = tuple(
                self.get_parameter("initial_pose_sigma")
                .get_parameter_value()
                .double_array_value.tolist()
            )
            particles = self.get_parameter("particles").get_parameter_value().integer_value
            sigma_v = self.get_parameter("sigma_v").get_parameter_value().double_value
            sigma_w = self.get_parameter("sigma_w").get_parameter_value().double_value
            sigma_z = self.get_parameter("sigma_z").get_parameter_value().double_value
            self._steps_btw_sense_updates = (
                self.get_parameter("steps_btw_sense_updates").get_parameter_value().integer_value
            )
            world = self.get_parameter("world").get_parameter_value().string_value
            use_ekf = self.get_parameter("use_ekf_when_localized").get_parameter_value().bool_value

            # Attribute and object initializations
            self._localized = False
            self._steps = 0
            map_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "maps", world + ".json")
            )
            self._particle_filter = ParticleFilter(
                self._dt,  # Usar el atributo guardado
                map_path,
                particle_count=particles,
                sigma_v=sigma_v,
                sigma_w=sigma_w,
                sigma_z=sigma_z,
                global_localization=global_localization,
                initial_pose=initial_pose,
                initial_pose_sigma=initial_pose_sigma,
                use_ekf_when_localized=use_ekf
            )

            if self._enable_plot:
                self._particle_filter.show("Initialization", save_figure=True)

            # Publishers
            # TODO: 3.1. Create the /pose publisher (PoseStamped message).
            self._pose_publisher = self.create_publisher(
                msg_type=PoseStamped, 
                topic="/pose", 
                qos_profile=QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.VOLATILE,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=10
                )
            )

            # Subscribers
            scan_qos_profile = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )

            self._subscribers: list[message_filters.Subscriber] = []
            self._subscribers.append(message_filters.Subscriber(self, Odometry, "odometry"))
            self._subscribers.append(
                message_filters.Subscriber(self, LaserScan, "scan", qos_profile=scan_qos_profile)
            )

            ts = message_filters.ApproximateTimeSynchronizer(
                self._subscribers, queue_size=10, slop=9
            )
            ts.registerCallback(self._compute_pose_callback)

            # TODO: 4.12. Add /pose to the synced subscriptions only if localization is enabled.

        except Exception:
            self.get_logger().error(f"{traceback.format_exc()}")
            return TransitionCallbackReturn.ERROR

        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Handles an activating transition.

        Args:
            state: Current lifecycle state.

        """
        self.get_logger().info(f"Transitioning from '{state.label}' to 'active' state.")

        return super().on_activate(state)

    def _compute_pose_callback(self, odom_msg: Odometry, scan_msg: LaserScan):
        """Subscriber callback. Executes a particle filter and publishes (x, y, theta) estimates.

        Args:
            odom_msg: Message containing odometry measurements.
            scan_msg: Message containing LiDAR sensor readings.

        """
        # Parse measurements
        z_v: float = odom_msg.twist.twist.linear.x
        z_w: float = odom_msg.twist.twist.angular.z
        z_scan: list[float] = scan_msg.ranges

        # # ADDED: Acumular odometría
        self._accumulate_odometry(z_v, z_w)
        
        # Verificar si es momento de hacer una actualización completa
        should_update = self._localized or not self._steps % self._steps_btw_sense_updates
        
        x_h, y_h, theta_h = float("inf"), float("inf"), float("inf")
        
        if should_update:
            # Ejecutar actualización integrada de movimiento y medición
            x_h, y_h, theta_h = self._execute_integrated_update(z_scan)
        else:
            # Siempre ejecutar el movimiento independientemente de la visualización
            self._execute_motion_step(z_v, z_w)
            
            # Visualizar solo si está habilitado
            if self._enable_plot:
                self._particle_filter.show("Move", save_figure=True)
        
        self._steps += 1

        # Publish
        self._publish_pose_estimate(x_h, y_h, theta_h)

    def _execute_measurement_step(self, z_us: list[float]) -> tuple[float, float, float]:
        """Executes and monitors the measurement step (sense) of the particle filter.

        Args:
            z_us: Distance from every ultrasonic sensor to the closest obstacle [m].

        Returns:
            Pose estimate (x_h, y_h, theta_h) [m, m, rad]; inf if cannot be computed.
        """
        pose = (float("inf"), float("inf"), float("inf"))

        if self._localized or not self._steps % self._steps_btw_sense_updates:
            start_time = time.perf_counter()
            self._particle_filter.resample(z_us)
            sense_time = time.perf_counter() - start_time

            self.get_logger().info(f"Sense step time: {sense_time:6.3f} s")

            if self._enable_plot:
                self._particle_filter.show("Sense", save_figure=True)

            start_time = time.perf_counter()
            self._localized, pose = self._particle_filter.compute_pose()
            clustering_time = time.perf_counter() - start_time

            self.get_logger().info(f"Clustering time: {clustering_time:6.3f} s")

        return pose

    def _execute_motion_step(self, z_v: float, z_w: float):
        """Executes and monitors the motion step (move) of the particle filter.

        Args:
            z_v: Odometric estimate of the linear velocity of the robot center [m/s].
            z_w: Odometric estimate of the angular velocity of the robot center [rad/s].
        """
        start_time = time.perf_counter()
        self._particle_filter.move(z_v, z_w)
        move_time = time.perf_counter() - start_time

        self.get_logger().info(f"Move step time: {move_time:7.3f} s")

    def _publish_pose_estimate(self, x_h: float, y_h: float, theta_h: float) -> None:
        """Publishes the robot's pose estimate in a custom amr_msgs.msg.PoseStamped message.

        Args:
            x_h: x coordinate estimate [m].
            y_h: y coordinate estimate [m].
            theta_h: Heading estimate [rad].

        """
        # TODO: 3.2. Complete the function body with your code (i.e., replace the pass statement).
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.localized = self._localized

        # Solo incluir pose si está localizado y los valores son válidos
        if self._localized and not (math.isinf(x_h) or math.isinf(y_h) or math.isinf(theta_h)):
            msg.pose.position.x = x_h
            msg.pose.position.y = y_h
            quat = euler2quat(0.0, 0.0, theta_h, "ryxz")
            msg.pose.orientation.w = quat[0]
            msg.pose.orientation.x = quat[1]
            msg.pose.orientation.y = quat[2]
            msg.pose.orientation.z = quat[3]

        self._pose_publisher.publish(msg)

    def _accumulate_odometry(self, z_v: float, z_w: float) -> None:
        """Acumula los datos de odometría entre actualizaciones de remuestreo.
        
        Args:
            z_v: Velocidad lineal [m/s].
            z_w: Velocidad angular [rad/s].
        """
        # # ADDED: Método completo
        # Para la velocidad lineal, acumulamos la distancia recorrida
        # teniendo en cuenta si el robot se mueve hacia adelante o hacia atrás
        self._accumulated_distance += z_v * self._dt  # Distancia = velocidad * tiempo
        
        # Para la velocidad angular, acumulamos la rotación total
        self._accumulated_rotation += z_w * self._dt  # Rotación = velocidad angular * tiempo
        
        # Incrementar contador de muestras
        self._odom_samples += 1
        
    def _execute_integrated_update(self, z_scan: list[float]) -> tuple[float, float, float]:
        """Ejecuta una actualización integrada de movimiento y medición basada en la odometría acumulada.
        
        Args:
            z_scan: Mediciones de distancia del LiDAR.
            
        Returns:
            Pose estimada (x_h, y_h, theta_h) [m, m, rad]; inf si no se puede calcular.
        """
        pose = (float("inf"), float("inf"), float("inf"))
        
        # Verificar que hayamos acumulado muestras
        if self._odom_samples > 0:
            # Calcular velocidades equivalentes para todo el período acumulado
            effective_v = self._accumulated_distance / (self._odom_samples * self._dt)
            effective_w = self._accumulated_rotation / (self._odom_samples * self._dt)
            
            # Usar el método combinado que maneja automáticamente PF y EKF
            start_time = time.perf_counter()
            self._localized, pose = self._particle_filter.move_and_resample(effective_v, effective_w, z_scan)
            total_time = time.perf_counter() - start_time
            
            # Registrar información sobre el tiempo de ejecución
            self.get_logger().info(f"Total update time: {total_time:.6f} s")
            
            # Visualizar solo si está habilitado
            if self._enable_plot:
                self._particle_filter.show("Update", save_figure=True)
                
            # Reiniciar acumuladores para el siguiente período
            self._accumulated_distance = 0.0
            self._accumulated_rotation = 0.0
            self._odom_samples = 0
            self._steps = 0
            
            # Registrar información sobre la localización
            if self._localized:
                self.get_logger().info(f"Robot localizado en: ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})")
                
                # Mostrar método usado (PF o EKF)
                method = "EKF" if self._particle_filter._using_ekf else "Filtro de Partículas"
                self.get_logger().info(f"Método actual: {method}")
        
        return pose


def main(args=None):
    rclpy.init(args=args)
    particle_filter_node = ParticleFilterNode()

    try:
        rclpy.spin(particle_filter_node)
    except KeyboardInterrupt:
        pass

    particle_filter_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
