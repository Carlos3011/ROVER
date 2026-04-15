import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import math
from pathlib import Path


class MappingNode(Node):

    def __init__(self):
        super().__init__('mapping_node')

        self.declare_parameter('update_period_s', 0.1)
        self.declare_parameter('image_width_px', 640)
        self.declare_parameter('camera_fov_deg', 60.0)
        self.declare_parameter('assumed_distance_m', 1.0)
        self.declare_parameter('output_path', 'mapa_lunar_oficial.txt')

        self._image_width_px = (
            self.get_parameter('image_width_px').get_parameter_value().integer_value
        )
        self._camera_fov_deg = (
            self.get_parameter('camera_fov_deg').get_parameter_value().double_value
        )
        self._assumed_distance_m = (
            self.get_parameter('assumed_distance_m').get_parameter_value().double_value
        )
        self._output_path = (
            self.get_parameter('output_path').get_parameter_value().string_value
        )

        self.subscription = self.create_subscription(
            String, '/detections', self.detection_callback, 10
        )
        self.sub_imu = self.create_subscription(String, '/terrain_status', self.terrain_cb, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.current_v = 0.0
        self.current_w = 0.0

        self.last_time = self.get_clock().now()
        self.map_rocks = []
        self.map_terrain = []

        self.get_logger().info("Mapping Node iniciado - Exploración Científica")
        update_period_s = (
            self.get_parameter('update_period_s').get_parameter_value().double_value
        )
        self.timer = self.create_timer(update_period_s, self.update_position)

    def update_position(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            return
        self.last_time = now

        self.x += self.current_v * math.cos(self.theta) * dt
        self.y += self.current_v * math.sin(self.theta) * dt
        self.theta += self.current_w * dt

    def terrain_cb(self, msg):
        tipo_terreno = msg.data.lower()
        nuevo_terreno = {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "tipo": tipo_terreno,
        }

        is_new = True
        for t in self.map_terrain:
            if (
                math.hypot(t["x"] - nuevo_terreno["x"], t["y"] - nuevo_terreno["y"])
                < 0.5
            ):
                is_new = False
                break

        if is_new:
            self.map_terrain.append(nuevo_terreno)
            self.get_logger().info(f"Accidente geográfico real registrado: {nuevo_terreno}")

    def detection_callback(self, msg):
        data = msg.data.split(',')
        if len(data) != 4:
            return

        label, cx, color, tamano = data
        if label.strip().lower() != "roca":
            return

        try:
            cx = int(cx)
        except ValueError:
            return

        width = max(int(self._image_width_px), 1)
        fov = float(self._camera_fov_deg)
        angle = (cx - width / 2.0) * (fov / width)
        angle_rad = math.radians(angle)

        distance = float(self._assumed_distance_m)
        rock_x = self.x + distance * math.cos(self.theta + angle_rad)
        rock_y = self.y + distance * math.sin(self.theta + angle_rad)

        rock = {
            "x": round(rock_x, 2),
            "y": round(rock_y, 2),
            "color": color,
            "tamano": tamano,
            "forma": "irregular",
            "textura": "no_determinada",
        }

        is_new = True
        for r in self.map_rocks:
            if math.hypot(r["x"] - rock["x"], r["y"] - rock["y"]) < 0.3:
                is_new = False
                break

        if is_new:
            self.map_rocks.append(rock)
            self.get_logger().info(f"Roca mapeada con LIDAR: {rock}")

    def get_color_marker(self, color_name):
        c = [1.0, 1.0, 1.0] # Default white
        if color_name == "rojo": c = [1.0, 0.0, 0.0]
        elif color_name == "azul": c = [0.0, 0.0, 1.0]
        elif color_name == "verde": c = [0.0, 1.0, 0.0]
        return c

    def publish_markers(self):
        rock_array = MarkerArray()
        for i, r in enumerate(self.map_rocks):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "rocks"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = r["x"]
            m.pose.position.y = r["y"]
            m.pose.position.z = 0.05 # Ligeramente sobre la arena
            m.pose.orientation.w = 1.0
            
            # Escala basada en tamaño si quisiéramos, por defecto 5cm
            escala = 0.05
            if r["tamano"] == "12cm3": escala = 0.12
            elif r["tamano"] == "10cm3": escala = 0.10
            elif r["tamano"] == "7cm3": escala = 0.07

            m.scale.x = escala
            m.scale.y = escala
            m.scale.z = escala
            
            rgbs = self.get_color_marker(r["color"])
            m.color.r = rgbs[0]
            m.color.g = rgbs[1]
            m.color.b = rgbs[2]
            m.color.a = 1.0
            rock_array.markers.append(m)
            
        self.rock_marker_pub.publish(rock_array)

        terr_array = MarkerArray()
        for i, t in enumerate(self.map_terrain):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "terrain"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = t["x"]
            m.pose.position.y = t["y"]
            m.pose.position.z = 0.02
            m.pose.orientation.w = 1.0
            m.scale.x = 0.4
            m.scale.y = 0.4
            m.scale.z = 0.05
            
            # Color por accidente (Amarillo: Surco, Morado: Pendiente, Cyan: Valle)
            if t["tipo"] == "surco":
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0 
            elif t["tipo"] == "pendiente":
                m.color.r, m.color.g, m.color.b = 0.8, 0.0, 0.8
            else:
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 1.0
            
            m.color.a = 0.6  # Translúcido
            terr_array.markers.append(m)
            
        self.terrain_marker_pub.publish(terr_array)

    def _write_map(self):
        output_path = Path(self._output_path)
        if output_path.parent and not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w', encoding='utf-8') as f:
            f.write("--- INVENTARIO DE ROCAS ---\n")
            for rock in self.map_rocks:
                f.write(str(rock) + "\n")
            f.write("\n--- ACCIDENTES GEOGRAFICOS ---\n")
            for terr in self.map_terrain:
                f.write(str(terr) + "\n")

    def destroy_node(self):
        try:
            self._write_map()
        except OSError as exc:
            self.get_logger().warning(f'No se pudo guardar el mapa: {exc}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
