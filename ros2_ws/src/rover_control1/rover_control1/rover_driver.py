import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import serial
import time

class HardwareBridge(Node):
    def __init__(self):
        super().__init__('hardware_bridge')
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.05)
            self.get_logger().info("Conectado a la ESP32 (Motores + Servos + IMU)")
        except:
            self.ser = None
            self.get_logger().warn("ESP32 no detectada")

        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.arm_sub = self.create_subscription(String, '/arm_cmd', self.arm_callback, 10)
        
        # Publicador de terrenos recuperado
        self.terrain_pub = self.create_publisher(String, '/terrain_status', 10)
        
        self.timer = self.create_timer(0.05, self.read_serial)

    def cmd_callback(self, msg):
        linear = max(min(msg.linear.x, 1.0), -1.0)
        angular = max(min(msg.angular.z, 1.0), -1.0)
        data = f"W,{linear:.2f},{angular:.2f}\n"
        self._send_serial(data)

    def arm_callback(self, msg):
        data = f"A,{msg.data}\n"
        self._send_serial(data)

    def read_serial(self):
        if not self.ser: return
        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                if line.startswith("I,"):
                    parts = line.split(',')
                    if len(parts) == 3:
                        pitch = float(parts[1])
                        roll = float(parts[2])
                        self.analyze_terrain(pitch, roll)
        except Exception as e:
            pass

    def analyze_terrain(self, pitch, roll):
        # Lógica científica para la convocatoria TMR
        terreno = None
        
        if abs(pitch) > 18.0:
            terreno = "pendiente"
        elif abs(roll) > 12.0:
            terreno = "valle"
        # Si la variación es brusca (requeriría guardar el historial), es un surco.
        # Por simplicidad para el TMR, usamos ángulos combinados
        elif abs(pitch) > 10.0 and abs(roll) > 10.0:
            terreno = "surco"

        if terreno:
            msg = String()
            msg.data = terreno
            self.terrain_pub.publish(msg)

    def _send_serial(self, data):
        if self.ser:
            try:
                self.ser.write(data.encode())
            except Exception as e:
                pass

    def destroy_node(self):
        if self.ser: self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HardwareBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
