from launch
import LaunchDescription
from launch_ros.actions
import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='rover_vision', executable='vision_node', output='screen'),
        Node(package='rover_mission', executable='mission_node', output='screen'),
        Node(package='rover_mapping', executable='mapping_node', output='screen'),
        Node(package='rover_control1', executable='rover_driver', output='screen'),
        Node(package='rover_monitor', executable='monitor_node', output='screen'),

        Node(
            package='sllidar_ros2',
            executable='sllidar_node',
            name='sllidar_node',
            parameters=[{'channel_type': 'serial',
                         'serial_port': '/dev/ttyUSB1', #cambiar puerto al correspondiente
                         'serial_baudrate': 115200,
                         'frame_id': 'laser',
                         'inverted': False,
                         'angle_compensate': True}],
            output='screen'
        ),
    ])
