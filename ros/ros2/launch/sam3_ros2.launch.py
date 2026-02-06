from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory("sam3_ros_wrapper")
    config_yaml = os.path.join(pkg_share, "config", "config.yaml")

    THIS_DIR = os.path.dirname(os.path.abspath(__file__)) # .../ros/ros1/launch

    server_script = os.path.join(THIS_DIR + "/../../../../../src/sam3_ros_wrapper/ros/scripts", "sam3_api_server.py")

    return LaunchDescription([
        Node(
            package="sam3_ros_wrapper",
            executable="sam3_ros2.py",
            name="sam3_ros_wrapper",
            output="screen",
            parameters=[config_yaml],
            prefix=['stdbuf', ' -oL'],
            additional_env={"PYTHONUNBUFFERED": "1"}
        ),
        ExecuteProcess(
            cmd=["python3.12", server_script],
            output="screen",
            additional_env={"PYTHONUNBUFFERED": "1"}
        ),
    ])
