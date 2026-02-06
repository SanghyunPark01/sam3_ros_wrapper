#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class ExampleChangePrompt(Node):
    def __init__(self):
        super().__init__("example_change_prompt")

        # ros2 params
        self.declare_parameter("prompt", "human")
        prompt = self.get_parameter("prompt").value

        self.pub = self.create_publisher(String, "/sam3_ros_wrapper/api/input/prompt", 10)

        # publish once, then exit shortly after
        msg = String()
        msg.data = str(prompt)
        self.pub.publish(msg)
        self.get_logger().info(f"Published: {msg.data}")

def main():
    rclpy.init()
    node = ExampleChangePrompt()
    rclpy.spin_once(node, timeout_sec=0.1)
    time.sleep(0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
