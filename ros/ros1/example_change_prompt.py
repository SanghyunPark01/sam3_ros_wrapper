#!/usr/bin/env python3
import sys
import rospy
from std_msgs.msg import String
import time

def main():
    rospy.init_node("example_chnage_prompt", argv=sys.argv)

    prompt = rospy.get_param("~prompt", "human")

    pub = rospy.Publisher("/sam3_ros_wrapper/api/input/prompt", String, queue_size=10)

    time.sleep(0.1)
    msg = String(data=f"{prompt}")
    pub.publish(msg)
    rospy.loginfo(f"Published: {msg.data}")

if __name__ == "__main__":
    main()