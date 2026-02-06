#!/usr/bin/env python3
import sys, os
import numpy as np
import cv2
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
)

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from segmentation_ros_msg.msg import SegmentationResult

from message_filters import Subscriber, ApproximateTimeSynchronizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))             # sam3_ros_wrapper
CONFIG_DIR = os.path.abspath(os.path.join(THIS_DIR, "../../share/sam3_ros_wrapper/config"))
# SCRIPTS_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))       # .../ros/
# if SCRIPTS_DIR not in sys.path:
#     sys.path.insert(0, SCRIPTS_DIR)

from scripts.shm_common import visualize_sam3_opencv_u8


class Sam3VizNode(Node):
    def __init__(self):
        super().__init__("sam3_viz_node")

        with open(os.path.join(CONFIG_DIR, "config.yaml"), "r", encoding="utf-8") as f:
            config__ = yaml.safe_load(f)

        img_sub_topic = config__["sam3_ros_wrapper"]["ros__parameters"]["input_img_topic"]

        res_sub_topic = "/sam3_ros_wrapper/api/output/result"

        img_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        res_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.__m_img_sub = Subscriber(self, Image, img_sub_topic, qos_profile=img_qos)
        self.__m_res_sub = Subscriber(self, SegmentationResult, res_sub_topic, qos_profile=res_qos)

        self.__m_synchronizer = ApproximateTimeSynchronizer(
            [self.__m_img_sub, self.__m_res_sub],
            queue_size=10000,
            slop=0.05,
            allow_headerless=False,
        )
        self.__m_synchronizer.registerCallback(self.__callbackSyncedData)

        self.__m_cv_bridge = CvBridge()

        self.get_logger().info(f"Subscribed image:  {img_sub_topic}")
        self.get_logger().info(f"Subscribed result: {res_sub_topic}")

    def __cvrtRos2Res(self, msg: SegmentationResult):
        N = int(msg.object_num)

        label_u16 = None
        H = W = None
        if N > 0 and msg.label_mask.data:
            label_u16 = self.__m_cv_bridge.imgmsg_to_cv2(msg.label_mask, desired_encoding="passthrough")
            label_u16 = np.asarray(label_u16, dtype=np.uint16)
            H, W = label_u16.shape[:2]

        boxes = np.asarray(msg.boxes, dtype=np.float32)
        if boxes.size == 0:
            boxes_f32 = np.zeros((0, 4), dtype=np.float32)
        else:
            if boxes.size % 4 != 0:
                self.get_logger().warn(f"boxes length is not multiple of 4: {boxes.size}")
            boxes_f32 = boxes.reshape(-1, 4).astype(np.float32)

        scores_f32 = np.asarray(msg.scores, dtype=np.float32).reshape(-1)

        if N > 0:
            boxes_f32 = boxes_f32[:N]
            scores_f32 = scores_f32[:N]
        else:
            boxes_f32 = np.zeros((0, 4), dtype=np.float32)
            scores_f32 = np.zeros((0,), dtype=np.float32)

        if label_u16 is None:
            mask_u8_n1hw = np.zeros((N, 1, 0, 0), dtype=np.uint8)
        else:
            masks_nhw = np.zeros((N, H, W), dtype=np.uint8)
            for i in range(N):
                masks_nhw[i] = (label_u16 == (i + 1)).astype(np.uint8)  # 0/1
            mask_u8_n1hw = masks_nhw[:, None, :, :]

        return {
            "header": msg.header,
            "prompt": msg.prompt,
            "object_num": N,
            "label_u16": label_u16,
            "mask_u8_n1hw": mask_u8_n1hw,
            "boxes_f32": boxes_f32,
            "scores_f32": scores_f32,
        }

    def __callbackSyncedData(self, img_msg: Image, res_msg: SegmentationResult):
        cv_img = self.__m_cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        res = self.__cvrtRos2Res(res_msg)

        vis = visualize_sam3_opencv_u8(cv_img, res["mask_u8_n1hw"], res["boxes_f32"], res["scores_f32"])
        cv2.imshow("subscribe_sam3", vis)
        cv2.waitKey(1)


def main():
    rclpy.init(args=sys.argv)
    node = Sam3VizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
