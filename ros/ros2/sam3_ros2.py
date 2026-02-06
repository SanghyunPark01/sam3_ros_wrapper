#!/usr/bin/env python3

import os
import sys
import copy
import queue
import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

# THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # .../ros/ros2
# SCRIPTS_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))  # .../ros/
# if SCRIPTS_DIR not in sys.path:
#     sys.path.insert(0, SCRIPTS_DIR)

from scripts.shm_client import Sam3ShmClient
from scripts.shm_common import visualize_sam3_opencv_u8, printColor

from segmentation_ros_msg.msg import SegmentationResult

class ROS2Wrapper(Node):
    def __init__(self):
        super().__init__("sam3_ros2_wrapper")

        self.__m_prompt = self.declare_parameter("default_prompt", "/default_prompt").value
        img_topic__ = self.declare_parameter("input_img_topic", "/input_img_topic").value
        print(img_topic__)
        self.__m_mode = self.declare_parameter("mode", "keep_last").value  # keep_last | keep_all
        self.__m_debug_log = self.declare_parameter("debug.log", True).value
        self.__m_debug_vis = self.declare_parameter("debug.vis", False).value
        
        if self.__m_mode not in ("keep_last", "keep_all"):
            self.__m_mode = "keep_last"

        # qos
        img_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        api_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        # img topic
        self.__m_sub_img = self.create_subscription(RosImage, img_topic__, self.__callbackImg, img_qos)

        # API
        self.__m_pub_sam_res = self.create_publisher(SegmentationResult, "/sam3_ros_wrapper/api/output/result", api_qos)
        self.__m_sub_sam_prompt = self.create_subscription(String, "/sam3_ros_wrapper/api/input/prompt", self.__callbackPrompt, api_qos)

        self.__m_img_queue = queue.Queue()
        self.__m_cv_bridge = CvBridge()
        self.m_client = Sam3ShmClient()

        self.__m_stop_evt = threading.Event()
        self.__m_worker = threading.Thread(target=self.__loopThread, daemon=True)

        self.__m_worker.start()

    def __callbackImg(self, img_msg: RosImage):
        if self.__m_mode == "keep_last":
            while self.__m_img_queue.qsize() > 0:
                try:
                    self.__m_img_queue.get_nowait()
                except queue.Empty:
                    break
        self.__m_img_queue.put(img_msg)

    def __callbackPrompt(self, msg: String):
        printColor(
            "[SAM3 ROS2 Wrapper] Change prompt!! {" + self.__m_prompt + " => " + msg.data + "}",
            "yellow",
            1,
        )
        self.__m_prompt = msg.data


    def __cvrtMasks2Label(self, mask_u8_n1hw: np.ndarray,
                           scores_f32_n: np.ndarray) -> np.ndarray:
        masks = mask_u8_n1hw[:, 0].astype(np.uint8)   # [N,H,W]
        scores = scores_f32_n.astype(np.float32)
        N, H, W = masks.shape

        label = np.zeros((H, W), dtype=np.uint16)
        best  = np.full((H, W), -np.inf, dtype=np.float32)

        for i in range(N):
            m = masks[i] != 0
            s = float(scores[i])
            upd = m & (s > best)
            label[upd] = i + 1
            best[upd] = s

        return label

    def __cvrtRes2Ros(self, header, prompt:str, mask_u8_n1hw: np.ndarray,
                         boxes_f32_n4: np.ndarray,
                         scores_f32_n: np.ndarray):
        
        if mask_u8_n1hw.shape[0] != boxes_f32_n4.shape[0] or \
            boxes_f32_n4.shape[0] != scores_f32_n.shape[0] or \
            scores_f32_n.shape[0] != mask_u8_n1hw.shape[0] :
            printColor("[SAM3 ROS1 Wrapper] Object Num is not same... Result is not reliable.", "yellow", True)

        N = int(min(
            mask_u8_n1hw.shape[0],
            boxes_f32_n4.shape[0],
            scores_f32_n.shape[0]
        ))

        if N == 0:
            msg__ = SegmentationResult()
            msg__.header = header
            msg__.prompt = prompt
            msg__.object_num = N

            return msg__

        label_u16 = self.__cvrtMasks2Label(
            mask_u8_n1hw[:N],
            scores_f32_n[:N]
        )

        msg__ = SegmentationResult()
        msg__.header = header
        msg__.prompt = prompt
        msg__.object_num = N

        msg__.label_mask = self.__m_cv_bridge.cv2_to_imgmsg(label_u16, encoding="16UC1")
        msg__.label_mask.header = header
        msg__.boxes = boxes_f32_n4[:N].reshape(-1).astype(np.float32).tolist()
        msg__.scores = scores_f32_n[:N].astype(np.float32).tolist()

        return msg__
    
    def __loopThread(self):
        loop_hz = 200
        loop_dt = 1.0 / max(1e-6, loop_hz)

        pending_seq__ = 0
        pending_header__ = None
        pending_img__ = None
        pending_prompt__ = ""
        
        try:
            printColor("[SAM3 ROS2 Wrapper] Start SAM3 client!", "green", 1)
            while rclpy.ok() and (not self.__m_stop_evt.is_set()):
                time.sleep(loop_dt)

                if pending_seq__ != 0:
                    out = self.m_client.try_read_output(min_seq=pending_seq__)
                    if out is not None:
                        seq, mask, boxes, scores = out
                        pending_seq__ = 0

                        result_ros_msg__ = self.__cvrtRes2Ros(
                            header=pending_header__, prompt=pending_prompt__,
                            mask_u8_n1hw=mask,
                            boxes_f32_n4=boxes,
                            scores_f32_n=scores
                        )

                        self.__m_pub_sam_res.publish(result_ros_msg__)

                        if self.__m_debug_log:
                            print("[SAM3 ROS2 Wrapper] mode: " + self.__m_mode 
                                  + " / prompt: " + pending_prompt__ + " / find " + str(scores.shape[0])
                                  + " object.")
                        
                        if self.__m_debug_vis:
                            vis__ = visualize_sam3_opencv_u8(pending_img__, mask, boxes, scores)
                            cv2.imshow("publish_sam3", vis__)
                            cv2.waitKey(1)

                    continue

                if self.__m_img_queue.qsize() == 0:
                    continue

                img_ros__ = self.__m_img_queue.get()
                pending_header__ = img_ros__.header
                img_cv__ = self.__m_cv_bridge.imgmsg_to_cv2(img_ros__)
                pending_img__ = img_cv__
                pending_prompt__ = copy.deepcopy(self.__m_prompt)
                pending_seq__ = self.m_client.write_input(img_cv__, pending_prompt__)

        finally:
            try:
                self.m_client.close()
            except Exception:
                pass

    def close(self):
        self.__stop_evt.set()
        if self.__worker.is_alive():
            self.__worker.join(timeout=2.0)

def main():
    rclpy.init()
    node = ROS2Wrapper()

    ex = MultiThreadedExecutor()
    ex.add_node(node)

    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()