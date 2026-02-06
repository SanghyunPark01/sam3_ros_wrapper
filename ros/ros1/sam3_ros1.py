#!/usr/bin/env python3

import sys
import os
import queue
import copy

from PIL import Image as pilImage
import cv2
import numpy as np

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image as rosImage
from cv_bridge import CvBridge
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../ros/ros1
SCRIPTS_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))  # .../ros/
SERVER_PATH = os.path.abspath(os.path.join(THIS_DIR, "./launch")) + "/run_server.sh"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scripts.shm_client import Sam3ShmClient
from scripts.shm_common import visualize_sam3_opencv_u8, printColor

from segmentation_ros_msg.msg import SegmentationResult

class ROS1Wrapper:
    def __init__(self):
        # param
        self.__m_prompt = rospy.get_param("/sam3_ros_wrapper/ros__parameters/default_prompt", "/default_prompt")
        img_topic__ = rospy.get_param("/sam3_ros_wrapper/ros__parameters/input_img_topic", "/input_img_topic")
        self.__m_mode = rospy.get_param("/sam3_ros_wrapper/ros__parameters/mode", "keep_last")
        self.__m_debug_log = rospy.get_param("/sam3_ros_wrapper/ros__parameters/debug/log", True)
        self.__m_debug_vis = rospy.get_param("/sam3_ros_wrapper/ros__parameters/debug/vis", False)

        if self.__m_mode != "keep_last" and self.__m_mode != "keep_all":
            self.__m_mode = "keep_last"

        # img topic
        self.__m_sub_img = rospy.Subscriber(img_topic__, rosImage, self.__callbackImg, queue_size=10)

        # API
        self.__m_pub_sam_res = rospy.Publisher("/sam3_ros_wrapper/api/output/result", SegmentationResult, queue_size=10)
        self.__m_sub_sam_prompt = rospy.Subscriber("/sam3_ros_wrapper/api/input/prompt", String, self.__callbackPrompt, queue_size=10)

        self.__m_img_queue = queue.Queue()
        self.__m_cv_bridge = CvBridge()

        # init client
        self.m_client = Sam3ShmClient()

    def __callbackImg(self, img_msg):
        if self.__m_mode == "keep_last":
            while self.__m_img_queue.qsize() > 0:
                self.__m_img_queue.get_nowait()
        self.__m_img_queue.put(img_msg)

    def __callbackPrompt(self, msg):
        printColor("[SAM3 ROS1 Wrapper] Change prompt!! {" + self.__m_prompt + " => " + msg.data + "}", "yellow", 1)
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
        print(msg__.boxes)
        msg__.scores = scores_f32_n[:N].astype(np.float32).tolist()

        return msg__
    
    def loop(self):
        rate_param__ = 200
        rate__ = rospy.Rate(rate_param__)
        printColor("[SAM3 ROS1 Wrapper] Start SAM3 client!", "green", 1)

        pending_seq__ = 0
        pending_header__ = None
        pending_img__ = None
        pending_prompt__ = ""
        try:
            while not rospy.is_shutdown():
                rate__.sleep()

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
                        
                        # if result_ros_msg__ is not None:
                        self.__m_pub_sam_res.publish(result_ros_msg__)

                        if self.__m_debug_log:
                            print("[SAM3 ROS1 Wrapper] mode: " + self.__m_mode 
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
            self.m_client.close()

def main():
    subprocess.Popen(["bash", SERVER_PATH])

    rospy.init_node("sam3_ros1_wrapper", anonymous=False)
    node = ROS1Wrapper()
    node.loop()

if __name__ == "__main__":
    main()