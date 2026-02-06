import sys, os
import numpy as np
import cv2
import yaml
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters

from segmentation_ros_msg.msg import SegmentationResult

THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../ros/ros1
CONFIG_DIR = os.path.abspath(os.path.join(THIS_DIR, "../config")) # .../ros/config
SCRIPTS_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))  # .../ros/
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
from scripts.shm_common import visualize_sam3_opencv_u8

class Sam3VizNode:
    def __init__(self):
        with open(CONFIG_DIR + "/config.yaml", "r", encoding="utf-8") as f:
            config__ = yaml.safe_load(f)
        img_sub_topic = config__["sam3_ros_wrapper"]["ros__parameters"]["input_img_topic"]
        res_sub_topic = "/sam3_ros_wrapper/api/output/result"
        self.__m_img_sub = message_filters.Subscriber(img_sub_topic, Image)
        self.__m_res_sub = message_filters.Subscriber(res_sub_topic, SegmentationResult)

        self.__m_synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self.__m_img_sub, self.__m_res_sub],
                queue_size=10000,
                slop=0.05,
                allow_headerless=False
            )
        
        self.__m_synchronizer.registerCallback(self.__callbackSyncedData)

        self.__m_cv_bridge = CvBridge()
    
    def __cvrtRos2Res(self, msg: SegmentationResult):
        N = int(msg.object_num)

        label_u16 = None
        H = W = None
        if N > 0 and msg.label_mask.data:
            label_u16 = self.__m_cv_bridge.imgmsg_to_cv2(msg.label_mask, desired_encoding="16UC1")
            label_u16 = np.asarray(label_u16, dtype=np.uint16)
            H, W = label_u16.shape[:2]

        boxes = np.asarray(msg.boxes, dtype=np.float32)
        if boxes.size == 0:
            boxes_f32 = np.zeros((0, 4), dtype=np.float32)
        else:
            if boxes.size % 4 != 0:
                rospy.logwarn("[sam3_viz] boxes length is not multiple of 4: %d", boxes.size)
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
                masks_nhw[i] = (label_u16 == (i + 1)).astype(np.uint8)
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
        cv_img = self.__m_cv_bridge.imgmsg_to_cv2(img_msg)
        res = self.__cvrtRos2Res(res_msg)

        vis = visualize_sam3_opencv_u8(cv_img, res["mask_u8_n1hw"], res["boxes_f32"], res["scores_f32"])
        cv2.imshow("subscribe_sam3", vis)
        cv2.waitKey(1)

    def spin(self):
        rospy.spin()

def main():
    rospy.init_node("sam3_viz_node", argv=sys.argv, anonymous=False)
    node = Sam3VizNode()
    node.spin()

if __name__ == "__main__":
    main()