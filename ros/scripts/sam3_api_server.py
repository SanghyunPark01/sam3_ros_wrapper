#!/usr/bin/env python3.12

import sys, os
import time
import yaml

import cv2

import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../ros/scripts
CONFIG_DIR = os.path.abspath(os.path.join(THIS_DIR, "../config")) # .../ros/config
PKG_DIR = os.path.abspath(os.path.join(THIS_DIR, "../.."))  # .../ros/sam3_ros_wrapper
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from shm_server import Sam3ShmServer
from shm_common import (
    IN_SHM_NAME, OUT_SHM_NAME,
    IN_HDR_FMT, IN_HDR_SIZE,
    OUT_HDR_FMT, OUT_HDR_SIZE,
    MAX_TXT_BYTES, MAX_OBJECTS,
    clamp_n, to_uint8_mask, to_f32,
    printColor
)

printColor("Waiting SAM3 LIBRARY..", "yellow", 1)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3:
    def __init__(self, checkpoint_path = ""):
        sam3_model__ = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False)
        self.__m_sam3_processor = Sam3Processor(sam3_model__)

    def inference(self, img, prompt):
        inference_state__ = self.__m_sam3_processor.set_image(img)
        output__ = self.__m_sam3_processor.set_text_prompt(state=inference_state__, prompt=prompt)

        masks, boxes, scores = output__["masks"], output__["boxes"], output__["scores"]

        return masks, boxes, scores

def run_server():
    with open(CONFIG_DIR + "/config.yaml", "r", encoding="utf-8") as f:
        config__ = yaml.safe_load(f)
    checkpoint_path__ = config__["sam3_ros_wrapper"]["ros__parameters"]["weight_path"]

    printColor("[SAM3 Server] Initializing...", "yellow", 1)
    sam3_infer__ = SAM3(checkpoint_path__)
    srv = Sam3ShmServer()
    
    printColor("[SAM3 Server] Wait for initial image...", "yellow", 1)

    srv.attach()
    printColor("[SAM3 Server] Initialize Success!", "green", 1)

    try:
        while True:
            item = srv.try_read_input()
            if item is None:
                time.sleep(0.002)
                continue

            seq, img, prompt = item
            img_pil__ = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            t_mask, t_boxes, t_scores = sam3_infer__.inference(img_pil__, prompt)

            mask_u8 = to_uint8_mask(t_mask.detach().cpu().numpy())
            boxes_f32 = to_f32(t_boxes.detach().cpu().numpy())
            scores_f32= to_f32(t_scores.detach().cpu().numpy())

            srv.write_output(seq, mask_u8, boxes_f32, scores_f32)

    finally:
        srv.close()
    pass

if __name__ == "__main__":
    run_server()