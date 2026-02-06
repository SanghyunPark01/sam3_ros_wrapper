import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


image_path = "/workspace/python_ws/sam3/sam3/for_test/img_2.png"

# Load the model
model = build_sam3_image_model(checkpoint_path= "/workspace/ros_workspace/sam_ws/src/sam3_ros_wrapper/weight/sam3.pt", load_from_HF=False)
processor = Sam3Processor(model)
# Load an image
# image_path = "./test_img.png"
image = Image.open(image_path)

t0 = time.time()

inference_state = processor.set_image(image)
# Prompt the model with text
# output = processor.set_text_prompt(state=inference_state, prompt="hay on the left and biggest")
output = processor.set_text_prompt(state=inference_state, prompt="fence")

t1 = time.time()

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print(type(masks), type(boxes), type(scores))

print(masks.size(), boxes.size(), scores.size())
print(masks.dtype, boxes.dtype, scores.dtype)

# visualize
print(t1 - t0)
plot_results(image, output)

# mask_numpy = masks[0].squeeze(0).cpu().numpy()
# gray = np.where(mask_numpy, 255, 0).astype(np.uint8)

# cv.imwrite("mask.png", gray)

plt.show()