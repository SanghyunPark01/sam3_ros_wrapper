import struct
import numpy as np
import cv2
import copy

# ---------- knobs ----------
MAX_TXT_BYTES = 256 * 1024     # 256KB
MAX_OBJECTS   = 64

IN_SHM_NAME   = "sam3_in_shm"
OUT_SHM_NAME  = "sam3_out_shm"

# ---------- header formats ----------
# IN header: seq,u32 h,w,c, dtype_code, img_bytes, txt_bytes, flags,reserved
# dtype_code: 1=uint8
IN_HDR_FMT  = "<QIIIIIII"
IN_HDR_SIZE = struct.calcsize(IN_HDR_FMT)

# OUT header: seq, h, w, n, mask_bytes, boxes_bytes, scores_bytes, flags
OUT_HDR_FMT  = "<QIIIIIII"
OUT_HDR_SIZE = struct.calcsize(OUT_HDR_FMT)

DTYPE_CODE_UINT8 = 1

def compute_in_shm_size(h: int, w: int, c: int = 3) -> int:
    img_bytes = h * w * c  # uint8
    return IN_HDR_SIZE + img_bytes + MAX_TXT_BYTES

def compute_out_shm_size(h: int, w: int) -> int:
    # mask uint8: MAX_OBJECTS * h*w
    # boxes float32: MAX_OBJECTS * 4 * 4
    # scores float32: MAX_OBJECTS * 4
    mask_bytes  = MAX_OBJECTS * h * w
    boxes_bytes = MAX_OBJECTS * 4 * 4
    scores_bytes= MAX_OBJECTS * 4
    return OUT_HDR_SIZE + mask_bytes + boxes_bytes + scores_bytes

def clamp_n(n: int) -> int:
    return int(min(max(n, 0), MAX_OBJECTS))

def to_uint8_mask(mask_bool_or_u8: np.ndarray) -> np.ndarray:
    # Accept bool or uint8; output contiguous uint8 in {0,1}
    if mask_bool_or_u8.dtype == np.bool_:
        m = mask_bool_or_u8.astype(np.uint8)
    elif mask_bool_or_u8.dtype == np.uint8:
        m = mask_bool_or_u8
    else:
        m = mask_bool_or_u8.astype(np.uint8)
    return np.ascontiguousarray(m)

def to_f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.float32, copy=False))

def _color_for_idx(i: int):
    # OpenCV BGR palette
    palette = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (128, 0, 255), (0, 128, 255), (255, 128, 0),
        (128, 255, 0), (0, 255, 128), (255, 0, 128),
    ]
    return palette[i % len(palette)]

def visualize_sam3_opencv_u8(
    img_rgb_u8__: np.ndarray,
    mask_u8_n1hw__: np.ndarray,   # uint8 [N,1,H,W] values 0/1
    boxes_f32_n4__: np.ndarray,   # float32 [N,4] XYXY in pixels
    scores_f32_n__: np.ndarray,   # float32 [N]
    alpha: float = 0.45,
    thickness: int = 2,
):
    img_rgb_u8 = copy.deepcopy(img_rgb_u8__)
    mask_u8_n1hw = copy.deepcopy(mask_u8_n1hw__)
    boxes_f32_n4 = copy.deepcopy(boxes_f32_n4__)
    scores_f32_n = copy.deepcopy(scores_f32_n__)
    
    if img_rgb_u8.dtype != np.uint8 or img_rgb_u8.ndim != 3 or img_rgb_u8.shape[2] != 3:
        raise ValueError(f"img must be uint8 HxWx3 (RGB). got {img_rgb_u8.shape} {img_rgb_u8.dtype}")

    H, W = img_rgb_u8.shape[:2]

    masks = np.asarray(mask_u8_n1hw, dtype=np.uint8)
    boxes = np.asarray(boxes_f32_n4, dtype=np.float32)
    scores = np.asarray(scores_f32_n, dtype=np.float32).reshape(-1)

    n = min(masks.shape[0], boxes.shape[0], scores.shape[0])

    if n > 0:
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError(f"mask must be [N,1,H,W], got {masks.shape}")
        if masks.shape[2] != H or masks.shape[3] != W:
            raise ValueError(f"mask H,W mismatch: mask {masks.shape[2:]}, img {(H,W)}")
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"boxes must be [N,4], got {boxes.shape}")

    # OpenCV wants BGR
    out = img_rgb_u8 # cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2BGR)
    overlay = out.copy()

    for i in range(n):
        color = _color_for_idx(i)

        m = masks[i, 0]  # (H,W) uint8 0/1
        idx = (m == 1)
        if np.any(idx):
            colored = np.zeros_like(overlay, dtype=np.uint8)
            colored[:] = color
            overlay[idx] = cv2.addWeighted(overlay[idx], 1.0 - alpha, colored[idx], alpha, 0.0)

        x1, y1, x2, y2 = boxes[i]
        x1 = int(round(float(x1))); y1 = int(round(float(y1)))
        x2 = int(round(float(x2))); y2 = int(round(float(y2)))

        # clamp to image bounds
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        prob = float(scores[i])
        label = f"id={i} p={prob:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1, th + 6)
        cv2.rectangle(overlay, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(
            overlay, label, (x1 + 3, ty - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
            2, lineType=cv2.LINE_AA
        )
    return overlay

def printColor(text, color="reset", bold=False, end="\n"):
    colors = {
        "black": 30, "red": 31, "green": 32, "yellow": 33,
        "blue": 34, "magenta": 35, "cyan": 36, "white": 37,
        "reset": 0,
    }
    style = []
    if bold:
        style.append("1")
    style.append(str(colors.get(color, 0)))
    code = ";".join(style)
    print(f"\033[{code}m{text}\033[0m", end=end)