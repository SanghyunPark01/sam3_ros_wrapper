import time
import struct
import numpy as np
from multiprocessing import shared_memory

from .shm_common import (
    IN_SHM_NAME, OUT_SHM_NAME,
    IN_HDR_FMT, IN_HDR_SIZE,
    OUT_HDR_FMT, OUT_HDR_SIZE,
    DTYPE_CODE_UINT8,
    MAX_TXT_BYTES, MAX_OBJECTS,
    compute_in_shm_size, compute_out_shm_size,
)


class Sam3ShmClient:
    def __init__(self):
        self.in_shm = None
        self.out_shm = None
        self.h = None
        self.w = None
        self.c = 3
        self.seq = 0

        # offsets/sizes for output payload (fixed once H,W known)
        self._mask_bytes = None
        self._boxes_bytes = None
        self._scores_bytes = None
        self._mask_off = None
        self._boxes_off = None
        self._scores_off = None

    def _unlink_if_exists(self, name: str):
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

    def init_from_first_image(self, img_u8_hwc: np.ndarray):
        assert img_u8_hwc.dtype == np.uint8 and img_u8_hwc.ndim == 3 and img_u8_hwc.shape[2] == 3
        self.h, self.w, self.c = map(int, img_u8_hwc.shape)

        # remove shm
        self._unlink_if_exists(IN_SHM_NAME)
        self._unlink_if_exists(OUT_SHM_NAME)

        in_size  = compute_in_shm_size(self.h, self.w, self.c)
        out_size = compute_out_shm_size(self.h, self.w)

        self.in_shm  = shared_memory.SharedMemory(name=IN_SHM_NAME,  create=True, size=in_size)
        self.out_shm = shared_memory.SharedMemory(name=OUT_SHM_NAME, create=True, size=out_size)

        # output payload layout (fixed):
        self._mask_bytes   = MAX_OBJECTS * self.h * self.w
        self._boxes_bytes  = MAX_OBJECTS * 4 * 4
        self._scores_bytes = MAX_OBJECTS * 4

        self._mask_off   = OUT_HDR_SIZE
        self._boxes_off  = self._mask_off  + self._mask_bytes
        self._scores_off = self._boxes_off + self._boxes_bytes

        # initial header = 0
        self.in_shm.buf[:IN_HDR_SIZE] = b"\x00" * IN_HDR_SIZE
        self.out_shm.buf[:OUT_HDR_SIZE] = b"\x00" * OUT_HDR_SIZE

    def write_input(self, img_u8_hwc: np.ndarray, prompt: str) -> int:
        if self.in_shm is None:
            self.init_from_first_image(img_u8_hwc)

        if img_u8_hwc.shape != (self.h, self.w, self.c) or img_u8_hwc.dtype != np.uint8:
            raise ValueError("Input image shape/dtype changed. Need re-init policy (unlink+recreate).")

        txt_b = prompt.encode("utf-8")
        if len(txt_b) > MAX_TXT_BYTES:
            raise ValueError(f"prompt too large: {len(txt_b)} bytes > {MAX_TXT_BYTES}")

        img_bytes = self.h * self.w * self.c
        payload_img_off = IN_HDR_SIZE
        payload_txt_off = payload_img_off + img_bytes

        self.seq += 1
        seq = self.seq

        # 1) payload write (img then text)
        self.in_shm.buf[payload_img_off:payload_img_off + img_bytes] = img_u8_hwc.tobytes(order="C")
        self.in_shm.buf[payload_txt_off:payload_txt_off + len(txt_b)] = txt_b

        # (optional) clear rest of txt area? not necessary if txt_bytes provided.

        # 2) header commit last
        # fields: seq,h,w,c,dtype_code,img_bytes,txt_bytes,flags
        flags = 0
        hdr = struct.pack(IN_HDR_FMT, seq, self.h, self.w, self.c, DTYPE_CODE_UINT8, img_bytes, len(txt_b), flags)
        self.in_shm.buf[:IN_HDR_SIZE] = hdr
        return seq

    def try_read_output(self, min_seq: int, copy: bool = True):
        if self.out_shm is None:
            return None

        # read header twice for consistency
        h1 = bytes(self.out_shm.buf[:OUT_HDR_SIZE])
        seq1, H, W, n, mask_b, boxes_b, scores_b, flags1 = struct.unpack(OUT_HDR_FMT, h1)
        if seq1 < min_seq or seq1 == 0:
            return None
        if H != self.h or W != self.w:
            return None

        # payload fixed slots (MAX_OBJECTS); but actual n may be <= MAX_OBJECTS
        # read payload snapshot
        mask_view   = self.out_shm.buf[self._mask_off:self._mask_off + self._mask_bytes]
        boxes_view  = self.out_shm.buf[self._boxes_off:self._boxes_off + self._boxes_bytes]
        scores_view = self.out_shm.buf[self._scores_off:self._scores_off + self._scores_bytes]

        if copy:
            mask_all   = np.frombuffer(mask_view, dtype=np.uint8).copy().reshape((MAX_OBJECTS, 1, self.h, self.w))
            boxes_all  = np.frombuffer(boxes_view, dtype=np.float32).copy().reshape((MAX_OBJECTS, 4))
            scores_all = np.frombuffer(scores_view, dtype=np.float32).copy().reshape((MAX_OBJECTS,))
        else:
            mask_all   = np.frombuffer(mask_view, dtype=np.uint8).reshape((MAX_OBJECTS, 1, self.h, self.w))
            boxes_all  = np.frombuffer(boxes_view, dtype=np.float32).reshape((MAX_OBJECTS, 4))
            scores_all = np.frombuffer(scores_view, dtype=np.float32).reshape((MAX_OBJECTS,))

        h2 = bytes(self.out_shm.buf[:OUT_HDR_SIZE])
        seq2, *_ = struct.unpack(OUT_HDR_FMT, h2)
        if seq1 != seq2:
            return None

        n = int(min(n, MAX_OBJECTS))
        mask  = mask_all[:n]
        boxes = boxes_all[:n]
        scores= scores_all[:n]
        return seq2, mask, boxes, scores

    def close(self):
        for shm in (self.in_shm, self.out_shm):
            if shm is not None:
                shm.close()
        # unlink at ROS node
        if self.in_shm is not None:
            self.in_shm.unlink()
        if self.out_shm is not None:
            self.out_shm.unlink()