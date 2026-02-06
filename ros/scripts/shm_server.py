import time
import struct
import numpy as np
from multiprocessing import shared_memory

from shm_common import (
    IN_SHM_NAME, OUT_SHM_NAME,
    IN_HDR_FMT, IN_HDR_SIZE,
    OUT_HDR_FMT, OUT_HDR_SIZE,
    MAX_TXT_BYTES, MAX_OBJECTS,
    clamp_n, to_uint8_mask, to_f32,
)

class Sam3ShmServer:
    def __init__(self):
        self.in_shm = None
        self.out_shm = None

        self.h = None
        self.w = None
        self.c = None

        # output fixed layout once attached and H,W known (same as client)
        self._mask_bytes = None
        self._boxes_bytes = None
        self._scores_bytes = None
        self._mask_off = None
        self._boxes_off = None
        self._scores_off = None

        self.last_seq = 0

    def attach(self):
        # wait for creator (ROS node) to create shm
        while True:
            try:
                self.in_shm = shared_memory.SharedMemory(name=IN_SHM_NAME, create=False)
                break
            except FileNotFoundError:
                time.sleep(0.05)

        while True:
            try:
                self.out_shm = shared_memory.SharedMemory(name=OUT_SHM_NAME, create=False)
                break
            except FileNotFoundError:
                time.sleep(0.05)

    def _init_output_layout(self, h: int, w: int):
        self.h, self.w = int(h), int(w)
        self._mask_bytes   = MAX_OBJECTS * self.h * self.w
        self._boxes_bytes  = MAX_OBJECTS * 4 * 4
        self._scores_bytes = MAX_OBJECTS * 4
        self._mask_off   = OUT_HDR_SIZE
        self._boxes_off  = self._mask_off  + self._mask_bytes
        self._scores_off = self._boxes_off + self._boxes_bytes

    def try_read_input(self):
        # header consistency check
        h1 = bytes(self.in_shm.buf[:IN_HDR_SIZE])
        seq1, H, W, C, dtype_code, img_bytes, txt_bytes, flags1 = struct.unpack(IN_HDR_FMT, h1)
        if seq1 == 0 or seq1 == self.last_seq:
            return None
        if txt_bytes > MAX_TXT_BYTES:
            # writer should prevent this; ignore if corrupted
            return None

        payload_img_off = IN_HDR_SIZE
        payload_txt_off = payload_img_off + img_bytes

        img_view = self.in_shm.buf[payload_img_off:payload_img_off + img_bytes]
        txt_view = self.in_shm.buf[payload_txt_off:payload_txt_off + txt_bytes]

        # snapshot copies
        img = np.frombuffer(img_view, dtype=np.uint8).copy().reshape((H, W, C))
        prompt = bytes(txt_view).decode("utf-8", errors="replace")

        h2 = bytes(self.in_shm.buf[:IN_HDR_SIZE])
        seq2, *_ = struct.unpack(IN_HDR_FMT, h2)
        if seq1 != seq2:
            return None

        if self.h is None:
            self._init_output_layout(H, W)

        self.last_seq = seq2
        return seq2, img, prompt

    def write_output(self, seq: int, mask_u8_n1hw: np.ndarray, boxes_f32_n4: np.ndarray, scores_f32_n: np.ndarray):
        # truncate to MAX_OBJECTS
        n = clamp_n(mask_u8_n1hw.shape[0])

        # allocate fixed slots, write only first n then zero rest
        # mask
        mask_all = np.zeros((MAX_OBJECTS, 1, self.h, self.w), dtype=np.uint8)
        boxes_all = np.zeros((MAX_OBJECTS, 4), dtype=np.float32)
        scores_all = np.zeros((MAX_OBJECTS,), dtype=np.float32)

        mask_all[:n] = mask_u8_n1hw[:n]
        boxes_all[:n] = boxes_f32_n4[:n]
        scores_all[:n] = scores_f32_n[:n]

        # 1) payload write first
        self.out_shm.buf[self._mask_off:self._mask_off + self._mask_bytes] = mask_all.tobytes(order="C")
        self.out_shm.buf[self._boxes_off:self._boxes_off + self._boxes_bytes] = boxes_all.tobytes(order="C")
        self.out_shm.buf[self._scores_off:self._scores_off + self._scores_bytes] = scores_all.tobytes(order="C")

        # 2) header commit last
        flags = 0
        hdr = struct.pack(
            OUT_HDR_FMT,
            int(seq),
            int(self.h),
            int(self.w),
            int(n),
            int(self._mask_bytes),
            int(self._boxes_bytes),
            int(self._scores_bytes),
            int(flags),
        )
        self.out_shm.buf[:OUT_HDR_SIZE] = hdr

    def close(self):
        if self.in_shm is not None:
            self.in_shm.close()
        if self.out_shm is not None:
            self.out_shm.close()
