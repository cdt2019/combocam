"""RTMPose 人体关键点（Apache-2.0，open-mmlab/mmpose；ONNX 来自 rtmposev1 onnx_sdk，Halpe26 版带脚趾、脚跟）。

top-down：给人框，输出 26 个关键点 (x, y, 置信度)。预处理、解码与 mmpose / rtmlib 一致：
框中心不变、四周放大 1.25 倍、补成 192:256 的比例，仿射到 192×256，RGB 按 ImageNet 均值方差归一化；
SimCC 解码：两个方向分别取 argmax，除以 split ratio 2，置信度取两个方向最大值中较小的那个。
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import gpu_io_enabled, ort, session_options  # 复用 detector 里设置好 CUDA DLL 路径的 onnxruntime
from .onnx_io import rtmpose_u8

# Halpe26 关键点编号
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
HEAD, NECK, HIP = 17, 18, 19
L_BIG_TOE, R_BIG_TOE, L_SMALL_TOE, R_SMALL_TOE, L_HEEL, R_HEEL = 20, 21, 22, 23, 24, 25

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


class RTMPose:
    def __init__(self, model_path: str, device: str = "auto"):
        providers: list = ["CPUExecutionProvider"]
        avail = ort.get_available_providers()
        if device in ("auto", "cuda") and "CUDAExecutionProvider" in avail:
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
        so = session_options()
        # 用 CUDA 时把 BGR→RGB、归一化放进模型里在 GPU 上做（见 onnx_io.py），CPU 只做仿射裁剪
        model = rtmpose_u8(model_path, MEAN, STD) if gpu_io_enabled(len(providers) > 1) else None
        self.u8 = model is not None
        self.sess = ort.InferenceSession(model or model_path, sess_options=so, providers=providers)
        self.device = "cuda" if self.sess.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.h, self.w = (int(inp.shape[1]), int(inp.shape[2])) if self.u8 else (int(inp.shape[2]), int(inp.shape[3]))  # 256, 192

    def _affine(self, box) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = box
        c = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
        w, h = (x2 - x1) * 1.25, (y2 - y1) * 1.25
        aspect = self.w / self.h
        if w > h * aspect:
            h = w / aspect
        else:
            w = h * aspect
        s = np.array([w, h], np.float32)
        # 把以 c 为中心、大小 s 的区域映射到 (w, h) 输入
        src = np.array([c - s / 2, [c[0] + s[0] / 2, c[1] - s[1] / 2], [c[0] - s[0] / 2, c[1] + s[1] / 2]], np.float32)
        dst = np.array([[0, 0], [self.w, 0], [0, self.h]], np.float32)
        return cv2.getAffineTransform(src, dst), c, s

    def __call__(self, img: np.ndarray, boxes: list) -> np.ndarray:
        """img 为 BGR uint8；boxes 为 [(x1,y1,x2,y2), ...]。返回 (N, 26, 3)：画布坐标 x, y 和置信度。"""
        if not boxes:
            return np.zeros((0, 26, 3), np.float32)
        blobs, metas = [], []
        for b in boxes:
            m, c, s = self._affine(b)
            crop = cv2.warpAffine(img, m, (self.w, self.h), flags=cv2.INTER_LINEAR)
            if self.u8:
                blobs.append(crop)
            else:
                x = (crop[:, :, ::-1].astype(np.float32) - MEAN) / STD
                blobs.append(x.transpose(2, 0, 1))
            metas.append((c, s))
        sx, sy = self.sess.run(None, {self.input_name: np.stack(blobs)})
        lx, ly = sx.argmax(-1), sy.argmax(-1)
        conf = np.minimum(sx.max(-1), sy.max(-1))
        out = np.zeros((len(boxes), sx.shape[1], 3), np.float32)
        for i, (c, s) in enumerate(metas):
            kx = lx[i] / 2.0 / self.w * s[0] + c[0] - s[0] / 2
            ky = ly[i] / 2.0 / self.h * s[1] + c[1] - s[1] / 2
            out[i] = np.stack([kx, ky, conf[i]], -1)
        return out


def pack_kpts(k: np.ndarray) -> list[int]:
    """26×3 → 扁平整数列表 [x, y, 置信度×100, ...]，存 JSON 省空间；body.Body 读这个格式。"""
    return [int(round(float(v))) for row in k for v in (row[0], row[1], row[2] * 100)]
