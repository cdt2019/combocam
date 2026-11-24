"""YOLOX ONNX 检测器（Apache-2.0，Megvii-BaseDetection/YOLOX release 0.1.1rc0）。

只关心两类 COCO 目标：person(0) 和 sports ball(32)。
预处理与官方 demo/ONNXRuntime/onnx_inference.py 一致：BGR、左上角对齐、灰色(114)填充、不做 mean/std 归一化。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

PERSON = 0
SPORTS_BALL = 32


def _add_cuda_dll_dir() -> None:
    """Windows 上让 onnxruntime-gpu 找到 CUDA 11.8 / cuDNN 8 的 DLL。

    设置 COMBOCAM_CUDA_DLL_DIR 指向含 cudart64_110.dll、cudnn64_8.dll 的目录
    （例如某个 torch cu118 环境的 site-packages/torch/lib）。Linux 云端用官方 CUDA 镜像，不需要这一步。
    """
    d = os.environ.get("COMBOCAM_CUDA_DLL_DIR")
    if d and os.path.isdir(d):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)


_add_cuda_dll_dir()
import onnxruntime as ort  # noqa: E402


def session_options(threads: int = 0) -> "ort.SessionOptions":
    """检测、关键点共用的 ONNX Runtime 会话设置。
    （2026-10-10 在 V100 上试过关掉线程池自旋、限制线程数、让 CUDA 睡眠等待结果，CPU 和用时都没有变化，所以保持默认。）"""
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def r(self) -> float:
        # 运动模糊会把框沿运动方向拉长，取短边更接近真实直径
        return min(self.x2 - self.x1, self.y2 - self.y1) / 2


def gpu_io_enabled(cuda: bool) -> bool:
    """在 GPU 上做输入转换、输出筛选（见 onnx_io.py）：用 CUDA 时默认开，COMBOCAM_GPU_IO=0 关掉（对比、排查用）。"""
    return cuda and os.environ.get("COMBOCAM_GPU_IO", "1") != "0"


class YoloxDetector:
    def __init__(self, model_path: str, device: str = "auto", threads: int = 0):
        from .onnx_io import yolox_u8

        providers: list = ["CPUExecutionProvider"]
        avail = ort.get_available_providers()
        if device in ("auto", "cuda") and "CUDAExecutionProvider" in avail:
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
        elif device == "cuda":
            raise RuntimeError(f"CUDAExecutionProvider 不可用，现有 {avail}")
        so = session_options(threads)
        model = yolox_u8(model_path, (PERSON, SPORTS_BALL)) if gpu_io_enabled(len(providers) > 1) else None
        self.u8 = model is not None  # 输入 uint8 HWC、有候选行输出的版本
        self.sess = ort.InferenceSession(model or model_path, sess_options=so, providers=providers)
        self.device = "cuda" if self.sess.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.out_name = self.sess.get_outputs()[0].name
        self.in_h, self.in_w = (int(inp.shape[1]), int(inp.shape[2])) if self.u8 else (int(inp.shape[2]), int(inp.shape[3]))
        self._grids, self._strides = self._make_grids(self.in_h, self.in_w)

    @staticmethod
    def _make_grids(h: int, w: int):
        grids, strides = [], []
        for s in (8, 16, 32):
            hs, ws = h // s, w // s
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((hs * ws, 1), s))
        return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def _preprocess(self, img: np.ndarray):
        r = min(self.in_h / img.shape[0], self.in_w / img.shape[1])
        nh, nw = int(round(img.shape[0] * r)), int(round(img.shape[1] * r))
        padded = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if self.u8:
            return padded[None], r  # 转 float、换维度顺序在 GPU 上做
        return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32), r

    def prepare(self, img: np.ndarray):
        """预处理（纯 CPU），可以放到别的线程里提前做；结果交给 detect(prepared=...)。"""
        return self._preprocess(img)

    def detect(
        self,
        img: np.ndarray | None,
        classes=(PERSON, SPORTS_BALL),
        score_thr: float = 0.05,
        offset=(0.0, 0.0),
        scale: float = 1.0,
        max_per_class: int = 6,
        prepared=None,
    ) -> list[Det]:
        """img 为 BGR uint8。返回坐标 = (img 坐标 / scale) + offset，用于把裁剪/缩放图的结果映射回画布坐标。
        prepared：prepare(img) 的结果，给了就不再预处理（img 可以传 None）。"""
        from .onnx_io import SEL_IDX, SEL_ROWS, SEL_THR

        blob, r = prepared if prepared is not None else self._preprocess(img)
        if self.u8 and score_thr >= SEL_THR and set(classes) <= {PERSON, SPORTS_BALL}:
            # 只拷回人或球分数够高的候选行（原始输出的超集，见 onnx_io.py），下面照原样解码
            out, idx = self.sess.run([SEL_ROWS, SEL_IDX], {self.input_name: blob})
            grids, strides = self._grids[idx], self._strides[idx]
        else:
            out = self.sess.run([self.out_name], {self.input_name: blob})[0][0]
            grids, strides = self._grids, self._strides
        out[:, :2] = (out[:, :2] + grids) * strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * strides
        dets: list[Det] = []
        for c in classes:
            scores = out[:, 4] * out[:, 5 + c]
            keep = scores >= score_thr
            if not keep.any():
                continue
            b = out[keep, :4]
            s = scores[keep]
            xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / r
            xywh = np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]]
            idx = cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), score_thr, 0.45)
            idx = np.array(idx).reshape(-1)[:max_per_class]
            for i in idx:
                x1, y1, x2, y2 = xyxy[i] / scale
                dets.append(Det(x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1], float(s[i]), c))
        return dets
