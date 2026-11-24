"""核对"在 GPU 上做输入转换、输出筛选"（combocam/onnx_io.py）和原来逐位相同，并比较每帧用时。要在有 CUDA 的机器上跑。

  python tools/debug/check_onnx_io.py samples/bench_30s.mp4 [帧数，默认 300]

逐帧比较：检测模型原始输出、检测框（整帧人 + 球，和在线复检用的 640×640 裁剪只找球）、关键点模型原始输出和关键点。
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from combocam.detector import PERSON, SPORTS_BALL, YoloxDetector  # noqa: E402
from combocam.pose import RTMPose  # noqa: E402
from combocam.video import FrameReader, canvas_filter, probe  # noqa: E402


def build(gpu_io: bool):
    os.environ["COMBOCAM_GPU_IO"] = "1" if gpu_io else "0"
    det = YoloxDetector(str(ROOT / "models/yolox_s.onnx"), "cuda")
    pose = RTMPose(str(ROOT / "models/rtmpose-m_halpe26_256x192.onnx"), "cuda")
    return det, pose


def main() -> None:
    src = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    old_det, old_pose = build(False)
    new_det, new_pose = build(True)
    assert new_det.u8 and new_pose.u8 and not old_det.u8, "新路径没启用（缺 onnx 包？）"
    p = probe(src)
    bad = {"检测原始输出": 0, "检测框": 0, "裁剪检测框": 0, "关键点原始输出": 0, "关键点": 0}
    T = {"旧 检测": 0.0, "新 检测": 0.0, "旧 关键点": 0.0, "新 关键点": 0.0}
    k = 0
    for f, frame in enumerate(FrameReader(src, canvas_filter(p), 1080, 1920)):
        if f >= n:
            break
        k += 1
        # 原始输出：新模型的 output（经过图里的 Cast + Transpose）和旧模型逐位比
        ob, _ = old_det._preprocess(frame)
        nb, _ = new_det._preprocess(frame)
        o1 = old_det.sess.run([old_det.out_name], {old_det.input_name: ob})[0]
        o2 = new_det.sess.run([new_det.out_name], {new_det.input_name: nb})[0]
        bad["检测原始输出"] += not np.array_equal(o1, o2)
        a = time.perf_counter()
        d1 = old_det.detect(frame, (PERSON, SPORTS_BALL), 0.05)
        T["旧 检测"] += time.perf_counter() - a
        a = time.perf_counter()
        d2 = new_det.detect(frame, (PERSON, SPORTS_BALL), 0.05)
        T["新 检测"] += time.perf_counter() - a
        bad["检测框"] += d1 != d2
        crop = frame[600:1240, 220:860]
        bad["裁剪检测框"] += old_det.detect(crop, (SPORTS_BALL,), 0.05, offset=(220, 600)) != new_det.detect(crop, (SPORTS_BALL,), 0.05, offset=(220, 600))
        persons = [d for d in d1 if d.cls == PERSON and d.score >= 0.3]
        if persons:
            box = [(d.x1, d.y1, d.x2, d.y2) for d in persons[:1]]
            m, _, _ = old_pose._affine(box[0])
            import cv2

            crop_p = cv2.warpAffine(frame, m, (old_pose.w, old_pose.h), flags=cv2.INTER_LINEAR)
            from combocam.pose import MEAN, STD

            x = ((crop_p[:, :, ::-1].astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
            r1 = old_pose.sess.run(None, {old_pose.input_name: x})
            r2 = new_pose.sess.run(None, {new_pose.input_name: crop_p[None]})
            bad["关键点原始输出"] += not all(np.array_equal(u, v) for u, v in zip(r1, r2))
            a = time.perf_counter()
            k1 = old_pose(frame, box)
            T["旧 关键点"] += time.perf_counter() - a
            a = time.perf_counter()
            k2 = new_pose(frame, box)
            T["新 关键点"] += time.perf_counter() - a
            bad["关键点"] += not np.array_equal(k1, k2)
    print(f"{k} 帧，不一致的帧数：" + "，".join(f"{name} {v}" for name, v in bad.items()))
    print("每帧用时：" + "，".join(f"{name} {v / k * 1000:.2f} ms" for name, v in T.items()))
    sys.exit(1 if any(bad.values()) else 0)


if __name__ == "__main__":
    main()
