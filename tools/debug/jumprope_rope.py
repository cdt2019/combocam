"""跳绳原型：绳子看不看得见？脚附近一条横带里"横贯的运动"有多宽，当作绳子扫过的信号。

  python tools/debug/jumprope_rope.py <名> [--from 帧] [--to 帧]

做法：横带 = 脚踝往上 0.12 个身高到地面往下 0.04 个身高、左右各 0.45 个身高；背景 = 前后 4 帧的逐像素中位数
（绳子转得快，在中位数里会消失）；和背景差得多的像素算"在动"。去掉两只脚所在的那几列（脚自己也在动），
数剩下的列里有多少在动，占比就是绳子信号。绳子贴地扫过时会横贯整条带子，脚动只占一小块。
输出每帧的信号、找出的波峰，和 labels.json 里的跳（如果有）对照。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from combocam.body import CONF  # noqa: E402
from combocam.jumprope import L_FOOT, R_FOOT  # noqa: E402
from combocam.video import FrameReader, canvas_filter, probe  # noqa: E402

OUT = ROOT / "out" / "jumprope"
HALF = 4  # 背景用前后各几帧


def rope_signal(name: str, f_from: int = 0, f_to: int = 10**9) -> tuple[np.ndarray, dict]:
    cache = json.loads((OUT / f"{name}.pose.json").read_text(encoding="utf-8"))
    boxes = cache["persons"]
    idx = [i for i, b in enumerate(boxes) if b]
    arr = np.array([boxes[i] for i in idx])
    full = np.array([np.interp(range(len(boxes)), idx, arr[:, k]) for k in range(4)]).T
    win = int(cache["fps"]) | 1
    cx_s = median_filter((full[:, 0] + full[:, 2]) / 2, size=win, mode="nearest")
    bot_s = median_filter(full[:, 3], size=win, mode="nearest")
    tall = float(np.median(arr[:, 3] - arr[:, 1]))
    half_w, up, down = 0.45 * tall, 0.12 * tall, 0.04 * tall
    src = ROOT / "samples" / "jumprope" / f"{name}.mp4"
    bands, feet_cols = [], []
    for f, frame in enumerate(FrameReader(str(src), canvas_filter(probe(str(src))), 1080, 1920)):
        if f < f_from:
            continue
        if f > f_to:
            break
        x0, y0 = int(round(cx_s[f] - half_w)), int(round(bot_s[f] - up))
        band = np.zeros((int(round(up + down)), int(round(2 * half_w)), 3), np.int16)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(1080, x0 + band.shape[1]), min(1920, y0 + band.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            band[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
        bands.append(band)
        # 两只脚占的列（关键点 x ± 0.06 个身高）
        cols = np.zeros(band.shape[1], bool)
        k = cache["kpts"][f]
        if k:
            a = np.asarray(k, np.float32).reshape(-1, 3)
            for i in L_FOOT + R_FOOT:
                if a[i, 2] >= CONF:
                    c0, c1 = int(a[i, 0] - 0.06 * tall - x0), int(a[i, 0] + 0.06 * tall - x0)
                    cols[max(0, c0):max(0, c1)] = True
        feet_cols.append(cols)
    import cv2

    n = len(bands)
    gray = [cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) for b in bands]
    sig = np.zeros(n)
    for t in range(n):
        h, w = gray[t].shape
        nb = []
        for j in range(max(0, t - HALF), min(n, t + HALF + 1)):
            if j == t:
                continue
            # 手持镜头会晃：先用相位相关求平移，把邻帧对齐到当前帧再取中位数当背景
            (dx, dy), _ = cv2.phaseCorrelate(gray[t], gray[j])
            m = np.float32([[1, 0, -dx], [0, 1, -dy]])
            nb.append(cv2.warpAffine(bands[j].astype(np.uint8), m, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.int16))
        bg = np.median(np.stack(nb), axis=0)
        mov = np.abs(bands[t] - bg).max(axis=2) > 40
        rows = mov.sum(axis=0)
        # 像细线：这一列在动的像素有，但不多（绳子只有几个像素粗；腿、纹理整块在动）
        thin = (rows >= 2) & (rows <= max(3, 0.18 * h))
        free = ~feet_cols[t]
        sig[t] = thin[free].mean() if free.any() else 0.0
    return sig, cache


def main() -> None:
    name = sys.argv[1]
    a = sys.argv
    f_from = int(a[a.index("--from") + 1]) if "--from" in a else 0
    f_to = int(a[a.index("--to") + 1]) if "--to" in a else 10**9
    sig, cache = rope_signal(name, f_from, f_to)
    pk, props = find_peaks(sig, prominence=0.15, distance=4)
    print(f"{name}：绳子信号中位 {np.median(sig):.2f}，波峰 {len(pk)} 个：{[(int(p) + f_from, round(float(sig[p]), 2)) for p in pk]}")
    labels = json.loads((ROOT / "samples" / "jumprope" / "labels.json").read_text(encoding="utf-8"))
    if name in labels:
        print(f"  人工标的跳：{labels[name]['jumps']}")
    # 每帧信号画成一行文字条（0–9 级），方便和帧号对着看
    levels = "".join(str(min(9, int(v * 10))) for v in sig)
    for r in range(0, len(levels), 60):
        print(f"  {r + f_from:4d} {levels[r:r + 60]}")


if __name__ == "__main__":
    main()
