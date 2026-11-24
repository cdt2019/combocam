"""跳绳标注辅助：像素"运动轨迹图"（kymograph）。每帧在人的中线上取一条竖带（宽约 6% 身高，左右平均成一列像素），
按时间从左到右排起来：人每跳一次，头顶、肩膀、脚底的边缘就一起往上拱一下，最高点对应的帧号从下面的刻度读。
只用像素，不用关键点算出来的起伏，标注和计数算法互相独立（人在哪儿、多高用人框定位，1 秒滑动中位数，不跟着跳动走）。

  python tools/label/jumprope_kymo.py 名 [--dir out/jumprope2_mp] [--px 3] [--from 帧] [--to 帧]   → <dir>/<名>.kymo.png

上半张是头和肩（放大），下半张是全身（脚底的边缘也能看出离地）；横向每帧 px 个像素宽，刻度每 10 帧一条、每 30 帧标数字。
最下面两行是从像素估的人的高度，两种办法各一行（A：头肩一段里随时间在动的最强边缘；B：相邻两帧竖带互相关求平移再累加，
静止的背景行不参与），波峰标黄色三角和帧号作为候选。哪种对得上上面的拱形用哪种，人工逐个核对、增删，再写进标注。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from combocam.jumprope_track import main_person  # noqa: E402
from combocam.video import FrameReader  # noqa: E402


def main() -> None:
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    name = a[0]
    d_dir = Path(opt("--dir", str(ROOT / "out" / "jumprope2_mp")))
    d = json.loads((d_dir / f"{name}.pose.json").read_text(encoding="utf-8"))
    px = int(opt("--px", str(max(1, min(5, 1800 // max(1, d["frames"]))))))  # 默认让整张宽约 1800 像素
    boxes, _, _ = main_person(d)
    n = len(boxes)
    f0, f1 = int(opt("--from", "0")), min(n, int(opt("--to", str(n))))
    idx = [i for i, b in enumerate(boxes) if b]
    arr = np.array([boxes[i] for i in idx])
    full = np.array([np.interp(range(n), idx, arr[:, k]) for k in range(4)]).T
    win = int(d["fps"]) | 1
    cx = median_filter((full[:, 0] + full[:, 2]) / 2, size=win, mode="nearest")
    top = median_filter(full[:, 1], size=win, mode="nearest")
    bot = median_filter(full[:, 3], size=win, mode="nearest")
    tall = float(np.median(arr[:, 3] - arr[:, 1]))
    hw = max(2, int(0.03 * tall))
    H_BODY, H_HEAD = 360, 240
    cols_body, cols_head = [], []
    src = next((p / f"{name}.mp4" for p in (ROOT / "samples" / "jumprope", ROOT / "samples" / "jumprope2", ROOT / "samples" / "jumprope3") if (p / f"{name}.mp4").exists()))
    for f, frame in enumerate(FrameReader(str(src), d["filter"], d["width"], d["height"], yuv_canvas=True)):
        if f < f0:
            continue
        if f >= f1:
            break
        x = int(round(cx[f]))
        y0, y1 = int(top[f] - 0.12 * tall), int(bot[f] + 0.06 * tall)
        band = np.full((y1 - y0, 2 * hw, 3), 0, np.uint8)
        sx0, sx1 = max(0, x - hw), min(frame.shape[1], x + hw)
        sy0, sy1 = max(0, y0), min(frame.shape[0], y1)
        if sx1 > sx0 and sy1 > sy0:
            band[sy0 - y0:sy1 - y0, sx0 - (x - hw):sx1 - (x - hw)] = frame[sy0:sy1, sx0:sx1]
        col = band.mean(axis=1).astype(np.uint8)  # (高, 3)
        cols_body.append(cv2.resize(col[:, None, :], (1, H_BODY), interpolation=cv2.INTER_AREA))
        head = col[: int(0.42 * len(col))]  # 头和肩
        cols_head.append(cv2.resize(head[:, None, :], (1, H_HEAD), interpolation=cv2.INTER_LINEAR))
    fps = d["fps"]

    def peaks_of(sig: np.ndarray, min_prom: float) -> list[int]:
        q, _ = find_peaks(np.convolve(sig, np.ones(3) / 3, mode="same"), distance=max(2, int(0.18 * fps)),
                          prominence=max(min_prom, 0.4 * np.std(sig)))
        return q.tolist()

    def detrend(y: np.ndarray) -> np.ndarray:
        return y - median_filter(y, size=int(fps) | 1, mode="nearest")

    # 候选 A（边缘）：头和肩那段灰度竖向梯度减去每行随时间的中位数（静止背景边缘减掉），最强几条动边缘的平均行号
    hs = np.concatenate(cols_head, axis=1).astype(np.float32).mean(axis=2)
    g = np.clip(np.abs(np.diff(hs, axis=0)) - np.median(np.abs(np.diff(hs, axis=0)), axis=1, keepdims=True), 0, None)
    g = cv2.GaussianBlur(g, (1, 5), 0)
    top_k = np.argsort(-g, axis=0)[:4]
    gk = np.take_along_axis(g, top_k, 0)
    yv = (top_k * gk).sum(0) / np.maximum(1e-6, gk.sum(0))
    sig_a = -detrend(median_filter(yv, size=3, mode="nearest"))
    # 候选 B（平移）：相邻两帧全身竖带互相关求竖向平移、累加；按行随时间的标准差加权，静止的背景行不参与
    stack = np.concatenate(cols_body, axis=1).astype(np.float32).mean(axis=2)
    m = stack.shape[1]
    wrow = stack.std(axis=1)
    wrow = wrow / (wrow.max() + 1e-6)
    R = 6
    step = np.zeros(m)
    for t in range(1, m):
        a_, b_ = stack[:, t - 1], stack[:, t]
        best, bs = -1e18, 0
        for s_ in range(-R, R + 1):  # b_[y] ≈ a_[y + s_]：s_ > 0 表示人往上移
            ya = slice(max(0, s_), len(a_) + min(0, s_))
            yb = slice(max(0, -s_), len(b_) + min(0, -s_))
            ww = wrow[yb]
            sc = -np.sum(ww * (b_[yb] - a_[ya]) ** 2) / max(1e-6, ww.sum())
            if sc > best:
                best, bs = sc, s_
        step[t] = bs
    sig_b = detrend(np.cumsum(step))
    cand = {"A": peaks_of(sig_a, 0.5), "B": peaks_of(sig_b, 1.0)}
    body = np.repeat(np.concatenate(cols_body, axis=1), px, axis=1)
    head = np.repeat(np.concatenate(cols_head, axis=1), px, axis=1)
    w = body.shape[1]
    ruler = np.full((26, w, 3), 255, np.uint8)
    for f in range(f0, f0 + len(cols_body)):
        x = (f - f0) * px
        if f % 10 == 0:
            for im in (head, body):
                im[:, x] = (im[:, x] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8) if f % 30 == 0 else (im[:, x] * 0.7).astype(np.uint8)
            cv2.line(ruler, (x, 0), (x, 8 if f % 30 else 14), (0, 0, 0), 1)
            if f % 30 == 0:
                cv2.putText(ruler, str(f), (x + 2, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    # 两种候选各一行：曲线 + 波峰（黄色三角、帧号）
    plots = []
    for key, sig in (("A", sig_a), ("B", sig_b)):
        plot = np.full((62, w, 3), 255, np.uint8)
        sc_ = 22 / max(1.0, np.percentile(np.abs(sig), 98))
        pts = np.array([[int((q + 0.5) * px), int(30 - sig[q] * sc_)] for q in range(m)], np.int32)
        cv2.polylines(plot, [pts], False, (120, 60, 0), 1, cv2.LINE_AA)
        cv2.putText(plot, key, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        for q in cand[key]:
            x = int((q + 0.5) * px)
            cv2.drawMarker(plot, (x, int(30 - sig[q] * sc_) - 4), (0, 170, 255), cv2.MARKER_TRIANGLE_DOWN, 8, 2)
            cv2.putText(plot, str(f0 + q), (x - 8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 180), 1, cv2.LINE_AA)
        plots.append(plot)
    img = np.vstack([head, np.full((3, w, 3), 255, np.uint8), body, ruler, *plots])
    out = d_dir / f"{name}.kymo.png"
    cv2.imwrite(str(out), img)
    print(out, img.shape)
    out_c = {k: [int(f0 + q) for q in v] for k, v in cand.items()}
    print("候选 A", len(out_c["A"]), out_c["A"])
    print("候选 B", len(out_c["B"]), out_c["B"])
    (d_dir / f"{name}.kymo.json").write_text(json.dumps({**out_c, "px": px}), encoding="utf-8")


if __name__ == "__main__":
    main()
