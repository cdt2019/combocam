"""人工标注辅助：只用检测框和选中的球轨迹（不用触球判定逻辑），列出所有"球的低点"和球丢失的时段，
每处画一行前后几帧的放大图，由人逐个判断是不是触球、落地、手接球。故意多列（阈值很低），宁多勿漏。

  python tools/label/label_helper.py out/heldout/<名>          # 读 <名>.debug.json 和 <名>.track.json（只用来源、尺寸、帧率）
输出 <名>.raw.png（球心 y 随帧变化，灰点=全部候选，蓝点=选中，黑色编号=候选低点，红色阴影=球丢失）
     <名>.lab_<k>.png（每行一个候选低点或丢失时段：前后各若干帧）
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--offs", default="-6,-3,-1,0,1,3,6")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--tile", type=int, default=200)
    ap.add_argument("--prom", type=float, default=0.2, help="低点突出度（球半径的倍数），故意很低")
    ap.add_argument("--gap", type=int, default=4, help="球连续丢失多少帧算一个丢失时段")
    a = ap.parse_args()
    pre = Path(a.prefix)
    dbg = json.loads(pre.with_name(pre.name + ".debug.json").read_text(encoding="utf-8"))
    tr = json.loads(pre.with_name(pre.name + ".track.json").read_text(encoding="utf-8"))
    W, H, n = tr["width"], tr["height"], tr["frames"]
    cands, sel = dbg["cands"], dbg["sel"]
    ys = np.full(n, np.nan)
    xs = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    for f, j in enumerate(sel[:n]):
        if j is not None:
            c = cands[f][j]
            xs[f], ys[f], rs[f] = c[0], c[1], c[2]
    r_med = float(np.nanmedian(rs)) if np.isfinite(rs).any() else 40.0
    ok = np.isfinite(ys)
    idx = np.arange(n)
    yi = np.interp(idx, idx[ok], ys[ok]) if ok.sum() > 1 else np.zeros(n)
    xi = np.interp(idx, idx[ok], xs[ok]) if ok.sum() > 1 else np.full(n, W / 2)
    peaks, _ = find_peaks(yi, prominence=a.prom * r_med, distance=3)
    # 丢失时段
    gaps, s = [], None
    for f in range(n):
        if not ok[f] and s is None:
            s = f
        if ok[f] and s is not None:
            if f - s >= a.gap:
                gaps.append((s, f - 1))
            s = None
    if s is not None and n - s >= a.gap:
        gaps.append((s, n - 1))
    events = [(int(p), f"L{k + 1}", float(xi[p]), float(yi[p])) for k, p in enumerate(peaks)]
    events += [((g0 + g1) // 2, f"G{k + 1} {g0}-{g1}", float(xi[max(g0 - 1, 0)]), float(yi[max(g0 - 1, 0)])) for k, (g0, g1) in enumerate(gaps)]
    events.sort()

    # ---- 轨迹图 ----
    PW, PH, m = 1600, 640, 40
    img = np.full((PH, PW, 3), 255, np.uint8)

    def X(f):
        return int(m + (PW - 2 * m) * f / max(n - 1, 1))

    def Y(y):
        return int(m + (PH - 2 * m) * y / H)

    for g0, g1 in gaps:
        cv2.rectangle(img, (X(g0), m), (X(g1), PH - m), (220, 220, 255), -1)
    for f in range(0, n, 30):
        cv2.line(img, (X(f), m), (X(f), PH - m), (235, 235, 235), 1)
        cv2.putText(img, str(f), (X(f) - 10, PH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    for f, fr in enumerate(cands[:n]):
        for c in fr:
            cv2.circle(img, (X(f), Y(c[1])), 2, (180, 180, 180), -1)
    for f in idx[ok]:
        cv2.circle(img, (X(f), Y(ys[f])), 2, (200, 120, 0), -1)
    for k, p in enumerate(peaks):
        cv2.line(img, (X(p), Y(yi[p]) + 4), (X(p), Y(yi[p]) + 30), (0, 0, 0), 1)
        cv2.putText(img, f"L{k + 1}", (X(p) - 8, Y(yi[p]) + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.imwrite(str(pre.with_name(pre.name + ".raw.png")), img)

    # ---- 逐个候选的放大图 ----
    offs = [int(v) for v in a.offs.split(",")]
    need = sorted({min(max(f + o, 0), n - 1) for f, *_ in events for o in offs} | {f for _, lab, *_ in events if lab.startswith("G") for f in range(int(lab.split()[1].split("-")[0]), int(lab.split()[1].split("-")[1]) + 1, 3)})
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", tr["source"]["path"], "-filter_complex", tr["source"]["filter"],
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], capture_output=True).stdout
    nf = len(raw) // (W * H * 3)
    frames = {f: np.frombuffer(raw, np.uint8)[f * W * H * 3:(f + 1) * W * H * 3].reshape(H, W, 3) for f in need if f < nf}
    crop = int(min(max(9 * r_med, 420), min(W, H)))
    T = a.tile
    rows = []
    for f0, lab, x, y in events:
        if lab.startswith("G"):
            g0, g1 = (int(v) for v in lab.split()[1].split("-"))
            fl = list(range(max(g0 - 2, 0), min(g1 + 3, n), max(1, (g1 - g0 + 5) // (len(offs) - 1))))[:len(offs)]
        else:
            fl = [min(max(f0 + o, 0), n - 1) for o in offs]
        tiles = []
        for f in fl:
            im = frames.get(f)
            if im is None:
                tiles.append(np.full((T, T, 3), 255, np.uint8))
                continue
            im = im.copy()
            if ok[f]:
                cv2.circle(im, (int(xs[f]), int(ys[f])), int(rs[f] * 1.25), (0, 255, 255), 2)
            cx, cy = int(x), int(y + crop * 0.1)
            x0, y0 = min(max(cx - crop // 2, 0), W - crop), min(max(cy - crop // 2, 0), H - crop)
            c = cv2.resize(im[y0:y0 + crop, x0:x0 + crop], (T, T), interpolation=cv2.INTER_AREA)
            cv2.putText(c, str(f), (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if f == f0 and not lab.startswith("G"):
                cv2.rectangle(c, (0, 0), (T - 1, T - 1), (0, 0, 255), 3)
            tiles.append(c)
        while len(tiles) < len(offs):  # 很短的丢失时段，帧不够就补白
            tiles.append(np.full((T, T, 3), 255, np.uint8))
        head = np.full((T, 120, 3), 255, np.uint8)
        cv2.putText(head, lab.split()[0], (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(head, f"@{f0}", (6, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        rows.append(np.hstack([head] + tiles))
    for k in range(0, len(rows), a.rows):
        cv2.imwrite(str(pre.with_name(f"{pre.name}.lab_{k // a.rows + 1}.png")), np.vstack(rows[k:k + a.rows]))
    print(f"{pre.name}: {len(peaks)} 个候选低点，{len(gaps)} 个丢失时段，{(len(rows) + a.rows - 1) // a.rows} 张图；球半径约 {r_med:.0f}px")


if __name__ == "__main__":
    main()
