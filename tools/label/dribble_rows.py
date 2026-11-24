"""运球动作标注辅助：每一下运球一行小图（触地前后各 10 帧，隔 2 帧一张），用来人工标单手 / 交叉 / 胯下（samples/dribble/moves.json）。

  python tools/label/dribble_rows.py 名 帧,帧,... [0.3,1.12 | full]
    → out/moves/<名>.rows_<k>.png（按这一帧的人框裁下半身，竖向取人框高度的比例）或 .full_<k>.png（整个真实画面），每张最多 10 行
看不清球是从腿前面还是后面过去的，用 tools/label/dribble_zoom.py 放大那几帧。
"""
import json
import sys
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from combocam.video import FrameReader  # noqa: E402
name, frames = sys.argv[1], [int(v) for v in sys.argv[2].split(",")]
arg3 = sys.argv[3] if len(sys.argv) > 3 else "0.3,1.12"
full = arg3 == "full"
band = [0, 0] if full else [float(v) for v in arg3.split(",")]
d = json.load(open(ROOT / "out" / "dribble" / f"{name}.ball.json", encoding="utf-8"))
n = d["frames"]
bx = np.array([b if b else [np.nan] * 4 for b in d["persons"]], np.float64).reshape(n, 4)
sm = np.array([np.nanmedian(bx[max(0, f - 5):f + 6], axis=0) if np.isfinite(bx[max(0, f - 5):f + 6, 0]).any() else [np.nan] * 4 for f in range(n)])
want = {}
for i, f0 in enumerate(frames):
    for c, g in enumerate(range(f0 - 10, f0 + 11, 2)):
        if 0 <= g < n:
            want.setdefault(g, []).append((i, c))
TW, TH = (240, 135) if full else (150, 190)
rows = [[np.full((TH, TW, 3), 255, np.uint8) for _ in range(11)] for _ in frames]
tag = "full" if full else "rows"
src = next(p / f"{name}.mp4" for p in (ROOT / "samples" / "dribble", ROOT / "samples" / "dribble2", ROOT / "samples" / "dribble3") if (p / f"{name}.mp4").exists())
(ROOT / "out" / "moves").mkdir(parents=True, exist_ok=True)
last = max(want)
for f, fr in enumerate(FrameReader(str(src), d["source"]["filter"], d["width"], d["height"])):
    if f > last:
        break
    if f not in want:
        continue
    if full:
        cb = d.get("content_bottom") or d["height"]
        top = d["height"] - cb
        t = cv2.resize(fr[int(top):int(cb), :], (TW, TH), interpolation=cv2.INTER_AREA)
        for i, c in want[f]:
            u = t.copy()
            col = (0, 0, 255) if f == frames[i] else (255, 255, 255)
            cv2.putText(u, str(f), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(u, str(f), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            rows[i][c] = u
        continue
    b = sm[frames[want[f][0][0]]] if np.isfinite(sm[frames[want[f][0][0]], 0]) else sm[f]
    x1, y1, x2, y2 = b
    h = y2 - y1
    a0, a1 = y1 + band[0] * h, y1 + band[1] * h
    hw = (a1 - a0) * TW / TH / 2
    xc = (x1 + x2) / 2
    M = cv2.getAffineTransform(np.float32([[xc - hw, a0], [xc + hw, a0], [xc - hw, a1]]), np.float32([[0, 0], [TW, 0], [0, TH]]))
    t = cv2.warpAffine(fr, M, (TW, TH), flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
    for i, c in want[f]:
        u = t.copy()
        col = (0, 0, 255) if f == frames[i] else (255, 255, 255)
        cv2.putText(u, str(f), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(u, str(f), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        rows[i][c] = u
for k in range(0, len(frames), 10):
    img = np.concatenate([np.concatenate([np.pad(t, ((1, 1), (1, 1), (0, 0)), constant_values=255) for t in r], axis=1)
                          for r in rows[k:k + 10]], axis=0)
    hdr = np.full((22, img.shape[1], 3), 255, np.uint8)
    cv2.putText(hdr, f"{name}  rows {k}-{min(len(frames), k + 10) - 1}: " + " ".join(map(str, frames[k:k + 10])), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(ROOT / "out" / "moves" / f"{name}.{tag}_{k // 10}.png"), np.concatenate([hdr, img]))
print(name, len(frames), "rows")
