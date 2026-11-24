"""放大看几帧（运球动作标注用）：在真实画面里按比例裁一块，拼成一张图。

  python tools/label/dribble_zoom.py 名 帧,帧,... x0,y0,x1,y1 输出.png [每行几张 8] [每张宽 300]
x0,y0,x1,y1 是占真实画面宽高的比例（横屏放进竖版时只算中间的真实画面）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from combocam.video import FrameReader  # noqa: E402


def main() -> None:
    a = sys.argv[1:]
    name, fs, box, out = a[0], [int(v) for v in a[1].split(",")], [float(v) for v in a[2].split(",")], a[3]
    per = int(a[4]) if len(a) > 4 else 8
    tw = int(a[5]) if len(a) > 5 else 300
    d = json.loads((ROOT / "out" / "dribble" / f"{name}.ball.json").read_text(encoding="utf-8"))
    cb = d.get("content_bottom") or d["height"]
    top = d["height"] - cb
    H = cb - top
    src = next(p / f"{name}.mp4" for p in (ROOT / "samples" / "dribble", ROOT / "samples" / "dribble2", ROOT / "samples" / "dribble3") if (p / f"{name}.mp4").exists())
    tiles = {}
    for f, fr in enumerate(FrameReader(str(src), d["source"]["filter"], d["width"], d["height"])):
        if f > max(fs):
            break
        if f in fs:
            c = fr[int(top + box[1] * H):int(top + box[3] * H), int(box[0] * d["width"]):int(box[2] * d["width"])]
            c = cv2.resize(c, (tw, int(tw * c.shape[0] / c.shape[1])))
            cv2.putText(c, str(f), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            tiles[f] = c
    rows = []
    for k in range(0, len(fs), per):
        row = [np.pad(tiles[f], ((1, 1), (1, 1), (0, 0)), constant_values=255) for f in fs[k:k + per]]
        if len(row) < per:
            row.append(np.full((row[0].shape[0], (tw + 2) * (per - len(row)), 3), 255, np.uint8))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(out, np.concatenate(rows))
    print(out)


if __name__ == "__main__":
    main()
