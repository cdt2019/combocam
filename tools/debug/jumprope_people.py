"""多人入镜调试图：逐帧所有人连成轨迹后，在均匀抽的几帧上画出每条轨迹的人框、编号和数到的跳数（选中的那条最粗）。

  python tools/debug/jumprope_people.py 名 [--dir out/jumprope_mp] [--n 8] [--h 480]   → <dir>/<名>.people.png

关键点缓存要有 frames_people（tools/eval/jumprope_extract.py 现在都存）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from combocam import people  # noqa: E402
from combocam.jumprope import count_jumps  # noqa: E402
from combocam.video import FrameReader  # noqa: E402

COLORS = [(60, 255, 60), (0, 200, 255), (255, 120, 0), (255, 0, 255), (0, 0, 255), (255, 255, 0), (128, 128, 255), (200, 200, 200)]


def main() -> None:
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    name = a[0]
    d_dir = Path(opt("--dir", str(ROOT / "out" / "jumprope_mp")))
    n_show, hh = int(opt("--n", "8")), int(opt("--h", "480"))
    d = json.loads((d_dir / f"{name}.pose.json").read_text(encoding="utf-8"))
    fp, fps, n = d["frames_people"], d["fps"], d["frames"]
    tracks = people.merge(fp, people.link(fp, fps))
    rows = []
    for t in tracks:
        _, kp = people.series(fp, t, n)
        rows.append((count_jumps(kp, fps, frame_h=d["height"]).get("total", 0), t))
    chosen, summary = people.select(fp, fps, d["height"])
    rank = sorted(range(len(rows)), key=lambda i: -len(rows[i][1]["frames"]))
    want = set(np.linspace(0, n - 1, n_show).astype(int).tolist())
    # 第一批在 samples/jumprope/，第二批、第三批（验证集）在 samples/jumprope2/、jumprope3/
    src = next((d / f"{name}.mp4" for d in (ROOT / "samples" / "jumprope", ROOT / "samples" / "jumprope2", ROOT / "samples" / "jumprope3") if (d / f"{name}.mp4").exists()),
               ROOT / "samples" / "jumprope" / f"{name}.mp4")
    tiles = []
    for f, frame in enumerate(FrameReader(str(src), d["filter"], d["width"], d["height"], yuv_canvas=True)):
        if f not in want:
            continue
        img = frame.copy()
        for k, i in enumerate(rank):
            jumps, t = rows[i]
            if f not in t["frames"]:
                continue
            x1, y1, x2, y2 = (int(v) for v in fp[f][t["frames"][f]]["box"])
            sel = t["frames"] == chosen["frames"]
            c = COLORS[k % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 14 if sel else 5)
            cv2.putText(img, f"#{k} {jumps}{' *' if sel else ''}", (x1 + 8, y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 2.4, c, 7)
        cv2.putText(img, str(f), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
        tiles.append(cv2.resize(img, (hh * d["width"] // d["height"], hh)))
    out = d_dir / f"{name}.people.png"
    cv2.imwrite(str(out), np.hstack(tiles))
    print(out, "；轨迹（按分数）：", [(r["jumps"], r["height"], r["from"], r["to"], r["score"]) for r in summary])


if __name__ == "__main__":
    main()
