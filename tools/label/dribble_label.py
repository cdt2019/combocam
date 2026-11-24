"""运球标注辅助：逐帧小图，和球的检测无关，用来人工读"球碰地"的帧号。

  python tools/label/dribble_label.py 名 [--dir out/dribble] [--from 帧] [--to 帧] [--step 1] [--cols 20] [--tw 100] [--band 0.35,1.12] [--wide 1]
                                     [--chunk 帧数] [--out 目录] [--follow]
                                     → <out，默认 dir>/<名>.strip_<起>.png；--chunk 时每这么多帧一张（只解码一遍）

裁剪框按整段的人框定（竖向取人框高度的 --band 比例，默认 0.35–1.12 即下半身到脚下；横向左右各扩 --wide 个框宽），
整段固定不跟着人动；没有人框就用画布下半部分。远景、球小时用 --band 0.6,1.1 只看膝盖到地面那一条，小图又宽又矮，球大一些。
--follow：镜头在动、人走近走远时，每帧按这一帧的人框（前后 ±5 帧中位数）裁，宽高比固定，红线是这一帧的人框底边。
每张小图左上角印帧号，底下一条红线是人框底边（脚）的中位数。
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
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    name = a[0]
    d_dir = Path(opt("--dir", str(ROOT / "out" / "dribble")))
    f0, f1 = int(opt("--from", "0")), int(opt("--to", "100000"))
    step, ncol, tw = int(opt("--step", "1")), int(opt("--cols", "20")), int(opt("--tw", "100"))
    d = json.loads((d_dir / f"{name}.ball.json").read_text(encoding="utf-8"))
    W, H = d["width"], d["height"]
    boxes = np.array([b for b in d["persons"] if b], np.float64)
    b0, b1 = (float(v) for v in opt("--band", "0.35,1.12").split(","))
    wide = float(opt("--wide", "1"))
    chunk = int(opt("--chunk", "0"))
    o_dir = Path(opt("--out", str(d_dir)))
    o_dir.mkdir(parents=True, exist_ok=True)
    if len(boxes):
        x1, y1, x2, y2 = np.median(boxes, axis=0)
        w = x2 - x1
        cx0, cx1 = max(0, int(x1 - wide * w)), min(W, int(x2 + wide * w))
        cy0, cy1 = max(0, int(y1 + b0 * (y2 - y1))), min(H, int(y1 + b1 * (y2 - y1)))
        foot = y2
    else:
        cx0, cx1, cy0, cy1, foot = 0, W, H // 2, H, H * 0.9
    th = max(8, int(round(tw * (cy1 - cy0) / max(1, cx1 - cx0))))
    gy = int(round((foot - cy0) / (cy1 - cy0) * th))
    src = next(p / f"{name}.mp4" for p in (ROOT / "samples" / "dribble", ROOT / "samples" / "dribble2", ROOT / "samples" / "dribble3") if (p / f"{name}.mp4").exists())
    follow = "--follow" in a
    if follow:
        n = d["frames"]
        bx = np.array([b if b else [np.nan] * 4 for b in d["persons"]], np.float64).reshape(n, 4)
        sm = np.array([np.nanmedian(bx[max(0, f - 5):f + 6], axis=0) if np.isfinite(bx[max(0, f - 5):f + 6, 0]).any()
                       else [np.nan] * 4 for f in range(n)])
        aspect = (cx1 - cx0) / max(1, cy1 - cy0)

    def crop(f: int, frame: np.ndarray) -> tuple[np.ndarray, int]:
        if not follow or not np.isfinite(sm[f, 0]):
            return frame[cy0:cy1, cx0:cx1], gy
        x1, y1, x2, y2 = sm[f]
        h = y2 - y1
        a0, a1 = y1 + b0 * h, y1 + b1 * h
        hw = (a1 - a0) * aspect / 2
        xc = (x1 + x2) / 2
        src = np.array([[xc - hw, a0], [xc + hw, a0], [xc - hw, a1]], np.float32)
        dst = np.array([[0, 0], [tw, 0], [0, th]], np.float32)
        t = cv2.warpAffine(frame, cv2.getAffineTransform(src, dst), (tw, th), flags=cv2.INTER_AREA, borderValue=(255, 255, 255))
        return t, int(round((y2 - a0) / (a1 - a0) * th))

    tiles: list = []
    start = f0

    def flush() -> None:
        if not tiles:
            return
        rows = []
        for k in range(0, len(tiles), ncol):
            row = tiles[k:k + ncol]
            row += [np.full_like(tiles[0], 255)] * (ncol - len(row))
            rows.append(np.concatenate([np.pad(t, ((1, 1), (1, 1), (0, 0)), constant_values=255) for t in row], axis=1))
        out = o_dir / f"{name}.strip_{start}.png"
        cv2.imwrite(str(out), np.concatenate(rows, axis=0))
        print(out, f"{len(tiles)} 张")
        tiles.clear()

    for f, frame in enumerate(FrameReader(str(src), d["source"]["filter"], W, H)):
        if f < f0 or (f - f0) % step:
            continue
        if f > f1:
            break
        if chunk and f - start >= chunk:
            flush()
            start = f
        t, ly = crop(f, frame)
        if t.shape[:2] != (th, tw):
            t = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.line(t, (0, min(th - 1, ly)), (tw - 1, min(th - 1, ly)), (0, 0, 255), 1)
        cv2.putText(t, str(f), (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, str(f), (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    flush()


if __name__ == "__main__":
    main()
