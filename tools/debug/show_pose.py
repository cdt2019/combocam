"""把缓存的关键点画到指定帧上，核对关键点质量。

  python tools/debug/show_pose.py out/pexels_38236117 24,39,139,358,408 out/pose_check.png [--crop 边长]
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (15, 20), (15, 22), (15, 24), (16, 21), (16, 23), (16, 25), (0, 17), (17, 18), (18, 19)]
LEFT, RIGHT = (255, 160, 0), (0, 120, 255)


def main() -> None:
    pre, frames, out = Path(sys.argv[1]), [int(v) for v in sys.argv[2].split(",")], sys.argv[3]
    crop = int(sys.argv[sys.argv.index("--crop") + 1]) if "--crop" in sys.argv else 0
    dbg = json.loads(pre.with_name(pre.name + ".debug.json").read_text(encoding="utf-8"))
    tr = json.loads(pre.with_name(pre.name + ".track.json").read_text(encoding="utf-8"))
    W, H = tr["width"], tr["height"]
    sel = "+".join("eq(n\\,%d)" % f for f in frames)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", tr["source"]["path"], "-filter_complex", tr["source"]["filter"] + ",select=" + sel,
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], capture_output=True).stdout
    tiles = []
    for k, f in enumerate(frames):
        im = np.frombuffer(raw, np.uint8)[k * W * H * 3:(k + 1) * W * H * 3].reshape(H, W, 3).copy()
        kp = dbg["kpts"][f]
        j = dbg["sel"][f]
        if j is not None:
            c = dbg["cands"][f][j]
            cv2.circle(im, (int(c[0]), int(c[1])), int(c[2]), (0, 255, 255), 3)
        cx, cy = W // 2, H // 2
        if kp:
            k3 = np.array(kp, np.float32).reshape(-1, 3)
            for a, b in EDGES:
                if k3[a, 2] > 30 and k3[b, 2] > 30:
                    col = LEFT if a % 2 == 1 and a < 17 or a in (20, 22, 24) else RIGHT
                    cv2.line(im, tuple(map(int, k3[a, :2])), tuple(map(int, k3[b, :2])), col, 3)
            for i, (x, y, s) in enumerate(k3):
                if s > 30:
                    cv2.circle(im, (int(x), int(y)), 6, (0, 255, 0) if i >= 20 else (255, 255, 255), -1)
            ok = k3[:, 2] > 30
            if ok.any():
                cx, cy = int(np.median(k3[ok, 0])), int(np.median(k3[ok, 1]))
        if crop:
            if j is not None:
                cx, cy = int(dbg["cands"][f][j][0]), int(dbg["cands"][f][j][1])
            x0, y0 = min(max(cx - crop // 2, 0), W - crop), min(max(cy - crop // 2, 0), H - crop)
            im = im[y0:y0 + crop, x0:x0 + crop]
        t = cv2.resize(im, (300, int(300 * im.shape[0] / im.shape[1])))
        cv2.putText(t, str(f), (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(t)
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tiles]
    cv2.imwrite(out, np.hstack(tiles))


if __name__ == "__main__":
    main()
