"""前后对比图：每段素材取"刚触球"和"球在空中"两个时刻，左改前、右改后。

  python tools/debug/compare_sheet.py out/before out out/compare.png pexels_38236117 pexels_36637447 ...
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def grab(path: Path, frame: int, w: int = 300) -> np.ndarray:
    h = int(w * 1920 / 1080)
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame}),scale={w}:{h}",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(h, w, 3).copy()


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    before, after, dst, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    rows = []
    for name in names:
        tr = json.loads((ROOT / "out" / f"{name}.track.json").read_text(encoding="utf-8"))
        tc = tr["touches"]
        fps = tr["fps"]
        a, b = (4, 6) if len(tc) >= 8 else (1, 2)
        picks = [int(round((tc[a]["t"] + 0.1) * fps)), int(round((tc[b]["t"] + 0.25) * fps))]
        tiles = []
        for f in picks:
            tiles.append(label(grab(before / f"{name}.neon.mp4", f), f"before  f{f}"))
            tiles.append(label(grab(after / f"{name}.neon.mp4", f), f"after   f{f}"))
            tiles.append(np.full((tiles[-1].shape[0], 12, 3), 255, np.uint8))
        rows.append(np.hstack(tiles[:-1]))
        rows.append(np.full((12, rows[-1].shape[1], 3), 255, np.uint8))
    cv2.imwrite(dst, np.vstack(rows[:-1]))
    print(dst)


if __name__ == "__main__":
    main()
