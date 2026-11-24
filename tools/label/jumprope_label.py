"""跳绳标注辅助：脚部逐帧小图，和关键点算法无关，用来人工数跳数。

  python tools/label/jumprope_label.py <名> [--from 帧] [--to 帧] [--cols 20] [--tw 108] [--step 1] [--dir 关键点缓存目录] [--body]

--body 裁全身（脚被挡住时看身体起落）；多人入镜的缓存按 combocam/people.py 选出的那个人裁。

每帧从同一个固定位置（人框底边、中心的中位数附近）裁出脚的区域，缩小后按帧号排成网格，左上角印帧号。
裁剪框不跟着人动，所以人离地时脚在小图里明显上移、和地面之间出现空隙。输出 out/jumprope/<名>.strip_<起>.png。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from combocam.jumprope_track import main_person  # noqa: E402
from combocam.video import FrameReader, canvas_filter, probe  # noqa: E402

OUT = ROOT / "out" / "jumprope"


def main() -> None:
    name = sys.argv[1]
    a = sys.argv
    f_from = int(a[a.index("--from") + 1]) if "--from" in a else 0
    f_to = int(a[a.index("--to") + 1]) if "--to" in a else 10**9
    ncol = int(a[a.index("--cols") + 1]) if "--cols" in a else 20
    out = Path(a[a.index("--dir") + 1]) if "--dir" in a else OUT
    cache = json.loads((out / f"{name}.pose.json").read_text(encoding="utf-8"))
    # 裁剪框跟着人走，但用 1 秒滑动中位数：手持镜头的慢慢晃动跟得上，跳起来的快速起伏（零点几秒一次）留在画面里
    from scipy.ndimage import median_filter

    boxes, _, _ = main_person(cache)
    idx = [i for i, b in enumerate(boxes) if b]
    arr = np.array([boxes[i] for i in idx])
    full = np.array([np.interp(range(len(boxes)), idx, arr[:, k]) for k in range(4)]).T
    win = int(cache["fps"]) | 1
    cx_s = median_filter((full[:, 0] + full[:, 2]) / 2, size=win, mode="nearest")
    bot_s = median_filter(full[:, 3], size=win, mode="nearest")
    tall = float(np.median(arr[:, 3] - arr[:, 1]))
    half_w, up, down = (0.3 * tall, 1.05 * tall, 0.06 * tall) if "--body" in a else (0.28 * tall, 0.22 * tall, 0.06 * tall)
    tw = int(a[a.index("--tw") + 1]) if "--tw" in a else 108
    step = int(a[a.index("--step") + 1]) if "--step" in a else 1  # 隔几帧取一帧（慢动作视频用）
    th = max(1, int(round(tw * (up + down) / (2 * half_w))))
    tiles = []
    # 第一批在 samples/jumprope/，第二批、第三批（验证集）在 samples/jumprope2/、jumprope3/
    src = next((d / f"{name}.mp4" for d in (ROOT / "samples" / "jumprope", ROOT / "samples" / "jumprope2", ROOT / "samples" / "jumprope3") if (d / f"{name}.mp4").exists()),
               ROOT / "samples" / "jumprope" / f"{name}.mp4")
    for f, frame in enumerate(FrameReader(str(src), canvas_filter(probe(str(src))), 1080, 1920)):
        if f < f_from:
            continue
        if f > f_to:
            break
        if (f - f_from) % step:
            continue
        cx, bottom = cx_s[min(f, len(cx_s) - 1)], bot_s[min(f, len(bot_s) - 1)]
        x0, y0 = int(round(cx - half_w)), int(round(bottom - up))
        crop = np.full((int(round(up + down)), int(round(2 * half_w)), 3), 128, np.uint8)  # 超出画面的部分填灰
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(1080, x0 + crop.shape[1]), min(1920, y0 + crop.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
        t = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        # 参考线：人框底边的 1 秒中位数（≈ 地面）
        gy = int(round(up / (up + down) * th))
        cv2.line(t, (0, gy), (tw - 1, gy), (0, 0, 255), 1)
        cv2.putText(t, str(f), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, str(f), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    rows = []
    for r in range(0, len(tiles), ncol):
        row = tiles[r:r + ncol]
        row += [np.full_like(tiles[0], 255)] * (ncol - len(row))
        rows.append(np.concatenate([np.pad(t, ((1, 1), (1, 1), (0, 0)), constant_values=255) for t in row], axis=1))
    img = np.concatenate(rows, axis=0)
    path = out / f"{name}.strip_{f_from}{'_body' if '--body' in a else ''}.png"
    cv2.imwrite(str(path), img)
    print(f"{path}：第 {f_from}–{f_from + len(tiles) - 1} 帧，{len(rows)} 行 × {ncol} 列；红线 = 人框底边中位数")


if __name__ == "__main__":
    main()
