"""把分析结果画成图，人工核对用。

  python tools/debug/inspect_track.py out/<名>            # 读 <名>.debug.json 和 <名>.track.json
输出：
  <名>.plot.png   纵轴=球心 y（向下为正），横轴=帧；灰点=候选（越深分越高），蓝点=选中，绿线=输出轨迹，
                  红竖线=触球（标序号），橙竖线=落地反弹
  <名>.sheet_<k>.png  按帧号排的缩略图（每 --step 帧一张），叠加输出轨迹点和触球标记，用来人工数触球
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def recompute(dbg: dict, tr: dict) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from combocam.analyze import postprocess
    from combocam.track import Cand, mark_static, ref_radius, select_track

    cands = [[Cand(*c) for c in fr] for fr in dbg["cands"]]
    persons = [tuple(p) if p else None for p in dbg["persons"]]
    r_ref = ref_radius(cands)
    static = mark_static(cands, tr["fps"], r_ref)
    sel = select_track(cands, tr["fps"], r_ref, static)
    _, _, ev = postprocess(cands, persons, tr["fps"], tr["height"], sel)
    dbg["sel"], dbg["static"] = sel, [s.tolist() for s in static]
    for k in ("ball", "touches", "apex", "combos", "highlight", "bounces", "calib", "warnings"):
        tr[k] = getattr(ev, k)
    tr["summary"].update({
        "touches": len(ev.touches),
        "best_combo": max((c["count"] for c in ev.combos), default=0),
        "best_height_m": max((x["height_m"] for x in ev.apex), default=None),
        "track_frames": sum(1 for s in sel if s is not None),
    })


def plot(dbg: dict, tr: dict, path: Path) -> None:
    n = len(dbg["cands"])
    H, W, pad = 700, max(1200, n * 3), 40
    img = np.full((H, W + 2 * pad, 3), 255, np.uint8)
    sy = (H - 2 * pad) / tr["height"]
    sx = W / max(1, n)
    X = lambda f: int(pad + f * sx)  # noqa: E731
    Y = lambda y: int(pad + y * sy)  # noqa: E731
    for k in range(0, tr["height"] + 1, 240):
        cv2.line(img, (pad, Y(k)), (pad + W, Y(k)), (235, 235, 235), 1)
        cv2.putText(img, str(k), (2, Y(k) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    for f in range(0, n, int(tr["fps"])):
        cv2.line(img, (X(f), pad), (X(f), H - pad), (235, 235, 235), 1)
        cv2.putText(img, str(f), (X(f) - 8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    if tr["calib"].get("ground_y"):
        cv2.line(img, (pad, Y(tr["calib"]["ground_y"])), (pad + W, Y(tr["calib"]["ground_y"])), (90, 160, 90), 1)
    for b in tr["bounces"]:
        f = b["t"] * tr["fps"]
        cv2.line(img, (X(f), pad), (X(f), H - pad), (0, 150, 255), 1)
    for t in tr["touches"]:
        f = t["t"] * tr["fps"]
        cv2.line(img, (X(f), pad), (X(f), H - pad), (60, 60, 230), 1)
        cv2.putText(img, f"{t['n']}{t['part'][0]}", (X(f) + 2, pad - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 230), 1)
    for f, fr in enumerate(dbg["cands"]):
        for j, c in enumerate(fr):
            g = int(220 - 200 * c[3])
            col = (255, 120, 0) if dbg["sel"][f] == j else (g, g, g)
            if dbg["static"][f] and j < len(dbg["static"][f]) and dbg["static"][f][j]:
                col = (200, 0, 200)
            cv2.circle(img, (X(f), Y(c[1])), 3 if dbg["sel"][f] == j else 2, col, -1)
    pts = [(X(b[0]), Y(b[2])) for b in tr["ball"]]
    for a, b in zip(pts, pts[1:]):
        if b[0] - a[0] <= 2 * sx + 1:
            cv2.line(img, a, b, (0, 170, 0), 1)
    cv2.imwrite(str(path), img)


def sheets(tr: dict, path_prefix: Path, step: int, cols: int = 10, rows: int = 4, w: int = 180, crop=None, frame_range=None) -> list[Path]:
    src, vf = tr["source"]["path"], tr["source"]["filter"]
    x0, y0, x1, y1 = crop or (0, 0, tr["width"], tr["height"])
    h = int(w * (y1 - y0) / (x1 - x0))
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", src, "-filter_complex", f"{vf},crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={w}:{h}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        capture_output=True, check=True,
    )
    frames = np.frombuffer(proc.stdout, np.uint8).reshape(-1, h, w, 3)
    ball = {b[0]: b for b in tr["ball"]}
    touch_f = {int(round(t["t"] * tr["fps"])): t for t in tr["touches"]}
    bounce_f = {int(round(b["t"] * tr["fps"])) for b in tr["bounces"]}
    s = w / (x1 - x0)
    tiles = []
    lo, hi = frame_range or (0, len(frames))
    for f in range(lo, min(hi, len(frames)), step):
        im = frames[f].copy()
        if f in ball:
            cv2.circle(im, (int((ball[f][1] - x0) * s), int((ball[f][2] - y0) * s)), max(3, int(ball[f][3] * s)), (0, 255, 0), 1)
        label = str(f)
        near_t = [t for tf, t in touch_f.items() if abs(tf - f) < step / 2 + 0.01]
        if near_t:
            label += f" T{near_t[0]['n']}"
            cv2.rectangle(im, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
        if any(abs(bf - f) < step / 2 + 0.01 for bf in bounce_f):
            label += " B"
            cv2.rectangle(im, (0, 0), (w - 1, h - 1), (0, 140, 255), 3)
        cv2.rectangle(im, (0, 0), (w, 18), (0, 0, 0), -1)
        cv2.putText(im, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(im)
    out = []
    per = cols * rows
    for k in range(0, len(tiles), per):
        chunk = tiles[k:k + per]
        chunk += [np.zeros_like(tiles[0])] * (per - len(chunk))
        grid = np.vstack([np.hstack(chunk[r * cols:(r + 1) * cols]) for r in range(rows)])
        p = Path(f"{path_prefix}.sheet_{k // per}.png")
        cv2.imwrite(str(p), grid)
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="out/<名>")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--crop", default="", help="x0,y0,x1,y1（画布坐标）")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--range", default="", help="起止帧 a,b")
    ap.add_argument("--tile", type=int, default=180, help="每格宽度")
    ap.add_argument("--recompute", action="store_true", help="用缓存的候选框重跑轨迹和事件逻辑（改了 track/events 后用）")
    a = ap.parse_args()
    pre = Path(a.prefix)
    dbg = json.loads(Path(f"{pre}.debug.json").read_text(encoding="utf-8"))
    tr = json.loads(Path(f"{pre}.track.json").read_text(encoding="utf-8"))
    if a.recompute:
        recompute(dbg, tr)
        Path(f"{pre}.debug.json").write_text(json.dumps(dbg), encoding="utf-8")
        Path(f"{pre}.track.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
        print(f"touches={len(tr['touches'])} combos={[c['count'] for c in tr['combos']]} bounces={len(tr['bounces'])} "
              f"parts={[t['part'][0] for t in tr['touches']]} calib={tr['calib']} warnings={tr['warnings']}")
    plot(dbg, tr, Path(f"{pre}.plot.png"))
    print(f"{pre}.plot.png")
    if not a.no_sheets:
        crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
        rng = tuple(int(v) for v in a.range.split(",")) if a.range else None
        for p in sheets(tr, pre, a.step, crop=crop, cols=a.cols, rows=a.rows, w=a.tile, frame_range=rng):
            print(p)


if __name__ == "__main__":
    main()
