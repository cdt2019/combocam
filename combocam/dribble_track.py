"""篮球运球：视频 → 球的轨迹和关键点（颠球的分析管线）→ 运球计数（combocam/dribble.py）→ 渲染用的轨迹 JSON。

轨迹 JSON 和颠球的同一个格式（render/fx.js 原样画：球的拖尾、触地的冲击波、计数、节奏、球的高度曲线、片尾），
sport = "dribble"，labels 把画面上的文字换成运球的：
  touches  每次触地 {t, x, y（球心）, n（这组第几下）, combo, move（same 单手 / cross 交叉 / legs 胯下 / behind 背后 / null 分不出）, hand_from, hand_to}
  combos   每组连续运球 {from, to, count}
  apex     每两次触地之间球到的最高点 {t, x, y, height_m（离地，按篮球半径 0.12 m 换算）, touch, combo}
  ball     球的轨迹 [帧, x, y, r]（选中的原始检测，短漏检线性补上；不用颠球的抛物线平滑）
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .dribble import ball_series, count_dribbles

BALL_R_M = 0.12  # 篮球半径（7 号球直径约 24 cm）
MIN_DRIBBLES = 3
LABELS = {
    "count": "连续运球", "tag": "{n} · 运球", "chart": "球的高度", "interval": "运球间隔", "best": "最高弹起", "unit": "下",
    "banner": "{n} 下！", "drop": "断了 · 重新计数", "hand": "断了 · 重新计数",
    "endTitle": "连续运球", "endTitleMulti": "最长连续运球", "endTotal": "共 {g} 组 · 总运球 {n} 下",
    "endBars": "每一下的弹起高度", "endBarsMulti": "最长那组每一下的弹起高度", "marks": "{n} 下", "askPre": "你能连续运球 ", "askPost": " 下吗？",
    # 花式动作（combocam/dribble.classify_moves）：球旁标签写"N · 交叉"、触地点弹"交叉！"、片尾写各动作次数；单手运球不特别标
    "moves": {"cross": "交叉", "legs": "胯下", "behind": "背后"}, "tagMove": "{n} · {m}", "movePop": "{m}！",
}


def _fill_short(v: np.ndarray, max_gap: int) -> np.ndarray:
    v = v.copy()
    good = np.flatnonzero(np.isfinite(v))
    for a, b in zip(good, good[1:]):
        if 1 < b - a <= max_gap + 1:
            v[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [v[a], v[b]])
    return v


def build_track(d: dict, src: str | None = None, params: dict | None = None) -> dict:
    """d：tools/eval/dribble_extract.py 的缓存（或 analyze_dribble 里同样的东西）。"""
    fps, n = d["fps"], d["frames"]
    x, y, r, r_ref = ball_series(d["cands"], fps, d.get("persons"))
    res = count_dribbles(x, y, r, r_ref, fps, d.get("persons"), d.get("kpts"), params, d.get("content_bottom"), d.get("width"))
    xf, yf, rf = (_fill_short(v, 3) for v in (x, y, r))
    ball = [[f, round(float(xf[f]), 1), round(float(yf[f]), 1), round(float(rf[f]), 1)] for f in range(n) if np.isfinite(yf[f])]
    px_per_m = r_ref / BALL_R_M
    touches, apex, combos = [], [], []
    for g_i, g in enumerate(res["groups"]):
        combos.append({"from": g["from"], "to": g["to"], "count": g["count"]})
    dr = res["dribbles"]
    for k, h in enumerate(dr):
        f = h["frame"]
        rc = float(rf[f]) if np.isfinite(rf[f]) else r_ref
        touches.append({"t": h["t"], "x": h["x"], "y": round(h["y"] - rc, 1), "part": "floor", "n": h["n"], "combo": h["group"],
                        "move": h.get("move"), "hand_from": h.get("hand_from"), "hand_to": h.get("hand_to")})
        # 这一下弹起的最高点：到下一次触地（同一组）之间，没有下一次就看 0.6 秒
        nxt = dr[k + 1]["frame"] if k + 1 < len(dr) and dr[k + 1]["group"] == h["group"] else min(n - 1, f + int(0.6 * fps))
        seg = yf[f + 1:nxt]
        if np.isfinite(seg).any():
            j = f + 1 + int(np.nanargmin(seg))
            apex.append({"t": round(j / fps, 3), "x": round(float(xf[j]), 1), "y": round(float(yf[j]), 1),
                         "height_m": round(max(0.0, (h["y"] - (yf[j] + rf[j])) / px_per_m), 2), "touch": h["n"], "combo": h["group"]})
    total = res["total"]
    status = {"ok": True, "code": "ok", "title": "", "message": ""} if total >= MIN_DRIBBLES else {
        "ok": False, "code": "no_dribble", "title": "没数到运球",
        "message": "请把手机固定放好，拍到全身和脚下的地面，原地连续运球 5 下以上再试一次。"}
    return {
        "sport": "dribble", "version": 1, "labels": LABELS,
        "fps": fps, "width": d["width"], "height": d["height"], "frames": n, "duration": round(n / fps, 3),
        "ball": ball, "touches": touches, "apex": apex, "combos": combos, "bounces": [], "highlight": None,
        "calib": {"ball_radius_px": round(r_ref, 1)},
        "scene": d.get("scene"),
        "source": {**(d.get("source") or {}), **({"path": str(Path(src).resolve())} if src else {})},
        "summary": {"total": total, "best": res["best"], "groups": len(res["groups"]), "period_s": res["period_s"],
                    "rate": round(60 / res["period_s"], 1) if res["period_s"] else None, "ball_frames": res["ball_frames"],
                    "inferred": sum(1 for h in dr if h["inferred"]), "view": res.get("view"), "moves": res.get("moves")},
        "dribbles": dr,
        "status": status, "warnings": [],
    }


def analyze_dribble(src: str, model: str, pose_model: str, device: str = "auto", fps: float = 30.0,
                    frame_cache: dict | None = None) -> tuple[dict, dict]:
    """返回 (轨迹 JSON, 缓存)。frame_cache 同 analyze.analyze。"""
    from .analyze import analyze
    from .video import content_bottom

    t0 = time.perf_counter()
    dbg: dict = {}
    tr = analyze(src, model=model, device=device, fps=fps, pose_model=pose_model, debug=dbg, frame_cache=frame_cache, content_only=True)
    d = {k: tr[k] for k in ("fps", "width", "height", "frames", "ball", "scene", "source", "timing")}
    d.update({"cands": dbg.get("cands"), "persons": dbg.get("persons"), "kpts": dbg.get("kpts"),
              "content_bottom": content_bottom(tr["source"]["probe"]["width"], tr["source"]["probe"]["height"], tr["width"], tr["height"])})
    out = build_track(d, src)
    out["timing"] = {**tr["timing"], "total_s": round(time.perf_counter() - t0, 2)}
    out["meta"] = tr["meta"]
    return out, d
