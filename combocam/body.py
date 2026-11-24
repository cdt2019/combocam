"""用人体关键点判断触点：地面在哪（支撑脚的脚底）、球碰的是哪个部位、是不是手/前臂。

关键点按帧存成扁平整数列表 [x, y, 置信度×100, ...]（26 个 Halpe 点），见 analyze.py。
"""

from __future__ import annotations

import numpy as np

from .pose import (HEAD, L_ANKLE, L_BIG_TOE, L_ELBOW, L_HEEL, L_KNEE, L_SHOULDER, L_SMALL_TOE, L_WRIST, NECK,
                   NOSE, R_ANKLE, R_BIG_TOE, R_ELBOW, R_HEEL, R_KNEE, R_SHOULDER, R_SMALL_TOE, R_WRIST)

CONF = 40  # 置信度 ×100 的阈值
SOLE = [L_BIG_TOE, R_BIG_TOE, L_SMALL_TOE, R_SMALL_TOE, L_HEEL, R_HEEL]
GROUPS = {
    "foot": [L_ANKLE, R_ANKLE, L_BIG_TOE, R_BIG_TOE, L_SMALL_TOE, R_SMALL_TOE, L_HEEL, R_HEEL],
    "thigh": [L_KNEE, R_KNEE],
    "head": [NOSE, 1, 2, 3, 4, HEAD],
    "chest": [L_SHOULDER, R_SHOULDER, NECK],
}
FOREARMS = [(L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)]
UPPER = [NOSE, 1, 2, 3, 4, L_SHOULDER, R_SHOULDER, HEAD, NECK]  # 头、颈、肩
TORSO = [L_SHOULDER, R_SHOULDER, NECK]  # 身体参照点：球相对它不动 = 被身体托着走


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-6), 0, 1))
    return float(np.hypot(*(p - (a + t * ab))))


class Body:
    def __init__(self, kpts: list):
        self.n = len(kpts)
        self.k = [np.asarray(k, np.float32).reshape(-1, 3) if k else None for k in kpts]
        sole = np.full(self.n, np.nan)
        for f, k in enumerate(self.k):
            if k is not None:
                ok = [i for i in SOLE if k[i, 2] >= CONF]
                if ok:
                    sole[f] = k[ok, 1].max()  # 最低的脚底点 ≈ 支撑脚踩地的位置
        self.sole = sole
        self.sole_all = float(np.nanmedian(sole)) if np.isfinite(sole).sum() >= 5 else None

    def ground(self, f: int) -> float | None:
        """f 帧附近的地面线：支撑脚脚底的中位数。先看前后 8 帧，没有就放宽到 45 帧，再没有用全片中位数。"""
        for w in (8, 45):
            win = self.sole[max(0, f - w):f + w + 1]
            if np.isfinite(win).sum() >= 3:
                return float(np.nanmedian(win))
        return self.sole_all

    def near(self, f: int, x: float, y: float, r: float, span: int = 2) -> dict[str, float]:
        """触点前后 span 帧内，球心到各部位（关键点）的最近距离，单位球半径。forearm = 到前臂线段的距离。"""
        best = {g: 99.0 for g in (*GROUPS, "forearm")}
        p = np.array([x, y], np.float32)
        for ff in range(max(0, f - span), min(self.n, f + span + 1)):
            k = self.k[ff]
            if k is None:
                continue
            for g, idx in GROUPS.items():
                ok = [i for i in idx if k[i, 2] >= CONF]
                if ok:
                    best[g] = min(best[g], float(np.min(np.hypot(k[ok, 0] - x, k[ok, 1] - y))) / r)
            for a, b in FOREARMS:
                if k[a, 2] >= CONF and k[b, 2] >= CONF:
                    best["forearm"] = min(best["forearm"], _seg_dist(p, k[a, :2], k[b, :2]) / r)
        return best

    def carried(self, ball: dict, f: int, r: float, win: int) -> tuple[float | None, float | None, float | None]:
        """触点之后 win 帧里，球是不是被身体托着走（头上平衡、颈后停球、在手臂上滚）。返回三个量（单位球半径）：
        move  球相对颈、两肩中点的最大位移：托着走时 < 0.75，颠起来的球 ≥ 0.9（多数 ≥ 1.7）；
        stay  球心离头/颈/肩关键点不到 1.6 的帧占比；
        sep   球离头/颈/肩关键点的最远距离。"""
        rel, near_n, tot, sep = [], 0, 0, 0.0
        for ff in range(f, min(self.n, f + win + 1)):
            b, k = ball.get(ff), self.k[ff]
            if b is None or k is None:
                continue
            ok = [i for i in TORSO if k[i, 2] >= CONF]
            if ok:
                rel.append((b[0] - k[ok, 0].mean(), b[1] - k[ok, 1].mean()))
            up = [i for i in UPPER if k[i, 2] >= CONF]
            if up and ff > f:
                d = float(np.min(np.hypot(k[up, 0] - b[0], k[up, 1] - b[1]))) / r
                tot += 1
                near_n += d < 1.6
                sep = max(sep, d)
        move = None
        if len(rel) >= 3:
            a = np.asarray(rel, np.float32)
            move = float(np.max(np.hypot(*(a - a[0]).T))) / r
        return move, (near_n / tot if tot else None), (sep if tot else None)
