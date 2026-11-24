"""从每帧的候选框里挑出"被颠的那个球"的轨迹。

做法：把所有帧的候选框连成一张有向无环图，求得分最高的一条路径（离线，前后帧一起看）。
- 节点得分 = 2×置信度 − 0.25：高分检测是骨架，低分候选只有在能把前后高分点连起来（省掉跳帧罚分）时才会被选上；
  尺寸和球径偏差过大的扣分；静止不动的候选（场边另一个球、地上的球、白鞋）不给分。
- 边代价 = 速度平方项 + 跳帧罚分；超过物理速度上限、尺寸突变的边不连。
- 允许"长跳"（丢失超过 K 帧后重新捕获），代价固定。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Cand:
    x: float
    y: float
    r: float
    s: float  # 置信度
    src: int = 0  # 0=整帧检测，1=局部放大复检


def ref_radius(cands: list[list[Cand]], default: float = 20.0) -> float:
    good = [c.r for fr in cands for c in fr if c.s >= 0.3]
    if len(good) >= 5:
        return float(np.median(good))
    allr = [c.r for fr in cands for c in fr]
    return float(np.median(allr)) if allr else default


def mark_static(cands: list[list[Cand]], fps: float, r_ref: float, win_s: float = 1.0, ratio: float = 0.6) -> list[np.ndarray]:
    """候选在前后 win_s 秒内大多数采样帧里都有一个几乎同位置的候选 → 静止物体。"""
    n = len(cands)
    pos = [np.array([[c.x, c.y] for c in fr], dtype=np.float64).reshape(-1, 2) for fr in cands]
    win = max(2, int(win_s * fps))
    step = max(1, int(fps / 10))
    tol = 0.35 * r_ref
    out = []
    for f in range(n):
        if len(pos[f]) == 0:
            out.append(np.zeros(0, bool))
            continue
        hits = np.zeros(len(pos[f]))
        total = 0
        for g in range(max(0, f - win), min(n, f + win + 1), step):
            if g == f:
                continue
            total += 1
            if len(pos[g]) == 0:
                continue
            d = np.linalg.norm(pos[f][:, None, :] - pos[g][None, :, :], axis=2).min(axis=1)
            hits += d < tol
        out.append(hits >= ratio * max(total, 1) if total >= 4 else np.zeros(len(pos[f]), bool))
    return out


def select_track(
    cands: list[list[Cand]],
    fps: float,
    r_ref: float,
    static: list[np.ndarray] | None = None,
    max_gap_s: float = 0.4,
    vmax_r_per_s: float = 90.0,
    w_score: float = 2.0,
    bias: float = -0.25,
    beta: float = 0.4,
    gamma: float = 0.25,
    long_base: float = 2.0,
    long_per_frame: float = 0.2,
    size_pen: float = 0.6,
) -> list[int | None]:
    """返回每帧选中的候选下标（None = 这一帧没有球）。

    vmax_r_per_s：球速上限，单位"球半径/秒"。5 号球半径 0.11 m，90 r/s ≈ 10 m/s，颠球够用。
    size_pen：半径和 r_ref 差 1.6 倍以上的候选扣的分（篮球运球时球离镜头远近变化大，大小已经按人框筛过，传 0）。
    """
    n = len(cands)
    K = max(3, int(round(max_gap_s * fps)))
    vmax = vmax_r_per_s * r_ref / fps  # 像素/帧
    pos = [np.array([[c.x, c.y] for c in fr], dtype=np.float64).reshape(-1, 2) for fr in cands]
    rad = [np.array([c.r for c in fr], dtype=np.float64) for fr in cands]
    rew = []
    for f, fr in enumerate(cands):
        r = np.array([w_score * c.s + bias for c in fr], dtype=np.float64)
        if len(r):
            size_dev = np.abs(np.log(np.array([c.r for c in fr]) / r_ref))
            r[size_dev > np.log(1.6)] -= size_pen
            if static is not None:
                r[static[f]] = -1.0
        rew.append(r)

    best: list[np.ndarray] = [np.zeros(0)] * n
    back: list[list] = [[] for _ in range(n)]
    # 长跳代价 = long_base + long_per_frame × 跳过帧数，这样连得上的低分候选优先于直接跳过。
    # jump_val[f] = max_{f'≤f} (best[f'] + long_per_frame·f')，用前缀最大值 O(1) 取到。
    jump_val = np.full(n, -np.inf)
    jump_arg: list[tuple[int, int] | None] = [None] * n
    end_val = np.full(n, -np.inf)
    end_arg: list[tuple[int, int] | None] = [None] * n
    for g in range(n):
        m = len(pos[g])
        if m == 0:
            best[g] = np.zeros(0)
        else:
            bg = rew[g].copy()  # 从这里开始一条新路径
            bk: list[tuple[int, int] | None] = [None] * m
            for gap in range(1, K + 1):
                f = g - gap
                if f < 0:
                    break
                if len(pos[f]) == 0:
                    continue
                d = np.linalg.norm(pos[f][:, None, :] - pos[g][None, :, :], axis=2) / gap
                size_ok = np.abs(np.log(rad[f][:, None] / rad[g][None, :])) < np.log(2.0)
                cost = beta * (d / vmax) ** 2 + gamma * (gap - 1)
                val = best[f][:, None] - cost
                val[(d > vmax) | ~size_ok] = -np.inf
                i = val.argmax(axis=0)
                v = val[i, np.arange(m)] + rew[g]
                upd = v > bg
                for j in np.nonzero(upd)[0]:
                    bg[j] = v[j]
                    bk[j] = (f, int(i[j]))
            fl = g - K - 1
            if fl >= 0 and jump_arg[fl] is not None:
                v = jump_val[fl] - long_per_frame * g - long_base + rew[g]
                for j in np.nonzero(v > bg)[0]:
                    bg[j] = v[j]
                    bk[j] = jump_arg[fl]
            best[g], back[g] = bg, bk
        jv, ja = (jump_val[g - 1], jump_arg[g - 1]) if g else (-np.inf, None)
        ev, ea = (end_val[g - 1], end_arg[g - 1]) if g else (-np.inf, None)
        if len(best[g]):
            j = int(best[g].argmax())
            if best[g][j] + long_per_frame * g > jv:
                jv, ja = float(best[g][j] + long_per_frame * g), (g, j)
            if best[g][j] > ev:
                ev, ea = float(best[g][j]), (g, j)
        jump_val[g], jump_arg[g], end_val[g], end_arg[g] = jv, ja, ev, ea

    sel: list[int | None] = [None] * n
    node = end_arg[n - 1] if n else None
    if node is None or end_val[n - 1] <= 0:
        return sel
    while node is not None:
        f, j = node
        sel[f] = j
        node = back[f][j]
    return sel
