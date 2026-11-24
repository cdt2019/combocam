"""从球的轨迹算出触球、最高点、连击、高光。

- 两次触球之间球只受重力：y(t) 是抛物线、x(t) 近似直线。逐段拟合后，
  相邻两段抛物线的交点就是触球时刻（亚帧精度），拟合曲线也用来补被身体挡住的帧。
- 触球 = 纵向运动由下落转为上升的拐点；球底贴近地面线的拐点是落地反弹，不计数。
- 高度用 5 号球直径 22 cm 做比例尺；同时用重力加速度反推一个比例尺做交叉校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .body import Body
from .track import Cand

BALL_DIAMETER_M = 0.22
G = 9.81


@dataclass
class Arc:
    t0: float
    t1: float
    ay: np.ndarray | None  # y = ay[0] t^2 + ay[1] t + ay[2]
    ax: np.ndarray | None  # x = ax[0] t + ax[1]
    n_obs: int

    def y(self, t):
        return np.polyval(self.ay, t)

    def x(self, t):
        return np.polyval(self.ax, t)


@dataclass
class Result:
    ball: list = field(default_factory=list)
    touches: list = field(default_factory=list)
    bounces: list = field(default_factory=list)
    apex: list = field(default_factory=list)
    combos: list = field(default_factory=list)
    highlight: dict | None = None
    calib: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _segments(frames: list[int], max_gap: int) -> list[list[int]]:
    segs: list[list[int]] = []
    for f in frames:
        if segs and f - segs[-1][-1] <= max_gap:
            segs[-1].append(f)
        else:
            segs.append([f])
    return segs


def _fit_arc(t: np.ndarray, x: np.ndarray, y: np.ndarray, r_ref: float) -> tuple[np.ndarray, np.ndarray] | None:
    """抛物线 + 直线拟合，剔除一次离群点后重拟。"""
    if len(t) < 4 or t[-1] - t[0] < 0.12:
        return None
    ay = np.polyfit(t, y, 2)
    res = np.abs(np.polyval(ay, t) - y)
    keep = res < max(0.6 * r_ref, 2.5 * np.median(res) + 1e-6)
    if keep.sum() >= 4 and not keep.all():
        t, x, y = t[keep], x[keep], y[keep]
        ay = np.polyfit(t, y, 2)
    ax = np.polyfit(t, x, 1)
    # 拟合质量门槛：残差大说明这一段不是单纯的抛体（球在脚上滚、被手接住、漏掉了一次触球），不用拟合结果
    ry = np.polyval(ay, t) - y
    rx = np.polyval(ax, t) - x
    if np.sqrt(np.mean(ry ** 2)) > 0.2 * r_ref or np.sqrt(np.mean(rx ** 2)) > 0.35 * r_ref:
        return None
    return ay, ax


def _intersect(a1: np.ndarray, a2: np.ndarray, t_guess: float, tol: float) -> float | None:
    """两段抛物线的交点。两条抛物线可能有两个交点：触球/落地是 V 形的最低点（y 最大），取 y 最大的那个。"""
    roots = np.roots(a1 - a2)
    roots = roots[np.isreal(roots)].real
    roots = roots[np.abs(roots - t_guess) <= tol]
    if len(roots) == 0:
        return None
    return float(roots[np.argmax(np.polyval(a1, roots))])


def _part(y: float, person: tuple | None) -> str:
    if person is None:
        return "unknown"
    x1, y1, x2, y2 = person
    rel = (y - y1) / max(1.0, y2 - y1)
    if rel < 0.2:
        return "head"
    if rel < 0.45:
        return "chest"
    if rel < 0.72:
        return "thigh"
    return "foot"


# 触点分类的阈值（单位：球半径），见 analyze_track 里"分类"一节
#   away       球离人框超过这么远：不是人碰的
#   hand_*     手接：球离前臂 < hand_fa，且脚、膝都在 hand_limb 之外
#   g_*        落地：球底低于地面线 g_gap，且（落下 < g_lift 或 弹起 < g_k × 落下）
#   low_*      地上玩球：触点前后 low_win 秒里，球底贴地（离地面线 < low_h）的时间占比 ≥ low_frac
#   carry_*    头上平衡、颈后停球：触点在头/胸附近（< carry_near），之后 carry_win 秒里球相对颈肩的位移 < carry_move；
#              或者球一直贴着头（贴着的帧占比 ≥ 0.9、最远 < 1）而且手扶着（离前臂 < carry_fa）
#   far_*      没碰到人：看得见脚，脚、大腿、头、胸离球都 ≥ far_part，而且是被动反弹（弹起 < far_k × 落下）——
#              球在镜头前落地、射门撞网、地上滚的球磕一下（2026-10-12 第 4 批留出集）
#   rest_*     从地上挑起：触点前球几乎没在落（落下 < rest_drop），球底离地面线 < rest_gap
#   alone      触点前后 alone 帧里都没有人：不是人碰的（人走出画面后球自己在弹；4 批留出集 407 次真触球没有一次是这样）
P = {"away": 18, "hand_fa": 1.5, "hand_limb": 2.5, "g_gap": -0.2, "g_k": 0.75, "g_lift": 1.0,
     "low_win": 0.5, "low_h": 0.7, "low_frac": 0.4,
     "carry_near": 2.0, "carry_win": 0.4, "carry_move": 0.75, "carry_fa": 2.0,
     "far_part": 2.5, "far_k": 0.75, "rest_drop": 0.5, "rest_gap": 1.0, "alone": 6}


def _carried(body: Body, ball: dict, f: int, r: float, fps: float, near: dict) -> bool:
    """头上平衡、颈后停球、在胸前托着：触点之后球跟着身体走，而不是飞出去（实测见 Body.carried）。"""
    move, stay, sep = body.carried(ball, f, r, max(3, int(round(P["carry_win"] * fps))))
    if move is not None and move < P["carry_move"]:
        return True
    return stay is not None and stay >= 0.9 and sep < 1.0 and near["forearm"] < P["carry_fa"]


def _low_fraction(body: Body, ball: dict, f: int, r: float, fps: float) -> float | None:
    """触点前后 low_win 秒内，球底离地面线不到 low_h 个球半径的帧占比。颠球时球只在触点那一瞬间靠近地面；
    在地上拨球、带球、小弹跳时球一直贴着地（实测：真触球 ≤ 0.39，地上玩球 0.4–1.0）。"""
    w = max(3, int(round(P["low_win"] * fps)))
    hs = []
    for ff in range(f - w, f + w + 1):
        b = ball.get(ff)
        g = body.ground(ff) if b is not None else None
        if g is not None:
            hs.append((g - (b[1] + r)) / r)
    return float(np.mean(np.array(hs) < P["low_h"])) if len(hs) >= w else None


def analyze_track(
    track: dict[int, Cand],
    fps: float,
    n_frames: int,
    persons: list[tuple | None],
    height: int,
    max_gap_s: float = 0.4,
    combo_break_s: float = 2.2,
    debug: dict | None = None,
    content_bottom: float | None = None,
    kpts: list | None = None,
) -> Result:
    """debug 不为空时，把每个候选触点的判断依据存到 debug["contacts"]（调参、排查用）。
    kpts：每帧主要人物的 Halpe26 关键点（扁平整数列表，见 body.py）；有它就用脚底定地面、用部位距离判断手接球。"""
    res = Result()
    if len(track) < 8:
        res.warnings.append("球的轨迹太短，没有识别到颠球")
        return res
    frames = sorted(track)
    K = max(3, int(round(max_gap_s * fps)))

    # ---- 比例尺：慢速（接近最高点）时的检测框最准 ----
    # 只用球在人附近（水平方向离人框不到一个框宽）的帧：球被踢远、滚到远处时在画面里小一半，混进来会把比例尺拉小，
    # 人附近的距离、高度全跟着算错（2026-10-12 加了放大复检后，远处滚走的球也跟得上了，38002813 就是这样）
    def near_person(f: int) -> bool:
        pb = persons[f] if f < len(persons) else None
        if pb is None:
            return True
        w = pb[2] - pb[0]
        return pb[0] - w <= track[f].x <= pb[2] + w
    near_frames = [f for f in frames if near_person(f)]
    base = near_frames if len(near_frames) >= max(8, 0.3 * len(frames)) else frames
    base_set = set(base)
    rs, speeds = [], []
    for a, b in zip(frames, frames[1:]):
        if b - a == 1 and a in base_set and b in base_set:
            ca, cb = track[a], track[b]
            speeds.append(np.hypot(cb.x - ca.x, cb.y - ca.y))
            rs.append((ca.r + cb.r) / 2)
    rs, speeds = np.array(rs), np.array(speeds)
    r_all = np.median([track[f].r for f in base])
    slow = speeds < 0.5 * r_all if len(speeds) else np.zeros(0, bool)
    r_ref = float(np.median(rs[slow])) if slow.sum() >= 5 else float(r_all)
    px_per_m = 2 * r_ref / BALL_DIAMETER_M

    # ---- 地面线：人框底边（脚）的中位数，忽略被画面底边截断的框 ----
    # 横屏素材放进竖版画布后，真实画面的下边缘在画布中间（content_bottom），不是画布底边
    bottom = content_bottom if content_bottom is not None else height
    feet = [p[3] for p in persons if p is not None and p[3] < bottom - 0.01 * height]
    ground_y = float(np.median(feet)) if len(feet) >= max(5, 0.2 * n_frames) else None
    if ground_y is None:
        res.warnings.append("没有稳定识别到人的全身，无法区分触球和落地反弹，高度改为相对触球点")

    touches_all: list[dict] = []
    arcs_all: list[Arc] = []
    ball_out: dict[int, tuple[float, float, float]] = {}

    for seg in _segments(frames, K):
        if len(seg) < 6:
            continue
        f0, f1 = seg[0], seg[-1]
        fr = np.arange(f0, f1 + 1)
        t = fr / fps
        obs = np.isin(fr, seg)
        ox = np.array([track[f].x for f in seg])
        oy = np.array([track[f].y for f in seg])
        orr = np.array([track[f].r for f in seg])
        x = np.interp(fr, seg, ox)
        y = np.interp(fr, seg, oy)
        y_raw = y.copy()  # 检测到的位置（只在漏检帧插值）；下面第 3 步会用拟合曲线改写 y
        r = np.interp(fr, seg, orr)
        ys = np.convolve(np.pad(y, 1, mode="edge"), np.ones(3) / 3, mode="valid")

        # 1) 粗找触球（y 向下为正）。两个判据取并集：
        #    a. y 的局部极大，突出度 ≥ 0.45 个球半径（约 5 cm）
        #    b. 冲量：纵向速度由"下落或静止"突变为"上升"，变化量 ≥ 4 帧重力造成的速度变化
        #       （覆盖"球刚弹起一点就被再碰"这种突出度很小的情况）
        peaks_a, _ = find_peaks(ys, prominence=0.45 * r_ref, distance=max(2, int(0.12 * fps)))
        peaks = set(peaks_a.tolist())
        if len(ys) >= 7:
            v = np.gradient(ys)
            d2 = np.diff(ys, 2)
            a_obs = max(0.5, float(np.median(d2[np.abs(d2) < 3 * np.median(np.abs(d2)) + 1e-6])))
            vb = np.convolve(v, np.ones(2) / 2, mode="same")
            for i in range(2, len(ys) - 2):
                v_before, v_after = vb[i - 1], vb[i + 2]
                if v_after - v_before <= -max(4 * a_obs, 0.15 * r_ref) and v_after < -0.5 * a_obs and v_before > -1.5 * a_obs:
                    j = i - 1 + int(np.argmax(ys[i - 1:i + 2]))
                    if all(abs(j - p) > 3 for p in peaks):
                        peaks.add(j)
        peaks = np.array(sorted(peaks), dtype=int)
        cut = [0, *peaks.tolist(), len(fr) - 1]

        # 2) 逐段拟合抛物线
        arcs: list[Arc] = []
        for a, b in zip(cut, cut[1:]):
            lo, hi = (a + 1 if a > 0 else a), (b - 1 if b < len(fr) - 1 else b)
            m = obs[lo:hi + 1]
            fit = _fit_arc(t[lo:hi + 1][m], x[lo:hi + 1][m], y[lo:hi + 1][m], r_ref)
            arcs.append(Arc(float(t[a]), float(t[b]), *(fit if fit else (None, None)), int(m.sum())))

        # 3) 用拟合曲线补帧、去抖；离拟合太远的检测当离群点
        for k, arc in enumerate(arcs):
            if arc.ay is None or arc.ay[0] <= 0:  # 没拟合上或"向上加速"的段不可信，保留线性插值
                continue
            a, b = cut[k], cut[k + 1]
            for i in range(a, b + 1):
                fy, fx = arc.y(t[i]), arc.x(t[i])
                if not obs[i] or abs(fy - y[i]) > 0.8 * r_ref or abs(fx - x[i]) > 0.8 * r_ref:
                    x[i], y[i] = fx, fy
        for i, f in enumerate(fr):
            ball_out[int(f)] = (float(x[i]), float(y[i]), float(r[i]))

        # 4) 精确触球时刻 = 相邻两段抛物线的交点
        for k, p in enumerate(peaks):
            a1, a2 = arcs[k], arcs[k + 1]
            tg = float(t[p])
            tt = None
            if a1.ay is not None and a2.ay is not None and a1.ay[0] > 0 and a2.ay[0] > 0:
                tt = _intersect(a1.ay, a2.ay, tg, 2.0 / fps)
            # 附近真正检测到的最低点：精确化结果不能比它高太多，判断落地也用它。
            # 只看真检测到的帧、不用拟合补出来的值：球停在脚背上（停球）时不是抛物线，
            # 拟合曲线会一路外推到地面，把停球误判成落地。
            win = [i for i in range(max(0, p - 2), min(len(fr), p + 3)) if obs[i]]
            y_low = float(max(y_raw[i] for i in win)) if win else float(np.max(y[max(0, p - 2):p + 3]))
            if tt is not None:
                tx = float((a1.x(tt) + a2.x(tt)) / 2)
                ty = float((a1.y(tt) + a2.y(tt)) / 2)
                if ty < y_low - 0.3 * r_ref:
                    tt = None
            if tt is None:
                j = max(0, p - 2) + int(np.argmax(y[max(0, p - 2):p + 3]))
                tx, ty, tt = float(x[j]), float(y[j]), float(t[j])
            # 触点最低处：两段抛物线的交点能补上两帧之间没拍到的最低点（快速下落时一帧能走一个球径），
            # 但最多只能比检测到的最低点低"触点前一帧的下落距离 + 0.3 个球半径"。
            # 停球时球几乎不动，交点却是外推出来的，可能比真实位置低一个球径以上，这时只用检测值。
            v_fall = max([0.0] + [float(y_raw[i] - y_raw[i - 1]) for i in range(max(1, p - 3), min(len(fr), p + 2))])
            y_bot = max(ty, y_low) if ty <= y_low + 0.3 * r_ref + v_fall else y_low
            # 落下高度 / 弹起高度（像素），用来区分贴地的脚触球和落地反弹
            lo_b, hi_a = cut[k], cut[k + 2]
            drop = y_bot - float(np.min(y[lo_b:p + 1]))
            rise = y_bot - float(np.min(y[p:hi_a + 1]))
            touches_all.append({"t": tt, "x": tx, "y": ty, "y_low": y_bot, "drop": drop, "rise": rise,
                                "frame": int(round(tt * fps)), "arc_before": a1, "arc_after": a2})
        arcs_all.extend(arcs)

    # ---- 重力反推比例尺，做交叉校验 ----
    acc = [2 * a.ay[0] for a in arcs_all if a.ay is not None and a.ay[0] > 0 and a.n_obs >= 6 and a.t1 - a.t0 >= 0.25]
    px_per_m_g = float(np.median(acc) / G) if len(acc) >= 3 else None
    if px_per_m_g:
        ratio = px_per_m_g / px_per_m
        if ratio < 0.25:
            res.warnings.append("重力加速度明显偏小，可能是慢动作视频，高度可能不准")
        elif not 0.6 < ratio < 1.6:
            res.warnings.append(f"球径比例尺和重力比例尺差 {ratio:.2f} 倍，高度仅供参考")

    # ---- 分类：触球 / 落地 / 手接 ----
    # 有关键点时（主路径）：
    #   地面线 = 支撑脚脚底（最低的脚趾、脚跟点）在触点前后的中位数，比人框底边准：近景、仰拍、横屏截断都不受影响。
    #   落地：球底比地面线低 0.2 个球半径以上，并且是"从地上挑起/在地上滚"（落下不足 1 个球半径）
    #        或"被动反弹"（弹起不到落下的 75%；足球在草地、水泥地上只能弹回 25%–70%，脚触球不受这个限制）。
    #        球在脚前面、离镜头更近时画面里会比脚底低一点，0.2 个球半径的余量就是给它的。
    #        实测（16 段 Pexels）：真落地、挑起的球底都在地面线下 0.23 以上；贴地脚触球大多在地面线以上。
    #   离人太远：球离人框超过 18 个球半径（约 2 米）时不可能是人碰的，是球自己在地上弹（被踢远之后常见）。
    #   手接：球贴着前臂或托在手掌上（离前臂 1.1 个球半径内），而脚、膝都在 2.2 个球半径外、头不在附近 → 不计数，断开连击。
    #        大腿触球时手常垂在大腿边、离球很近，靠"膝盖在不在附近"区分（实测大腿触球 ≤ 1.6，手托球 ≥ 2.8）。
    # 没有关键点时退回旧规则：人框底边当地面线，再用"弹起高度"区分贴地的脚触球。
    body = Body(kpts) if kpts else None  # 只拍到腰以上时没有脚底点：仍用关键点判断手接，地面线退回旧规则
    touches, bounces = [], []
    for tc in touches_all:
        f = min(max(tc["frame"], 0), n_frames - 1)
        drop, rise, gap = tc["drop"], tc["rise"], None
        near = body.near(f, tc["x"], tc["y_low"], r_ref) if body is not None else None
        g = body.ground(f) if body is not None and body.sole_all is not None else None
        kind = "touch"
        pb = persons[f] if f < len(persons) else None
        away = None
        if pb is not None:
            dx = max(pb[0] - tc["x"], 0.0, tc["x"] - pb[2])
            dy = max(pb[1] - tc["y_low"], 0.0, tc["y_low"] - pb[3])
            away = float(np.hypot(dx, dy)) / r_ref
        low = None
        alone = not any(persons[ff] is not None for ff in range(max(0, f - P["alone"]), min(len(persons), f + P["alone"] + 1)))
        if (away is not None and away > P["away"]) or alone:
            kind = "ground"  # 离人太远，或者前后都没有人（人走出画面后球自己在弹）
        elif near is not None and near["forearm"] < P["hand_fa"] and min(near["foot"], near["thigh"]) > P["hand_limb"]:
            kind = "hand"
        elif near is not None and min(near["head"], near["chest"]) < P["carry_near"] and _carried(body, ball_out, f, r_ref, fps, near):
            kind = "stall"  # 球被头、颈托着走（平衡、停球）：不算触球，也不断连击
        elif near is not None and near["foot"] < 99 and min(near[k] for k in ("foot", "thigh", "head", "chest")) >= P["far_part"]                 and rise < P["far_k"] * drop:
            kind = "ground"  # 被动反弹、离人的各部位都远：球没碰到人（在镜头前落地、撞网、地上的球磕一下）
        elif g is not None:
            gap = g - (tc["y_low"] + r_ref)
            low = _low_fraction(body, ball_out, f, r_ref, fps)
            if gap < P["g_gap"] * r_ref and (drop < P["g_lift"] * r_ref or rise < P["g_k"] * drop):
                kind = "ground"
            elif gap < r_ref and low is not None and low >= P["low_frac"]:
                kind = "ground"  # 球在地上滚、拨、带：前后一段时间都贴着地，不是颠球
            elif drop < P["rest_drop"] * r_ref and gap < P["rest_gap"] * r_ref:
                kind = "ground"  # 球本来停在地上，被脚挑起来：不算触球（停在脚背上再挑起的球离地面高，照样算）
        else:
            gap = ground_y - (tc["y_low"] + r_ref) if ground_y is not None else None
            near_ground = gap is not None and gap < 0.6 * r_ref
            a1 = tc["arc_before"]
            energetic = drop >= r_ref and rise >= 0.85 * drop
            kicked = gap is not None and gap >= -0.25 * r_ref and (
                (gap >= 0 and a1.ay is not None and a1.ay[0] > 0 and drop >= r_ref and rise > 1.5 * drop) or energetic)
            if near_ground and not kicked:
                kind = "ground"
        if debug is not None:
            debug.setdefault("contacts", []).append({
                "frame": tc["frame"], "x": round(tc["x"], 1), "y": round(tc["y"], 1), "y_low": round(tc["y_low"], 1), "r": round(r_ref, 1),
                "gap_r": None if gap is None else round(gap / r_ref, 2), "drop_r": round(drop / r_ref, 2), "rise_r": round(rise / r_ref, 2),
                "kind": kind, "bounce": kind != "touch", "low": None if low is None else round(low, 2), "near": None if near is None else {k: round(v, 2) for k, v in near.items()}})
        if kind == "stall":
            continue
        if kind != "touch":
            tc["kind"] = kind
            bounces.append(tc)
        else:
            if near is not None and min(near[k] for k in ("foot", "thigh", "head", "chest")) < 3.0:
                tc["part"] = min(("foot", "thigh", "head", "chest"), key=lambda k: near[k])
            else:
                tc["part"] = _part(tc["y"], persons[f] if f < len(persons) else None)
            touches.append(tc)

    # ---- 连击：中间落地、用手接或间隔过长就断 ----
    # 慢动作素材里两次触球之间的视频时长会被拉长：按重力反推的慢放倍数放宽"间隔过长"（最多 6 倍）。
    # 重力比例尺不可靠（<3 段可用的抛物线）时不放宽。
    slow = 1.0
    if px_per_m_g:
        slow = float(np.clip((px_per_m / px_per_m_g) ** 0.5, 1.0, 6.0))
    combos: list[list[dict]] = []
    for tc in touches:
        new = not combos
        if combos:
            prev = combos[-1][-1]
            if tc["t"] - prev["t"] > combo_break_s * slow or any(prev["t"] < b["t"] < tc["t"] for b in bounces):
                new = True
        if new:
            combos.append([tc])
        else:
            combos[-1].append(tc)

    # ---- 最高点：每次触球后的那一段抛物线顶点 ----
    apex = []
    for ci, combo in enumerate(combos):
        for n, tc in enumerate(combo, 1):
            tc["n"], tc["combo"] = n, ci
            arc = tc["arc_after"]
            if arc.ay is None or arc.ay[0] <= 0:
                continue
            ta = -arc.ay[1] / (2 * arc.ay[0])
            if not tc["t"] < ta < arc.t1 + 0.05:
                continue
            ya, xa = float(arc.y(ta)), float(arc.x(ta))
            base = ground_y if ground_y is not None else tc["y"]
            h = (base - ya) / px_per_m
            apex.append({"t": round(float(ta), 3), "x": round(xa, 1), "y": round(ya, 1), "height_m": round(max(0.0, h), 2), "touch": n, "combo": ci})

    # ---- 高光：最高的一颠，前后各留一点 ----
    if apex:
        top = max(apex, key=lambda a: a["height_m"])
        combo = combos[top["combo"]]
        k = top["touch"] - 1
        t_from = combo[k]["t"] - 0.4
        t_to = combo[k + 1]["t"] + 0.4 if k + 1 < len(combo) else top["t"] + 0.9
        res.highlight = {"from": round(max(0.0, t_from), 2), "to": round(min(n_frames / fps, t_to), 2)}

    res.ball = [[f, round(v[0], 1), round(v[1], 1), round(v[2], 1)] for f, v in sorted(ball_out.items())]
    res.touches = [
        {"t": round(tc["t"], 3), "x": round(tc["x"], 1), "y": round(tc["y"], 1), "part": tc["part"], "n": tc["n"], "combo": tc["combo"]}
        for tc in touches
    ]
    res.bounces = [{"t": round(b["t"], 3), "x": round(b["x"], 1), "y": round(b["y"], 1), "kind": b.get("kind", "ground")} for b in bounces]
    res.apex = apex
    res.combos = [{"from": round(c[0]["t"], 3), "to": round(c[-1]["t"], 3), "count": len(c)} for c in combos]
    res.calib = {
        "ball_radius_px": round(r_ref, 1),
        "px_per_m": round(px_per_m, 1),
        "px_per_m_gravity": round(px_per_m_g, 1) if px_per_m_g else None,
        "ground_y": round(ground_y, 1) if ground_y is not None else None,
    }
    return res
