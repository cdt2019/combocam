"""篮球运球计数原型：球每碰一次地算运一下球。

球的轨迹用颠球那套（整帧检测 + 在线复检 + Viterbi 选轨迹，见 combocam/track.py），这里不用颠球的抛物线平滑——
那会把触地的尖角磨圆——直接用选中的原始检测。
- 触地 = 球的竖直位置（画面 y，向下为正）的局部最低点，而且是"V"字形的急转：下落速度和上升速度都快
  （按球半径算，每秒至少 P["v_min"] 个半径；前后各看 2 帧，触地常落在两帧之间），和手在上面接球时缓慢减速、转向不一样；
- 在腿的高度以下：球的底边（y + r）低于膝盖（关键点），或者没有膝盖时低于人框高度的 P["low_frac"]；
- 靠近人：水平方向在人框左右扩出 P["near_w"] 个框宽以内（投篮、传球后球滚走、别人的球不算）；
- 触地那一刻球常常糊掉没检测到：漏检不超过 P["gap_max"] 帧时先线性补上找波峰，波峰落在漏检里或紧挨着漏检，就用漏检前、
  后的观测点各连一条直线（前后各看 1–3 帧，取最快的速度），交点当触地时刻（inferred）；
- 更长的漏检（不超过 P["long_gap_s"] 秒）：漏检前快速下落、漏检后快速弹起，也算一次触地——横屏素材球碰地时半个出了
  画面底边、球从身后绕过去被挡住，都是这样；
- 球的轨迹：明显太小的检测不要——篮球直径约人高的 0.13（24 cm / 1.8 m），按这一帧的人框高度算出预期半径，小于它
  SEL["r_lo"] 倍的去掉（人走近走远、镜头推拉时跟着人框变；没有人框就按整段的参考半径）。不设上限：只拍到腿、或者球离
  镜头近时，球比按人框估的大三四倍（4742388、8337368）；
  运球时球常常离开画面（只拍到腿时球每一下都升到画面外），选轨迹时长时间没球的跳过代价调低，不然前面一段会被丢掉；
  球离镜头近时比预期大一两倍，选轨迹时不再因为大小扣分（size_pen = 0）；
- 球碰到画面底边（横屏素材的真实画面底边、竖屏的画布底边，bottom）时只框到露出来的那部分，球心偏高、半径偏小——
  这时用球的上沿加附近帧的正常半径当球心；
- 门槛（v_min、弹起高度 prom_r）按这一刻前后 0.5 秒的球半径算，不按整段的：球离镜头远近、暗处检测框只框到亮的半截，
  都会让画面上的速度、高度差一两成；"V"字两边一快一慢也算，快的够 v_min、慢的够 P["weak"] 倍（斜着拍、在最低处停了一下）；
- 连续运球：相邻两次触地的间隔不超过 P["break_k"] 倍最近的节奏（至少 P["break_floor"] 秒），否则断开；这组还不到 3 个间隔时，
  节奏取这几个间隔和整段节奏里大的（一个间隔说明不了节奏）。标注过的片段里连续运球的间隔最长 1.6 秒（慢运球），
  拿着球停下来的至少 2.1 秒，break_floor 取 1.8 秒。一组至少 P["min_group"] = 2 下（停下来以后只运两下也算，只有一下的不算）；
- 触地在画面底边外（球的下半截出了画面、触地那几帧检测不到）：
  · 球碰到底边时检测框只有露出来的那截，按上沿加附近的正常半径修正球心；
  · 触地落在漏检里时，弹起高度按两段直线的交点（触地点）算，不按看得见的最低点；
  · 选轨迹时低分的检测（球糊了、半个出了画面，0.1–0.2 分）也要：SEL["bias"] 从颠球的 -0.25 调到 -0.15；
  · 帧率换算的重复帧（dup_frames）先去掉、在原视频的帧上数：重复帧夹在触地前后会让那一边的速度只剩一半；
  · 球在最低处停一两帧（差不到 P["flat_r"] 个半径）时，量速度跳过这段平底。

每一下的动作（classify_moves，只在人面对镜头时判）：
- 用哪只手：球在这一下前、后两个最高点（手托球的地方）在身体中线（两髋中点）的哪一边；离中线太近（不到 MOVE["side_min"]
  个人框宽）就看不出来。前后两只手不一样 = 换手；
- 换手的话，交叉（体前变向）还是胯下：胯下时球从两腿之间穿过去，从镜头看会有一段在小腿后面——触地后几帧
  （从前往后穿）或者触地前（从后往前穿），球在膝盖以下、贴着小腿的地方没检测到或者突然变小（被挡住一部分）。
  交叉时球一直在腿前面。触地点在两脚连线前面多远也有关系，但机位低时差别被压扁（5586698 交叉的触地点和胯下一样），只当参考；
- 背后：球从身体一侧的髋部高度躲到身后，从另一侧出来。有两种：
  · 触地也在身后、看不见：上面按"V"字找不到这一下。count_dribbles 另找一段连续漏检（3 帧到 P["btb_max_s"] 秒），
    进去前球在一侧快速往下，出来后在另一侧往上；进去、出来都在髋到膝的高度附近；球不比两髋宽太多（离镜头近的球不可能
    藏在髋后面）；不是贴着画面边。按两段直线的交点当触地时刻，标成 behind；
  · 触地看得见（在另一侧脚边）：触地前有一段这样的漏检，也判成背后；
  只在面对镜头时找。背对镜头时，球在身前被身体挡住也是这个样子。
- 转身运球、背对镜头、侧对镜头的都还分不出来（move = None）。

还没做：转身运球，双手运两个球，投篮。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .pose import (L_ANKLE, L_BIG_TOE, L_HEEL, L_HIP, L_KNEE, L_SHOULDER, L_SMALL_TOE, R_ANKLE, R_BIG_TOE, R_HEEL, R_HIP, R_KNEE,
                   R_SHOULDER, R_SMALL_TOE)
from .track import Cand, mark_static, ref_radius, select_track

P = {"v_min": 6.0, "prom_r": 1.5, "low_frac": 0.7, "near_w": 0.6, "gap_max": 4, "long_gap_s": 0.35, "break_k": 2.2,
     "break_floor": 1.8, "min_group": 2, "conf": 40, "local_r": True, "weak": 0.5, "dedup": True, "flat": 2, "flat_r": 0.2,
     "btb_max_s": 0.8, "btb_side": 0.3, "btb_side_max": 2.5, "btb_h": (-0.8, 0.8), "btb_vup": 0.5, "btb_edge": 2.0}
SEL = {"r_body": 0.065, "r_lo": 0.3, "long_base": 1.0, "long_per_frame": 0.05, "size_pen": 0.0, "bias": -0.15}
MOVE = {"apex_s": 0.7, "side_min": 0.08, "side_diff": 0.35, "conf": 30, "low": 0.5, "near": 0.9, "shrink": 0.85,
        "after": (2, 6), "after_frac": 0.4, "legs_score": 2, "before_w": 0.67, "dl_back": 0.3, "dl_front": 0.8, "edge": 1.0,
        "btb_side": 0.3, "btb_up": 0.8, "btb_hidden": 3}
MOVE_NAMES = {"same": "单手", "cross": "交叉", "legs": "胯下", "behind": "背后"}


def ball_series(cands_raw: list, fps: float, persons: list | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """缓存的每帧候选 [[x, y, r, s, src], ...]（和人框）→ 选中的球轨迹 x、y、r（没有球的帧是 NaN）和参考半径。"""
    cands = [[Cand(*c[:4], c[4] if len(c) > 4 else 0) for c in fr] for fr in cands_raw]
    n = len(cands)
    x, y, r = (np.full(n, np.nan) for _ in range(3))
    if not any(cands):
        return x, y, r, 20.0
    r_ref = ref_radius(cands)
    exp = np.full(n, r_ref)
    if persons:
        h = np.array([b[3] - b[1] if b else np.nan for b in persons], np.float64)
        for f in range(n):
            w = h[max(0, f - 5):f + 6]
            if np.isfinite(w).any():
                exp[f] = SEL["r_body"] * float(np.nanmedian(w))
    cands = [[c for c in fr if c.r >= SEL["r_lo"] * exp[f]] for f, fr in enumerate(cands)]
    sel = select_track(cands, fps, r_ref, mark_static(cands, fps, r_ref),
                       long_base=SEL["long_base"], long_per_frame=SEL["long_per_frame"], size_pen=SEL["size_pen"], bias=SEL["bias"])
    for f, j in enumerate(sel):
        if j is not None:
            c = cands[f][j]
            x[f], y[f], r[f] = c.x, c.y, c.r
    return x, y, r, r_ref


def _kp_y(kpts: list, idx: list[int], conf: int) -> np.ndarray:
    out = np.full(len(kpts), np.nan)
    for f, k in enumerate(kpts or []):
        if k:
            a = np.asarray(k, np.float32).reshape(-1, 3)
            ok = [i for i in idx if a[i, 2] >= conf]
            if ok:
                out[f] = float(a[ok, 1].mean())
    return out


def dup_frames(persons: list | None, kpts: list | None, n: int) -> np.ndarray:
    """帧率换算出来的重复帧（24/25 帧的视频转成 30 帧时每 4–5 帧重复一帧）：人框或关键点和上一帧一模一样（同一张图
    检测出来的结果逐位相同，不同的帧几乎不可能）。"""
    dup = np.zeros(n, bool)
    for f in range(1, n):
        for seq in (persons, kpts):
            if seq and seq[f] and seq[f] == seq[f - 1]:
                dup[f] = True
    # 帧率换算不会连着重复两帧（原视频 15 帧以上），重复帧也不会超过四成；不像的（定格画面、人工构造的数据）就不当重复帧
    if dup.mean() > 0.4 or (dup[1:] & dup[:-1]).any():
        return np.zeros(n, bool)
    return dup


def count_dribbles(x: np.ndarray, y: np.ndarray, r: np.ndarray, r_ref: float, fps: float, persons: list | None = None,
                   kpts: list | None = None, params: dict | None = None, bottom: float | None = None,
                   width: float | None = None) -> dict:
    """返回 {dribbles: [{t, frame, x, y, n, group, inferred, hand_from, hand_to, move}], groups: [{from, to, count, rate}],
    total, best, period_s, view, moves}。
    bottom：画面里真实内容的底边（像素 y），球碰到它时按上沿修正球心；width：画面宽（判动作时球贴着左右边不算被腿挡住）。

    重复帧和上一帧是同一时刻：先去掉（上一帧没检到球、重复帧检到了，就算到上一帧上），在原视频的帧上数，帧号再换回来。
    不去的话，重复帧夹在触地前后会让那一边的速度只剩一半（6070824 第 278 帧前面正好是重复帧，下落速度量出来几乎是 0）。"""
    p = {**P, **(params or {})}
    n = len(y)
    y, r = _fix_bottom(y, r, r_ref, fps, bottom)
    dup = dup_frames(persons, kpts, n) if p["dedup"] else np.zeros(n, bool)
    if not dup.any():
        res = _count(x, y, r, r_ref, fps, persons, kpts, p, width, bottom)
    else:
        xk, yk, rk = x.copy(), y.copy(), r.copy()
        for f in np.flatnonzero(dup):
            if not np.isfinite(yk[f - 1]) and np.isfinite(yk[f]):
                xk[f - 1], yk[f - 1], rk[f - 1] = xk[f], yk[f], rk[f]
        keep = np.flatnonzero(~dup)
        sub = lambda seq: [seq[i] for i in keep] if seq is not None else None  # noqa: E731
        res = _count(xk[keep], yk[keep], rk[keep], r_ref, fps * len(keep) / n, sub(persons), sub(kpts), p, width, bottom)
        for h in res["dribbles"]:
            h["frame"] = int(keep[min(h["frame"], len(keep) - 1)])
            h["t"] = round(h["frame"] / fps, 3)
        for gi, g in enumerate(res["groups"]):
            ts = [h["t"] for h in res["dribbles"] if h["group"] == gi]
            g["from"], g["to"] = ts[0], ts[-1]
        res["ball_frames"] = int(np.isfinite(y).sum())
    # 动作在原来的帧上判（重复帧里球被腿挡住也算挡住的帧，去掉会让胯下的证据少一半）
    box = np.array([b if b else [np.nan] * 4 for b in (persons or [None] * n)], np.float64).reshape(n, 4)
    res["view"] = classify_moves(res["dribbles"], x, y, r, fps, box, kpts, (width, bottom))
    res["moves"] = {k: sum(1 for h in res["dribbles"] if h.get("move") == k) for k in MOVE_NAMES}
    return res


def _fix_bottom(y: np.ndarray, r: np.ndarray, r_ref: float, fps: float, bottom: float | None) -> tuple[np.ndarray, np.ndarray]:
    """球碰到画面底边时检测框只框到露出来的那截：用上沿加附近帧的正常半径当球心。"""
    n = len(y)
    if bottom is None or not np.isfinite(y).any():
        return y, r
    win = max(1, int(0.5 * fps))
    rr = np.array([np.nanmedian(r[max(0, f - win):f + win + 1]) if np.isfinite(r[max(0, f - win):f + win + 1]).any()
                   else np.nan for f in range(n)])
    cut = np.isfinite(y) & (y + r >= bottom - 0.15 * r_ref) & (r < rr)
    return np.where(cut, y - r + rr, y), np.where(cut, rr, r)


def _count(x: np.ndarray, y: np.ndarray, r: np.ndarray, r_ref: float, fps: float, persons: list | None,
           kpts: list | None, p: dict, width: float | None, bottom: float | None = None) -> dict:
    """数触地、分组（不判动作）。y、r 已经按画面底边修正过；bottom 只用来判断背后运球的球是不是出了画面。"""
    n = len(y)
    knee = _kp_y(kpts, [L_KNEE, R_KNEE], p["conf"]) if kpts else np.full(n, np.nan)
    box = np.array([b if b else [np.nan] * 4 for b in (persons or [None] * n)], np.float64).reshape(n, 4)
    # 门槛按这一刻附近的球大小（前后 0.5 秒检测半径的中位数）算：人走近走远、球离镜头远近不同时，同样的速度、
    # 弹起高度在画面上差一两成（10341431 暗光片段里球离得远的那几下半径只有整段中位数的八成多，刚好卡在门槛下面）
    win_r = max(1, int(0.5 * fps))
    r_loc = np.full(n, r_ref)
    if p["local_r"]:
        for f in range(n):
            seg = r[max(0, f - win_r):f + win_r + 1]
            if np.isfinite(seg).any():
                r_loc[f] = float(np.nanmedian(seg))
    v_min_at = p["v_min"] * r_loc / fps  # 像素/帧
    v_min = p["v_min"] * r_ref / fps

    def low_enough(f: int, yb: float) -> bool:
        if np.isfinite(knee[f]):
            return yb >= knee[f]
        b = box[f]
        if np.isfinite(b[1]):
            return yb >= b[1] + p["low_frac"] * (b[3] - b[1])
        return True

    def near(f: int, xb: float) -> bool:
        b = box[f]
        if not np.isfinite(b[0]):
            return True
        w = b[2] - b[0]
        return b[0] - p["near_w"] * w <= xb <= b[2] + p["near_w"] * w

    obs = np.isfinite(y)
    hits: list[dict] = []

    def gap_at(f: int) -> tuple[int, int] | None:
        """f 所在的、或紧挨着 f 的漏检段 [起, 止]。"""
        if not obs[f]:
            s = e = f
        elif f > 0 and not obs[f - 1]:
            s = e = f - 1
        elif f + 1 < n and not obs[f + 1]:
            s = e = f + 1
        else:
            return None
        while s > 0 and not obs[s - 1]:
            s -= 1
        while e + 1 < n and not obs[e + 1]:
            e += 1
        return s, e

    def flat(v: np.ndarray, okv: np.ndarray, a: int, sign: int) -> int:
        """a 往前（sign=-1）/往后（+1）紧挨着、和 a 差不到 P["flat_r"] 个半径的点有几个（最多 P["flat"] 个）：球在最低处
        停一两帧（压扁、滚一下），量速度时跳过这段平底。"""
        j = 0
        while j < p["flat"] and 0 <= a + sign * (j + 1) < n and okv[a + sign * (j + 1)]                 and abs(v[a + sign * (j + 1)] - v[a]) < p["flat_r"] * r_loc[a]:
            j += 1
        return j

    def speed(a: int, sign: int) -> float | None:
        """观测点 a 和它前面（sign=-1）/后面（+1）1–3 帧内（跳过平底）的观测点之间的平均速度，取最快的（向 a 靠近地面为正）。"""
        ks = range(1, 4 + flat(y, obs, a, sign))
        vs = [(y[a] - y[a + sign * k]) / k for k in ks if 0 <= a + sign * k < n and obs[a + sign * k]]
        return max(vs, default=None)

    def bridge(s: int, e: int) -> tuple[float, float, float, float] | None:
        """漏检段 [s, e] 前后各连一条直线求交点 → (触地时刻, 那时的 y, 下落速度, 弹起速度)。"""
        a1, b0 = s - 1, e + 1
        if a1 < 0 or b0 >= n:
            return None
        v_dn, v_up = speed(a1, -1), speed(b0, 1)
        if v_dn is None or v_up is None or v_dn <= 0 or v_up <= 0:
            return None
        fc = (y[b0] - y[a1] + v_dn * a1 + v_up * b0) / (v_dn + v_up)
        if not a1 <= fc <= b0:
            return None
        return fc, y[a1] + v_dn * (fc - a1), v_dn, v_up

    def add(fc: float, yc: float, f: int, inferred: bool) -> None:
        fi = int(round(fc))
        rc = float(r[f]) if np.isfinite(r[f]) else r_ref
        xc = float(x[f]) if np.isfinite(x[f]) else float(np.nanmean(x[max(0, f - 3):f + 4]))
        yb = yc + rc
        if low_enough(fi, yb) and near(fi, xc):
            hits.append({"frame": fi, "x": xc, "y": float(yb), "inferred": bool(inferred)})
    # 短漏检（≤ gap_max 帧）线性补上，好在整条曲线上找波峰；补出来的点不当真，触地落在漏检里时下面另算
    yi = y.copy()
    good = np.flatnonzero(obs)
    for a, b in zip(good, good[1:]):
        if 1 < b - a <= p["gap_max"] + 1:
            yi[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [y[a], y[b]])
    ok = np.isfinite(yi)
    yz = np.where(ok, yi, np.nanmin(yi) if ok.any() else 0)  # 长漏检处当成很高，不会成为波峰
    peaks, props = find_peaks(yz, prominence=p["prom_r"] * float(np.min(r_loc)) / 2, distance=max(2, int(0.12 * fps)))
    for f, prom in zip(peaks, props["prominences"]):
        if not ok[f]:
            continue
        g = gap_at(f)
        br = bridge(*g) if g and g[1] - g[0] + 1 <= p["gap_max"] else None
        # 弹起高度按触地点算：触地落在漏检里时，看得见的最低点比触地点高（球出了画面底边、糊掉），用两段直线的交点
        if prom + (max(0.0, br[1] - yi[f]) if br else 0.0) < p["prom_r"] * r_loc[f]:
            continue
        if br:
            # 触地落在漏检里或紧挨着漏检：用漏检两边的直线（线性补出来的点会把下落、弹起磨平）
            fc, yc, v_dn, v_up = br
            inferred = not (obs[int(round(fc))] and abs(fc - round(fc)) < 0.25)
        else:
            # "V"字形：前后的平均速度都快。前后各取 1、2、3 帧的平均速度里最大的——触地常落在两帧之间（只看相邻一帧会把
            # 一边算成几乎不动），25 帧的视频转成 30 帧时每 6 帧重复一帧（5586698 触地后正好重复一帧，2 帧平均速度只剩一半）
            fc, yc, inferred = float(f), float(yi[f]), not obs[f]
            v_dn = max([(yc - yi[f - k]) / k for k in range(1, 4 + flat(yi, ok, f, -1)) if f - k >= 0 and ok[f - k]], default=0.0)
            v_up = max([(yc - yi[f + k]) / k for k in range(1, 4 + flat(yi, ok, f, 1)) if f + k < n and ok[f + k]], default=0.0)
        # 一边快、一边慢一点也算：快的那边要够 v_min，慢的那边够一半（斜着拍下去的、球在最低处停了一两帧的，竖直方向只剩
        # 一半速度，10341431 第 122、148 帧）。手在上面接球的转向两边都慢，还是不算
        vm = v_min_at[f]
        if max(v_dn, v_up) >= vm and min(v_dn, v_up) >= p["weak"] * vm:
            add(fc, yc, f, inferred)
    # 更长的漏检：前面快速下落、后面快速弹起
    long_gap = int(round(p["long_gap_s"] * fps))
    f = 0
    while f < n:
        if obs[f]:
            f += 1
            continue
        s = f
        while f < n and not obs[f]:
            f += 1
        e = f - 1
        if p["gap_max"] < e - s + 1 <= long_gap:
            br = bridge(s, e)
            if br and min(br[2], br[3]) >= v_min:
                add(br[0], br[1], s - 1, True)
    # 背后运球（只在面对镜头时找）：球在髋部高度一侧快速往下、躲到身后，弹地后从另一侧往上出来。触地在身后，
    # 看不见（或者只在两腿之间露一下，离镜头远、显得小），触地点在画面上比脚高，上面的找法都数不到
    if kpts and facing(kpts) == "front":
        K = [np.asarray(k, np.float32).reshape(-1, 3) if k else None for k in kpts]
        btb_max = int(round(p["btb_max_s"] * fps))
        f = 0
        while f < n:
            if obs[f]:
                f += 1
                continue
            s = f
            while f < n and not obs[f]:
                f += 1
            e = f - 1
            a1, b0 = s - 1, f
            if not 3 <= e - s + 1 <= btb_max or a1 < 0 or b0 >= n or K[a1] is None or K[b0] is None:
                continue
            if any(s - 2 <= h["frame"] <= e + 2 for h in hits):
                continue
            ra, rb = (_behind_rel(K[j], x[j], y[j], r[j], p["conf"]) for j in (a1, b0))
            if ra is None or rb is None or not (ra[2] and rb[2]):
                continue
            (sa, ha, _), (sb, hb, _) = ra, rb
            if sa * sb >= 0 or min(abs(sa), abs(sb)) < p["btb_side"] or max(abs(sa), abs(sb)) > p["btb_side_max"]:
                continue
            if not (p["btb_h"][0] <= ha <= p["btb_h"][1] and p["btb_h"][0] <= hb <= p["btb_h"][1]):
                continue
            v_dn, v_up = speed(a1, -1), speed(b0, 1)
            if v_dn is None or v_up is None or v_dn < v_min or v_up < p["btb_vup"] * v_min:
                continue
            # 球是出了画面（横屏素材的底边、左右边），不是躲到身后
            m = p["btb_edge"] * r_ref
            if bottom is not None and max(y[a1] + r[a1], y[b0] + r[b0]) > bottom - m:
                continue
            if width is not None and min(x[a1], x[b0]) < m or width is not None and max(x[a1], x[b0]) > width - m:
                continue
            br = bridge(s, e)
            fi = int(round(br[0])) if br else (s + e) // 2
            a = K[fi] if K[fi] is not None else K[a1]
            feet = [a[i, 1] for i in (L_ANKLE, R_ANKLE) if a[i, 2] >= p["conf"]]
            hits.append({"frame": fi, "x": float((a[L_HIP, 0] + a[R_HIP, 0]) / 2),
                         "y": float(max(feet) if feet else box[fi, 3]), "inferred": True, "behind": True,
                         "hand_from": "L" if sa > 0 else "R", "hand_to": "L" if sb > 0 else "R"})
    hits.sort(key=lambda h: h["frame"])
    sep = max(2, int(0.12 * fps))
    hits = [h for k, h in enumerate(hits) if k == 0 or h["frame"] - hits[k - 1]["frame"] >= sep]
    # 3) 分组
    ts = [h["frame"] / fps for h in hits]
    T = float(np.median(np.diff(ts))) if len(ts) >= 3 else 0.6
    groups: list[list[dict]] = []
    for h, t in zip(hits, ts):
        h["t"] = round(t, 3)
        if groups:
            last = groups[-1]
            recent = [g["t"] for g in last[-6:]]
            # 这组还不到 3 个间隔时，一个间隔说明不了节奏（换手、换动作时快慢差一倍，暗处还会漏掉几下）：取它和整段节奏里大的
            beat = float(np.median(np.diff(recent))) if len(recent) >= 2 else T
            if len(recent) < 4:
                beat = max(beat, T)
            if t - last[-1]["t"] <= max(p["break_k"] * beat, p["break_floor"]):
                last.append(h)
                continue
        groups.append([h])
    kept = [g for g in groups if len(g) >= p["min_group"]]
    res: dict = {"dribbles": [], "groups": [], "period_s": round(T, 3), "r_ref": round(r_ref, 1)}
    for gi, g in enumerate(kept):
        for k, h in enumerate(g, 1):
            res["dribbles"].append({**h, "x": round(h["x"], 1), "y": round(h["y"], 1), "n": k, "group": gi})
        ivs = np.diff([h["t"] for h in g])
        res["groups"].append({"from": g[0]["t"], "to": g[-1]["t"], "count": len(g), "rate": round(60 / float(np.median(ivs)), 1)})
    res["total"] = len(res["dribbles"])
    res["best"] = max((g["count"] for g in res["groups"]), default=0)
    res["ball_frames"] = int(obs.sum())
    return res


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(1e-6, float(np.dot(ab, ab))), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def facing(kpts: list | None, conf: int = 40, share: float = 0.7) -> str | None:
    """整段人朝哪边："front" 面对镜头（左髋、左肩在画面右边），"back" 背对；侧身、转来转去的（两边都不到 share）是 None。
    不看鼻子：只拍到腿的片段也要能判。"""
    votes = {"front": 0, "back": 0}
    for k in kpts or []:
        if not k:
            continue
        a = np.asarray(k, np.float32).reshape(-1, 3)
        dx = [a[li, 0] - a[ri, 0] for li, ri in ((L_HIP, R_HIP), (L_SHOULDER, R_SHOULDER)) if min(a[li, 2], a[ri, 2]) >= conf]
        if dx:
            votes["front" if dx[0] > 0 else "back"] += 1
    total = sum(votes.values())
    best = max(votes, key=votes.get)
    return best if total and votes[best] >= share * total else None


def _feet_line(a: np.ndarray, conf: int) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """两只脚着地的点（踝、脚尖、脚跟里画面最低的），按 x 排好。"""
    pts = []
    for idx in ((L_ANKLE, L_BIG_TOE, L_SMALL_TOE, L_HEEL), (R_ANKLE, R_BIG_TOE, R_SMALL_TOE, R_HEEL)):
        ok = [a[i] for i in idx if a[i, 2] >= conf]
        if not ok:
            return None
        q = np.array(ok)
        pts.append((float(q[:, 0].mean()), float(q[:, 1].max())))
    return tuple(sorted(pts))  # type: ignore[return-value]


def _behind_rel(a: np.ndarray, xb: float, yb: float, rb: float, conf: int) -> tuple[float, float, bool] | None:
    """球相对身体的位置：(横向 按两髋距离、0 = 两髋中点，竖向 0 = 髋、1 = 膝，躲得到身后吗——球直径不比两髋距离大太多；
    球离镜头比人近得多时显得很大，不可能藏在髋后面)。"""
    if min(a[L_HIP, 2], a[R_HIP, 2], a[L_KNEE, 2], a[R_KNEE, 2]) < conf:
        return None
    c, w = (a[L_HIP, 0] + a[R_HIP, 0]) / 2, max(1.0, abs(a[L_HIP, 0] - a[R_HIP, 0]))
    hy, ky = (a[L_HIP, 1] + a[R_HIP, 1]) / 2, (a[L_KNEE, 1] + a[R_KNEE, 1]) / 2
    return float((xb - c) / w), float((yb - hy) / max(1.0, ky - hy)), bool(2 * rb <= BTB_SIZE * w)


BTB_SIZE = 1.3  # 球直径最多是两髋距离的几倍还算躲得到身后（真人：两髋关节约 25–30 cm，篮球 24 cm）


def classify_moves(dribbles: list[dict], x: np.ndarray, y: np.ndarray, r: np.ndarray, fps: float,
                   box: np.ndarray, kpts: list | None, frame: tuple[float | None, float | None] = (None, None)) -> str | None:
    """给每一下加 hand_from、hand_to（人自己的 "L"/"R"，None = 看不出）和 move（"same"/"cross"/"legs"/"behind"，None = 分不出），
    返回整段的朝向（facing）。只在面对镜头时判，其余 move 都是 None。frame = (画面宽, 真实画面底边)。"""
    m = MOVE
    for h in dribbles:
        if h.get("behind"):  # count_dribbles 按"躲到身后"找到的那一下，已经知道是背后
            h["move"] = "behind"
        else:
            h.update(hand_from=None, hand_to=None, move=None)
    view = facing(kpts)
    if view != "front" or not dribbles:
        return view
    n = len(y)
    K = [np.asarray(k, np.float32).reshape(-1, 3) if k else None for k in kpts]
    obs = np.isfinite(y)
    good = np.flatnonzero(obs)
    if len(good) < 2:
        return view
    xi, yi = np.interp(np.arange(n), good, x[good]), np.interp(np.arange(n), good, y[good])
    span = int(m["apex_s"] * fps)
    W, Bt = frame

    def apex(a: int, b: int) -> int | None:
        seg = y[max(0, a):max(0, b)]
        return max(0, a) + int(np.nanargmin(seg)) if len(seg) and np.isfinite(seg).any() else None

    def side(j: int | None) -> float | None:
        """球在身体中线哪边，按人框宽算；正 = 画面右边 = 面对镜头的人的左手。"""
        if j is None or not np.isfinite(box[j, 0]):
            return None
        a = K[j]
        if a is not None and min(a[L_HIP, 2], a[R_HIP, 2]) >= m["conf"]:
            c = (a[L_HIP, 0] + a[R_HIP, 0]) / 2
        else:
            c = (box[j, 0] + box[j, 2]) / 2
        return float((x[j] - c) / (box[j, 2] - box[j, 0]))

    def hidden_near_shin(frames: range, rl: float, cut: set[int]) -> int:
        """这些帧里球在膝盖以下、贴着小腿，而且没检测到或明显变小（被腿挡住）的帧数。cut 里的帧是被画面底边切掉的，不算。"""
        cnt = 0
        for g in frames:
            if not 0 <= g < n or K[g] is None or g in cut:
                continue
            a = K[g]
            pt = np.array([xi[g], yi[g]])
            # 球贴着画面边：没检测到、变小是被画面切掉，不是被腿挡住
            if pt[0] < m["edge"] * rl or (W and pt[0] > W - m["edge"] * rl) or (Bt and pt[1] > Bt - m["edge"] * rl):
                continue
            near = False
            for kn, an in ((L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE)):
                if min(a[kn, 2], a[an, 2]) < m["conf"] or pt[1] < a[kn, 1] - m["low"] * rl:
                    continue
                near |= _seg_dist(pt, a[kn, :2], a[an, :2]) < m["near"] * rl
            if near and (not obs[g] or r[g] < m["shrink"] * rl):
                cnt += 1
        return cnt

    def hidden_behind_hips(lo: int, hi: int) -> bool:
        """lo..hi（触地前）里有一段连续 MOVE["btb_hidden"] 帧以上的漏检：进去前、出来后的球在身体两侧（各离两髋中点
        MOVE["btb_side"] 个两髋距离以上）、髋到膝上面一点的高度，球的大小躲得到身后，而且不是贴着画面边。"""
        g = max(1, lo)
        while g < min(hi, n - 1):
            if obs[g]:
                g += 1
                continue
            s = g
            while g < n - 1 and not obs[g]:
                g += 1
            a1, b0 = s - 1, g
            if g - s < m["btb_hidden"] or K[a1] is None or K[b0] is None:
                continue
            ra, rb = _behind_rel(K[a1], x[a1], y[a1], r[a1], m["conf"]), _behind_rel(K[b0], x[b0], y[b0], r[b0], m["conf"])
            if ra is None or rb is None or not (ra[2] and rb[2]) or ra[0] * rb[0] >= 0 or min(abs(ra[0]), abs(rb[0])) < m["btb_side"]:
                continue
            if not all(-m["btb_up"] <= v <= 0.8 for v in (ra[1], rb[1])):
                continue
            rl = float(np.nanmedian([r[a1], r[b0]]))
            if any((W and (x[j] < m["edge"] * rl or x[j] > W - m["edge"] * rl)) or (Bt and y[j] + r[j] > Bt - m["edge"] * rl) for j in (a1, b0)):
                continue
            return True
        return False

    frames = [h["frame"] for h in dribbles]
    hand = lambda s: None if s is None or abs(s) < m["side_min"] else ("L" if s > 0 else "R")  # noqa: E731
    for i, h in enumerate(dribbles):
        if h.get("behind"):
            continue
        f = h["frame"]
        lo = max(frames[i - 1] if i else f - span, f - span)
        hi = min(frames[i + 1] if i + 1 < len(frames) else f + span, f + span, n - 1)
        ja, jb = apex(lo + 1, f), apex(f + 1, hi)
        sa, sb = side(ja), side(jb)
        if sa is None or sb is None:
            continue
        ha, hb = hand(sa), hand(sb)
        if ha is None or hb is None or ha == hb or abs(sa - sb) < m["side_diff"]:
            # 没换手：一边离中线太近看不出来时，按另一边的手
            h["hand_from"] = h["hand_to"] = ha or hb
            h["move"] = "same" if (ha or hb) else None
            continue
        h["hand_from"], h["hand_to"] = ha, hb
        seg = r[max(0, lo):hi + 1]
        rl = float(np.nanmedian(seg)) if np.isfinite(seg).any() else float(np.nanmedian(r))
        # 触地时球碰到真实画面底边：触地前后那一段漏检是球出了画面，不是被腿挡住
        cut: set[int] = set()
        if Bt and h["y"] >= Bt - m["edge"] * rl:
            g = f
            while g > 0 and not obs[g - 1]:
                g -= 1
            while g < n and (not obs[g] or g <= f):
                cut.add(g)
                g += 1
        a0, a1 = m["after"]
        a1 = max(a1, int(round(m["after_frac"] * ((jb or f) - f))))
        after = hidden_near_shin(range(f + a0, min(jb or f + a1, f + a1) + 1), rl, cut)
        before = hidden_near_shin(range(ja or f, f - 2), rl, cut)
        # 触地点在两脚连线前面多远（球半径）：明显在后面/贴着脚算胯下的证据，明显在前面算交叉的证据
        depth = 0
        fl = _feet_line(K[f], m["conf"]) if K[f] is not None else None
        if fl:
            (x1, y1), (x2, y2) = fl
            t = float(np.clip((h["x"] - x1) / (x2 - x1), 0, 1)) if x2 - x1 > 1 else 0.5
            dl = (h["y"] - (y1 + t * (y2 - y1))) / rl
            depth = 1 if dl < m["dl_back"] else (-1 if dl > m["dl_front"] else 0)
        score = after + m["before_w"] * before + depth
        h["move"] = "legs" if score >= m["legs_score"] else "cross"
        # 背后：触地前球在髋部高度躲到身后（从背后绕过去，在另一侧触地）
        if hidden_behind_hips(ja or f, f - 1):
            h["move"] = "behind"
    return view
