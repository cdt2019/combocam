"""跳绳计数原型：只看人体关键点（绳子太细太快，正常速度的 30 帧视频里大多只在每圈的一两帧里有一道淡弧线）。

- 主信号是髋部（左右髋中点）的上下起伏：每跳一次，整个人离地一次，髋部画出一个小抛物线。
- 尺度用身高（头顶关键点到脚底的中位距离）：跳绳离地 3–6 cm，约为身高的 2%–3.5%，远景近景同一套阈值。
- 一跳 = 髋部高度的一个波峰（突出度 ≥ P["prom"] 个身高），并且这一刻至少一只脚离地 ≥ P["foot_lift"] 个身高。
  只要求一只脚：左右脚交替跳（拳击步、跑步跳）时总有一只脚接近地面，每一步也算一跳；
  两只脚都没离地的起伏（屈膝晃动、说话时点头带动）不算。每只脚的地面线 = 附近一段时间里这只脚最低位置（85 分位）。
- 节奏按这段视频自己的跳速算：先粗找一遍波峰，用间隔中位数估出周期 T，去趋势窗口、断绳门槛都按 T 缩放，
  慢动作视频（一跳两三秒）也能数。
- 断绳：两跳间隔超过 P["break_k"] × 最近的节奏，连跳从 1 重新数；少于 P["min_group"] 跳的一组不算（多半是准备动作、走动）。

还没做：双摇（一跳两圈）、绊绳和空跳的区分（要看绳子或听声音）、多人入镜时选人、慢动作视频的真实跳速。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, percentile_filter
from scipy.signal import find_peaks

from .body import CONF
from .pose import (HEAD, L_BIG_TOE, L_HEEL, L_HIP, L_SHOULDER, L_SMALL_TOE, L_WRIST, NOSE, R_BIG_TOE, R_HEEL, R_HIP, R_SHOULDER,
                   R_SMALL_TOE, R_WRIST)

P = {"prom": 0.010, "min_gap": 0.20, "foot_lift": 0.006, "break_k": 1.8, "break_floor": 0.4, "min_group": 3,
     "detrend_s": 1.0, "detrend_T": 2.5, "edge_rel": 0.35, "both_rel": 0.22, "tiny_rel": 0.2, "small_rel": 0.5, "half_beat": 0.65, "reach_beats": 2.5,
     "missed": (1.6, 2.6), "arm_min": 0.5, "feet_rule": None, "absent_max_s": 2.0, "absent_phase": 0.3, "signal": "trunk", "outlier": 0.012, "multi_air_s": 0.27, "wrist_prom": 0.15, "wrist_rise": 0.02, "multi_lift_max": 0.3, "multi_arm": 0.3, "triple_air_s": 0.42,
     "bounce": None, "pass_phase": 0.55, "arm_period": None}
L_FOOT = [L_BIG_TOE, L_SMALL_TOE, L_HEEL]
R_FOOT = [R_BIG_TOE, R_SMALL_TOE, R_HEEL]


def _series(kpts: list, idx: list[int], how: str) -> np.ndarray:
    out = np.full(len(kpts), np.nan)
    for f, k in enumerate(kpts):
        if not k:
            continue
        a = np.asarray(k, np.float32).reshape(-1, 3)
        ok = [i for i in idx if a[i, 2] >= CONF]
        if ok:
            out[f] = a[ok, 1].mean() if how == "mean" else a[ok, 1].max() if how == "max" else a[ok, 1].min()
    return out


def _fill(y: np.ndarray, max_gap: int) -> np.ndarray:
    """短缺口（≤ max_gap 帧）线性插值，长缺口保持 NaN。"""
    y = y.copy()
    good = np.flatnonzero(~np.isnan(y))
    for a, b in zip(good, good[1:]):
        if 1 < b - a <= max_gap + 1:
            y[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [y[a], y[b]])
    return y


def _trunk(kpts: list, hip: np.ndarray, tall: float, fps: float, thr: float) -> np.ndarray:
    """身体上下起伏的信号：平时就是髋部；某一帧髋部明显是"孤例"——肩膀和头彼此一致、都和髋部差出 thr 个身高、
    而且差的方向相同——就换成肩膀和头的平均（身体被挡住一部分时姿态模型常把髋部猜错、乱跳）。
    肩膀、头按附近 1.5 秒里和髋部的差（滑动中位数）对齐，人往前走变大、身体前倾时也对得上。
    不直接取三者的中位数：真实视频里低头看绳、走近镜头时头和肩膀自己在动，会把停下后的动作数成跳、把最后一跳弄丢
    （4260133、4761432），这样只在髋部出错时才用另外两处。遮挡压力测试见 tools/eval/jumprope_occlusion.py。"""
    if not np.isfinite(tall) or tall <= 0:
        return hip
    win = max(3, int(round(1.5 * fps)) | 1)
    aligned = []
    for idx, how in (([L_SHOULDER, R_SHOULDER], "mean"), ([HEAD, NOSE], "min")):
        y = _fill(_series(kpts, idx, how), 3)
        dlt = hip - y
        ok = np.isfinite(dlt)
        if ok.sum() < 5:
            return hip
        off = median_filter(np.interp(np.arange(len(y)), np.flatnonzero(ok), dlt[ok]), size=win, mode="nearest")
        aligned.append(y + off)
    s_, h_ = aligned
    t = thr * tall
    d_s, d_h = hip - s_, hip - h_
    bad = (np.abs(s_ - h_) < t) & (np.abs(d_s) > t) & (np.abs(d_h) > t) & (np.sign(d_s) == np.sign(d_h))
    # 只换短的一段（≤ 0.2 秒）：关键点出错是一闪一闪的；连着好几帧都"不一致"是真的姿势变了——
    # 弯腰、前倾时肩膀和头一起往下、髋部不动，看起来也像髋部是孤例（4260133 停下后低头看绳子）
    max_run = max(3, int(round(0.2 * fps)))
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j < len(bad) and bad[j]:
                j += 1
            if j - i > max_run:
                bad[i:j] = False
            i = j
        else:
            i += 1
    out = hip.copy()
    out[bad] = 0.5 * (s_[bad] + h_[bad])
    return out


def _peaks(hip: np.ndarray, tall: float, fps: float, win_s: float, min_gap_s: float, prom: float):
    win = max(3, int(round(win_s * fps)) | 1)
    base = median_filter(np.nan_to_num(hip, nan=np.nanmedian(hip)), size=win, mode="nearest")
    h = -(hip - base) / tall  # 向上为正，单位身高
    hs = np.convolve(np.nan_to_num(h), np.ones(3) / 3, mode="same")
    pk, props = find_peaks(hs, prominence=prom, distance=max(2, int(round(min_gap_s * fps))))
    return h, pk, props["prominences"]


def _foot_lift(sole: np.ndarray, tall: float, fps: float, win_s: float) -> np.ndarray:
    """这只脚比附近的地面线（这段时间里这只脚的最低位置，85 分位）高多少，单位身高。"""
    win = max(3, int(round(win_s * fps)) | 1)
    ground = percentile_filter(np.nan_to_num(sole, nan=np.nanmedian(sole)), 85, size=win, mode="nearest")
    return -(sole - ground) / tall


def arm_period(kpts: list, fps: float, a: int, b: int) -> tuple[float | None, float]:
    """摇绳一圈的周期：a–b 帧里手腕（相对肩膀）上下位置的自相关第一个明显的峰（> 0.3）。返回 (周期秒数, 峰高)。
    取第一个峰、不取最大的峰：左右手不对称时两倍周期处的峰可能更高。"""
    y = (_fill(_series(kpts, [L_WRIST, R_WRIST], "mean"), 3) - _fill(_series(kpts, [L_SHOULDER, R_SHOULDER], "mean"), 3))[a:b]
    y = y[np.isfinite(y)]
    if len(y) < 2 * fps:
        return None, 0.0
    y = y - median_filter(y, size=int(fps) | 1, mode="nearest")
    y = (y - y.mean()) / (y.std() + 1e-9)
    ac = np.correlate(y, y, "full")[len(y) - 1:] / len(y)
    for k in range(int(0.2 * fps) + 1, min(len(ac) - 1, int(3.5 * fps))):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.3:
            return k / fps, float(ac[k])
    return None, 0.0


def wrist_phase(kpts: list, fps: float, tall: float, period: float | None = None) -> np.ndarray:
    """每帧在摇绳周期里的相位：0 = 手腕（相对肩膀）到最高点，按相邻两个最高点之间线性插值，0–1；看不清的地方是 NaN。
    实测 33 段视频（两批标注）：绳子过脚、人跳到最高的那一刻，相位都在 0.5–0.6（手往下走、绳子扫过脚底）。
    period：摇绳周期（秒，见 arm_period），给了就要求两个最高点至少隔 0.6 个周期——一圈跳两下时，垫的那一步身体一颠，
    手腕也跟着有个小起伏（2%–3% 身高，摇绳的起伏 15%–35%），不能当成一圈。"""
    n = len(kpts)
    out = np.full(n, np.nan)
    y = _fill(_series(kpts, [L_WRIST, R_WRIST], "mean"), 3) - _fill(_series(kpts, [L_SHOULDER, R_SHOULDER], "mean"), 3)
    if np.isfinite(y).sum() < fps:
        return out
    y = np.nan_to_num(y - median_filter(np.nan_to_num(y, nan=np.nanmedian(y)), size=int(fps) | 1, mode="nearest"))
    ys = np.convolve(y, np.ones(3) / 3, mode="same")
    dist = max(2, int(0.15 * fps), int(0.6 * period * fps) if period else 0)
    highs, _ = find_peaks(-ys, prominence=P["wrist_rise"] * tall, distance=dist)
    for a, b in zip(highs, highs[1:]):
        if b - a < 3 * fps:
            out[a:b] = (np.arange(a, b) - a) / (b - a)
    return out


def _interval(res: dict) -> float | None:
    """同一组里相邻两跳间隔的中位数（秒）。"""
    ivs = [b["t"] - a["t"] for a, b in zip(res["jumps"], res["jumps"][1:]) if a["group"] == b["group"]]
    return float(np.median(ivs)) if ivs else None


def rope_passes(kpts: list, fps: float, jumps: list[dict], speed: float = 1.0, feet_visible: bool = True,
                tall: float | None = None, lifts: list[float | None] | None = None, params: dict | None = None) -> list[int]:
    """每一跳绳子过脚几次：1 单摇、2 双摇、3 三摇。

    绳子转一圈手腕转一圈，数这一跳离地到下一跳离地之间手腕（相对肩膀）到最高点的次数。手腕关键点会抖，
    所以还要腾空时间够长（真实时间 ≥ P["multi_air_s"]）才认多摇：绳子在空中过脚两次，人得跳得高、在空中待得久。
    实测（2785536、2785538 同一个人，单摇转双摇）：双摇脚离地 0.12–0.16 个身高、腾空 0.30–0.40 秒、手腕最高点两两成对；
    单摇脚离地 0.05–0.07、腾空约 0.23 秒、手腕一跳一次。4 倍慢放、手腕抖的视频里零星数出 2 圈的，腾空只有 0.13–0.20 秒，被这条挡掉。
    19548255（8 倍慢放，逐帧看得见绳子在一次腾空里两次过脚）也是双摇。另外两条：
    - 每一圈手腕要抬起至少 P["wrist_rise"] 个身高（tall 给定时）：真双摇 0.023 以上（19548255 人小、光线暗时最低），
      手腕关键点抖出来的小波峰只有 0.001–0.014；
    - 第一跳到最后一跳之间手腕要有摇绳的节奏（自相关 ≥ P["multi_arm"]，见 arm_period）：多人乱跑的操场（36619323）里手乱挥，没有节奏；
    - 三摇要腾空 ≥ P["triple_air_s"] 秒，不够的按双摇；
    - 脚离地不超过 P["multi_lift_max"] 个身高（lifts 给定时）：跳大绳（8034400）时人收腿高跳，脚离地 0.4–0.7，手不摇绳、随跳甩动，
      会数出两个"最高点"。
    jumps 要有 up_t、land_t（见 combocam/jumprope_track.person_track）；脚不在画面里时腾空时间不可靠，一律按单摇。"""
    p = {**P, **(params or {})}
    out = [1] * len(jumps)
    if not feet_visible or len(jumps) < 2:
        return out
    _, strength = arm_period(kpts, fps, jumps[0]["frame"], jumps[-1]["frame"] + 1)
    if strength < p["multi_arm"]:
        return out
    y = _fill(_series(kpts, [L_WRIST, R_WRIST], "mean"), 3) - _fill(_series(kpts, [L_SHOULDER, R_SHOULDER], "mean"), 3)
    if np.isfinite(y).sum() < fps:
        return out
    y = np.nan_to_num(y - median_filter(np.nan_to_num(y, nan=np.nanmedian(y)), size=int(fps) | 1, mode="nearest"))
    ys = np.convolve(y, np.ones(3) / 3, mode="same")
    rise = p["wrist_rise"] * tall if tall else p["wrist_prom"] * np.std(ys)
    highs, _ = find_peaks(-ys, prominence=rise)  # 画面坐标 y 向下，取反后的波峰 = 手到最高点
    for i, j in enumerate(jumps):
        if j.get("inferred") or j.get("up_t") is None:
            continue
        air = (j["land_t"] - j["up_t"]) / max(speed, 1e-3)
        if air < p["multi_air_s"] or (lifts and lifts[i] is not None and lifts[i] > p["multi_lift_max"]):
            continue
        a = j["up_t"] * fps
        b = jumps[i + 1]["up_t"] * fps if i + 1 < len(jumps) and jumps[i + 1].get("up_t") is not None and jumps[i + 1]["group"] == j["group"]             else (j["land_t"] + (j["land_t"] - j["up_t"])) * fps
        out[i] = int(np.clip(np.sum((highs >= a) & (highs < b)), 1, 3 if air >= p["triple_air_s"] else 2))
    return out


def count_jumps(kpts: list, fps: float, params: dict | None = None, debug: dict | None = None, frame_h: int = 1920) -> dict:
    """计数。默认"两只脚都离地"、去掉两跳之间的小颠动（见 _count）。

    手腕的摇绳节奏清楚（自相关 ≥ P["arm_min"]）时拿它核对：绳子每过脚一次手腕转一圈。髋部每圈都起伏一次（候选间隔 ≈ 摇绳周期），
    数出来的跳却只有摇绳圈数的一半——是每一步都过绳的跑步跳、左右脚交替发力：一步髋部起伏大、一步小，小的那步单脚着地，
    被"两只脚都离地""比相邻小很多"两条规则去掉了（4761426 前排的人：手腕 0.57 秒一圈，髋部 0.58 秒起伏一次，大小交替，原来只数到 4）。
    这时改成"一只脚离地就算"、不去小颠动再数一遍，数出来的节奏对上摇绳节奏才用它。
    反过来，一圈绳子髋部起伏两次的（交替抬脚、一圈换两次脚，5469576：手腕 1.17 秒、髋部 0.57 秒），原来的规则正好数对，不动；
    手腕看不清的（逆光剪影 6573725，自相关 0.39）不核对。params["feet_rule"] 给定（both / any）时也不核对。"""
    p = {**P, **(params or {})}
    d1: dict = {}
    r = _count(kpts, fps, {**p, "feet_rule": p.get("feet_rule") or "both"}, d1, frame_h)
    r["feet_rule"] = p.get("feet_rule") or "both"
    cand = d1.get("cand", [])
    if not p.get("feet_rule") and len(cand) >= 3:
        W, strength = arm_period(kpts, fps, cand[0]["frame"], cand[-1]["frame"] + 1)
        r["arm_period_s"], r["arm_strength"] = (None if W is None else round(W, 3)), round(strength, 2)
        A = _interval(r)
        C = float(np.median(np.diff([c["t"] for c in cand])))
        # 只数到摇绳圈数的一半，或者几乎没数到（不到 3 跳时没有间隔可比）
        under = A is None or len(r["jumps"]) < 3 or 1.6 <= A / W <= 2.4 if W else False
        if W and strength >= p["arm_min"] and 0.8 <= C / W <= 1.25 and under:
            d2: dict = {}
            r2 = _count(kpts, fps, {**p, "feet_rule": "any", "small_rel": 0.0}, d2, frame_h)
            A2 = _interval(r2)
            if A2 and 0.8 <= A2 / W <= 1.25:
                r2.update({"feet_rule": "any", "arm_period_s": r["arm_period_s"], "arm_strength": r["arm_strength"]})
                r, d1 = r2, d2
        # 一圈跳两下：数出来的跳是摇绳圈数的两倍（绳子在头顶时垫一下，和过绳那下一样高，"比相邻小很多"去不掉）。
        # 只留过绳的那一下（见 _count 里的 bounce），节奏对上摇绳节奏才用
        elif W and A and strength >= p["arm_min"] and 0.4 <= A / W <= 0.62 and not p.get("bounce"):  # params["bounce"] 给定时不核对
            d2 = {}
            r2 = _count(kpts, fps, {**p, "feet_rule": "both", "bounce": "double", "arm_period": W}, d2, frame_h)
            A2 = _interval(r2)
            if A2 and 0.8 <= A2 / W <= 1.25:
                r2.update({"feet_rule": "both", "arm_period_s": r["arm_period_s"], "arm_strength": r["arm_strength"]})
                r, d1 = r2, d2
    r.setdefault("bounce", "single")
    if debug is not None:
        debug.update(d1)
    return r


def _count(kpts: list, fps: float, p: dict, debug: dict | None, frame_h: int) -> dict:
    n = len(kpts)
    hip = _fill(_series(kpts, [L_HIP, R_HIP], "mean"), 3)
    l_sole = _fill(_series(kpts, L_FOOT, "max"), 3)
    r_sole = _fill(_series(kpts, R_FOOT, "max"), 3)
    head = _series(kpts, [HEAD, NOSE], "min")
    low = np.fmax(l_sole, r_sole)
    # 脚在不在画面里：脚点可信（置信度够、在画面内）的帧占有人帧的一半以上。脚被画面底边切掉时（近景、只拍到膝盖），
    # 模型会把脚"猜"到画面外，置信度也低——这时不做离地检查，身高按头到髋的距离推算（髋在身高约 52% 处）
    has_person = sum(1 for k in kpts if k)
    feet_ok = int(np.sum(np.isfinite(low) & (low < frame_h - 2)))
    feet_visible = has_person > 0 and feet_ok >= 0.5 * has_person
    if feet_visible:
        tall = float(np.nanmedian(low - head)) if np.isfinite(low - head).sum() >= 5 else np.nan
    else:
        tall = float(np.nanmedian(hip - head)) / 0.48 if np.isfinite(hip - head).sum() >= 5 else np.nan
    res = {"jumps": [], "groups": [], "breaks": [], "body_px": None if np.isnan(tall) else round(tall, 1), "period_s": None,
           "feet_visible": bool(feet_visible)}
    if not feet_visible:
        res["warning"] = "脚不在画面里，只按髋部起伏计数（原地晃动也可能被算进去）"
    if np.isnan(tall) or np.isfinite(hip).sum() < fps:
        res["warning"] = "没有稳定识别到全身"
        res["total"], res["best"] = 0, 0
        return res
    # 第一遍粗找，估周期；慢的（一跳超过 0.6 秒，多半是慢动作）按周期放大窗口再找一遍
    sig = _trunk(kpts, hip, tall, fps, p["outlier"]) if p["signal"] == "trunk" else hip
    h, pk, prom = _peaks(sig, tall, fps, p["detrend_s"], p["min_gap"], p["prom"])
    T = float(np.median(np.diff(pk))) / fps if len(pk) >= 3 else 0.4
    if T > 0.6:
        h, pk, prom = _peaks(sig, tall, fps, max(p["detrend_s"], p["detrend_T"] * T), 0.5 * T, p["prom"])
        T = float(np.median(np.diff(pk))) / fps if len(pk) >= 3 else T
    res["period_s"] = round(T, 3)
    win_s = max(p["detrend_s"], p["detrend_T"] * T)
    lift_l, lift_r = _foot_lift(l_sole, tall, fps, win_s), _foot_lift(r_sole, tall, fps, win_s)
    cand = []
    reach = max(2, int(round(0.15 * T * fps)))  # 波峰前后这几帧里看脚有没有离地
    for i, f in enumerate(pk):
        lo, hi = max(0, f - reach), min(n, f + reach + 1)
        ll = np.nanmax(lift_l[lo:hi]) if np.isfinite(lift_l[lo:hi]).any() else np.nan
        lr = np.nanmax(lift_r[lo:hi]) if np.isfinite(lift_r[lo:hi]).any() else np.nan
        both = np.isfinite(ll) and np.isfinite(lr)
        hi_l, lo_l = (max(ll, lr), min(ll, lr)) if both else (np.nan, np.nan)
        # 两只脚都要离地：较低那只至少是较高那只的 P["both_rel"]。单脚抬起（交替抬脚时另一只脚踩着地）不算一跳——
        # 实测（5469576）绳子只在两脚都离地的那一下过脚，单脚起伏是两次过绳之间的换脚
        if p["feet_rule"] == "any":  # 一只脚离地就算（单脚交替的跑步跳，见 count_jumps）
            feet_up = both and hi_l >= p["foot_lift"]
        else:
            feet_up = both and hi_l >= p["foot_lift"] and lo_l >= p["both_rel"] * hi_l
        # 这一刻有一只脚的关键点缺了（深色鞋踩在深色地上、被别人挡住）：没法判断，和"脚不在画面里"一样只看髋部。
        # 6390396 里穿黑衣服的人后半段脚点全丢，按"两只脚都要离地"会把 28 个清清楚楚的髋部起伏只数成 12 个
        cand.append({"frame": int(f), "t": round(f / fps, 3), "prom": round(float(prom[i]), 4),
                     "lift_l": None if not np.isfinite(ll) else round(float(ll), 4),
                     "lift_r": None if not np.isfinite(lr) else round(float(lr), 4),
                     "ok": bool(not feet_visible or not both or feet_up)})
    # 起伏比前后相邻候选小很多的，分三种：两跳之间的垫步（不在节拍上，离前一个或后一个只有半拍左右）、
    # 起跳前和停下后的小动作（只有一边有大的跳）、在节拍上但跳得低的真跳（两边隔一拍左右都有大的跳）。
    # 只看大小分不开：标注的 6 段里垫步最多到邻跳的 0.35；没标注的视频里在节拍上的真跳（逐帧看过脚确实离地，8691734 第 120 帧）
    # 也有 0.36–0.45 的。所以：不到 P["tiny_rel"] 的一律不算；不到 P["small_rel"] 的，离相邻候选不到 P["half_beat"] 拍、
    # 或者前后 P["reach_beats"] 拍内有一边没有大的跳，不算。拍长按附近大的候选之间的间隔中位数（跳速会变）。
    ratio = []
    for i, c in enumerate(cand):
        nb = [cand[k]["prom"] for k in (i - 1, i + 1) if 0 <= k < len(cand)]
        ratio.append(c["prom"] / max(nb) if nb else 1.0)
    big = [k for k, c in enumerate(cand) if c["ok"] and ratio[k] >= p["small_rel"]]
    for i, c in enumerate(cand):
        if not c["ok"] or ratio[i] >= p["small_rel"]:
            continue
        hop = ratio[i] < p["tiny_rel"]
        if not hop:
            near = [cand[k]["t"] for k in big if k < i][-4:] + [cand[k]["t"] for k in big if k > i][:4]
            ivs = [y - x for x, y in zip(near, near[1:]) if not (x < c["t"] < y)]
            beat = float(np.median(ivs)) if ivs else T
            gaps = [abs(cand[k]["t"] - c["t"]) for k in (i - 1, i + 1) if 0 <= k < len(cand) and cand[k]["ok"]]
            prev_big = [cand[k]["t"] for k in big if k < i][-1:]
            next_big = [cand[k]["t"] for k in big if k > i][:1]
            edge = not (prev_big and next_big and c["t"] - prev_big[0] <= p["reach_beats"] * beat
                        and next_big[0] - c["t"] <= p["reach_beats"] * beat)
            hop = edge or (bool(gaps) and min(gaps) < p["half_beat"] * beat)
        if hop:
            c["ok"] = False
            c["small"] = True
    # 漏掉的一跳：相邻两个通过的跳隔了约两拍（P["missed"]，拍长按附近的间隔），中间有个被去掉的候选离两边都约一拍
    # （脚点被挡住、髋部起伏小了点）——补回来。不补的话这里会被当成断绳（6390396 穿黑衣服的人第 15 跳后面）。
    # 两跳之间的垫步离一边只有半拍，不会被补；交替抬脚、一圈绳子换两次脚（5469576）时通过的跳本来就隔一拍，也不会。
    acc = [k for k, c in enumerate(cand) if c["ok"]]
    for x, (ka, kb) in enumerate(zip(acc, acc[1:])):
        ta, tb = cand[ka]["t"], cand[kb]["t"]
        near = [cand[k]["t"] for k in acc[max(0, x - 3):x + 5]]
        ivs = [v - u for u, v in zip(near, near[1:]) if not (u == ta and v == tb)]
        if len(ivs) < 2:
            continue
        beat = float(np.median(ivs))
        if not p["missed"][0] * beat <= tb - ta <= p["missed"][1] * beat:
            continue
        mids = [k for k in range(ka + 1, kb) if ratio[k] >= p["tiny_rel"]
                and 0.6 * beat <= cand[k]["t"] - ta <= 1.4 * beat and 0.6 * beat <= tb - cand[k]["t"] <= 1.4 * beat]
        if mids:
            k = max(mids, key=lambda k: cand[k]["prom"])
            cand[k]["ok"] = True
            cand[k]["recovered"] = True
    if p.get("bounce") == "double":
        # 一圈跳两下：通过检查的跳按手腕相位分成相差半圈的两簇（相位乘 2 后取圆周平均，两簇落到同一处），
        # 离"过绳相位" P["pass_phase"] 近的那簇是过绳的跳，另一簇是绳子在头顶时的垫步
        res["bounce"] = "double"
        ph = wrist_phase(kpts, fps, tall, p.get("arm_period"))
        ok_k = [k for k, c in enumerate(cand) if c["ok"] and np.isfinite(ph[cand[k]["frame"]])]
        if len(ok_k) >= 4:
            z = np.mean(np.exp(4j * np.pi * ph[[cand[k]["frame"] for k in ok_k]]))
            c0 = (np.angle(z) / (4 * np.pi)) % 0.5  # 两簇的中心：c0 和 c0 + 0.5
            cd = lambda a, b: abs((a - b + 0.5) % 1 - 0.5)  # noqa: E731  圆周距离
            center = c0 if cd(c0, p["pass_phase"]) <= cd(c0 + 0.5, p["pass_phase"]) else c0 + 0.5
            res["pass_phase"] = round(float(center), 3)
            for k, c in enumerate(cand):
                if not c["ok"]:
                    continue
                f = c["frame"]
                if np.isfinite(ph[f]) and cd(ph[f], center) > 0.25:
                    c["ok"] = False
                    c["between"] = True  # 垫步
    jumps = [c for c in cand if c["ok"]]
    # 分组：间隔突然变长就断开（门槛按最近几跳的节奏；一组刚开头时按整段通过检查的跳的节奏，
    # 不能用全部候选的周期——去掉了小颠动、换脚之后，真跳的间隔可能是候选的两倍）
    T_ok = float(np.median(np.diff([j["t"] for j in jumps]))) if len(jumps) >= 3 else T
    present = np.isfinite(_series(kpts, [L_HIP, R_HIP], "mean"))  # 这一帧看得见这个人（髋部关键点可信）
    groups: list[list[dict]] = []
    for j in jumps:
        if groups:
            last = groups[-1]
            recent = last[-6:]
            ivs = [b["t"] - a["t"] for a, b in zip(recent, recent[1:])]  # 最近 5 个间隔
            beat = float(np.median(ivs)) if ivs else T_ok
            gap = j["t"] - last[-1]["t"]
            # 空档里一大半时间人不在画面里（有人从镜头前走过、被挡住、短暂出画）：不是断绳，这组接着数。
            # 空档不长（≤ P["absent_max_s"]）、空档后的跳正好落在节拍上（空档是整数拍，差不到 P["absent_phase"] 拍）时，
            # 按节拍补上看不见的那几跳（标 inferred，成片里照样计数，但不画脚下的效果）。
            # 要先于断绳门槛判断：挡住 1 拍多的空档本身不到断绳门槛（1.8 拍），会被当成一次慢一点的跳，漏数
            f0, f1 = last[-1]["frame"] + 1, j["frame"]
            if f1 > f0 and gap <= p["absent_max_s"] and (~present[f0:f1]).mean() >= 0.5:
                k = int(round(gap / beat))
                if k >= 2 and abs(gap / beat - k) <= p["absent_phase"]:
                    t_last, med = last[-1]["t"], float(np.median([x["prom"] for x in last]))
                    for i in range(1, k):
                        ti = t_last + i * gap / k
                        last.append({"t": round(ti, 3), "frame": int(round(ti * fps)), "prom": med, "inferred": True})
                last.append(j)
                continue
            if gap <= max(p["break_k"] * beat, p["break_floor"]):
                last.append(j)
                continue
        groups.append([j])
    # 一组的头尾：起伏不到这组中位数 P["edge_rel"] 的去掉（起跳前、停下后的小动作，手持镜头的晃动）
    for g in groups:
        if len(g) >= 3:
            med = float(np.median([j["prom"] for j in g]))
            while g and g[0]["prom"] < p["edge_rel"] * med:
                g.pop(0)
            while g and g[-1]["prom"] < p["edge_rel"] * med:
                g.pop()
    kept = [g for g in groups if len(g) >= p["min_group"]]
    for gi, g in enumerate(kept):
        for k, j in enumerate(g, 1):
            res["jumps"].append({"t": j["t"], "frame": j["frame"], "n": k, "group": gi, **({"inferred": True} if j.get("inferred") else {})})
        ivs = np.diff([j["t"] for j in g])
        res["groups"].append({"from": g[0]["t"], "to": g[-1]["t"], "count": len(g), "bpm": round(60 / float(np.median(ivs)), 1)})
        if gi + 1 < len(kept):  # 断在这组最后一跳之后、本该跳下一次的时候
            res["breaks"].append({"t": round(g[-1]["t"] + float(np.median(ivs)), 3)})
    res["total"] = len(res["jumps"])
    res["inferred"] = sum(1 for j in res["jumps"] if j.get("inferred"))
    res["best"] = max((g["count"] for g in res["groups"]), default=0)
    if debug is not None:
        debug.update({"h": h, "lift_l": lift_l, "lift_r": lift_r, "cand": cand, "tall": tall, "T": T})
    return res
