"""跳绳：视频 → 逐帧主要人物的关键点 → 计数（combocam/jumprope.py）→ 渲染用的轨迹 JSON。

轨迹 JSON 里 sport = "jumprope"，render/render.mjs 据此改用 render/jumprope_fx.js 画特效。和颠球的轨迹共用的字段
（width、height、fps、frames、source、scene、status）含义相同；跳绳自己的：
  jumps   每一跳 {t, frame, n（这组第几跳）, group, total（总第几跳）, up_t / land_t（离地、落地时刻）, passes（绳子过脚几次：1 单摇、2 双摇、3 三摇）,
          x（两脚中间）, y / y_up（落地、离地那一刻最低的脚底）}
  groups  每组连跳 {from, to, count, bpm}；breaks 断绳 {t, count（断之前这组跳了几个）, x, y}
  body    每帧 [脚的中点 x, 地面线 y, 人框左 x, 人框右 x, 两脚离地高度 ×1000（单位身高）, 髋部起伏 ×1000（单位身高）, 头顶 y]，
          没人时为 null
"""

from __future__ import annotations

import queue
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, percentile_filter, uniform_filter1d

from . import people
from .jumprope import L_FOOT, P, R_FOOT, _fill, _series, count_jumps, rope_passes
from .pose import HEAD, L_HIP, L_SHOULDER, L_WRIST, NOSE, R_HIP, R_SHOULDER, R_WRIST
from .scene import SceneStats
from .video import FrameReader, cache_complete, canvas_filter, content_bottom, frame_cache_path, probe, remove_cache

MIN_JUMPS = 3  # 少于这么多跳不出片
# 慢动作识别：腾空时身体自由落体，髋部轨迹是加速度为 g 的抛物线。按假定身高把像素换成米，
# 视频慢放 k 倍时量出来的 g 是真实的 1/k²。实测（2026-10-11，34 段 Pexels）：正常速度 8–11 m/s²（儿童偏大，身高按成人算了），
# 2 倍慢放约 2.5，4 倍约 0.6。视在 g 低于 SLOW_G 才当慢动作，按 √(9.8/g) 把跳速、用时换回真实时间。
ASSUMED_HEIGHT_M = 1.65
SLOW_G = 6.5


def main_person(d: dict) -> tuple[list, list, list]:
    """关键点缓存里的主要人物：有逐帧所有人（frames_people）就按 combocam/people.py 连轨迹、挑在跳绳的那个人；
    旧缓存只有一个人，直接用。返回 (逐帧人框, 逐帧关键点, 各轨迹摘要)。"""
    fp = d.get("frames_people")
    if not fp:
        return d["persons"], d["kpts"], []
    chosen, tracks = people.select(fp, d["fps"], d["height"])
    if chosen is None:
        n = len(fp)
        return [None] * n, [None] * n, []
    persons, kpts = people.series(fp, chosen, len(fp))
    return persons, kpts, tracks


def extract(src: str, det, pose, fps: float = 30.0, width: int = 1080, height: int = 1920, frame_cache: dict | None = None,
            pipeline: bool = True) -> dict:
    """逐帧所有人（远处的小人、超过 6 个的不要，见 people.keep_boxes）跑 26 点关键点、记衣服颜色，顺带统计画面亮度；
    跑完按 people.select 挑出在跳绳的主要人物（persons、kpts），逐帧所有人留在 frames_people 里（换选人规则不用重跑模型）。
    frame_cache 同 analyze.analyze：传入 dict 时够条件就把解码帧存进内存盘给渲染用，路径写进 frame_cache["path"]，调用方用完删掉。
    pipeline：同 analyze._pass1 的三级流水线（① 取帧、亮度、检测预处理 → ② 检测、挑人 → ③ 关键点，各占一个线程），
    算的东西和顺序都不变，结果和逐帧串行完全一样；False 时逐帧串行（核对用）。
    """
    import threading

    from .analyze import _END, _get, _put
    from .detector import PERSON
    from .pose import pack_kpts

    p = probe(src)
    vf = canvas_filter(p, width, height, fps)
    scene = SceneStats(fps)
    n_people: list[int] = []
    everyone: dict[int, list] = {}
    t0 = time.perf_counter()
    cache = frame_cache_path(p, fps, size=(width, height)) if frame_cache is not None else None
    try:
        reader = FrameReader(src, vf, width, height, cache=cache, yuv_canvas=True)
    except BaseException:
        remove_cache(cache)
        raise

    def step(dets) -> list:
        n_people.append(sum(1 for d in dets if d.score >= 0.5))
        boxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets]
        return [boxes[i] for i in people.keep_boxes(boxes)]

    def keypoints_of(f: int, frame, boxes: list) -> None:
        ks = pose(frame, boxes)
        everyone[f] = [{"box": [round(v, 1) for v in b], "kpts": pack_kpts(k), "hist": people.appearance(frame, b, k)}
                       for b, k in zip(boxes, ks)]

    if not pipeline:
        for f, frame in enumerate(reader):
            scene.add(frame)
            boxes = step(det.detect(frame, classes=(PERSON,), score_thr=0.3))
            if boxes:
                keypoints_of(f, frame, boxes)
    else:
        stop = threading.Event()
        errors: list[BaseException] = []
        q_prep: queue.Queue = queue.Queue(maxsize=4)
        q_pose: queue.Queue = queue.Queue(maxsize=8)

        def prep():
            try:
                for f, frame in enumerate(reader):
                    scene.add(frame)
                    if not _put(q_prep, (f, frame, det.prepare(frame)), stop):
                        return
            except BaseException as e:  # noqa: BLE001  交给主线程抛
                errors.append(e)
                stop.set()
            finally:
                _put(q_prep, _END, stop)

        def keypoints():
            try:
                while (item := _get(q_pose, stop)) is not _END:
                    keypoints_of(*item)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                stop.set()

        threads = [threading.Thread(target=prep, daemon=True), threading.Thread(target=keypoints, daemon=True)]
        for t in threads:
            t.start()
        try:
            while (item := _get(q_prep, stop)) is not _END:
                f, frame, prepared = item
                boxes = step(det.detect(None, classes=(PERSON,), score_thr=0.3, prepared=prepared))
                if boxes:
                    _put(q_pose, (f, frame, boxes), stop)
        except BaseException:
            stop.set()
            raise
        finally:
            _put(q_pose, _END, stop)
            if stop.is_set():
                reader.close()
            for t in threads:
                t.join()
        if errors:
            raise errors[0]
    n = len(n_people)
    if cache and not cache_complete(cache, p, n, (width, height)):
        remove_cache(cache)
        cache = None
    if frame_cache is not None:
        frame_cache["path"] = cache
    d = {
        "source": Path(src).name, "fps": fps, "width": width, "height": height, "frames": n,
        "content_bottom": content_bottom(p.width, p.height, width, height),
        "probe": {"width": p.width, "height": p.height, "fps": round(p.fps, 3), "duration": round(p.duration, 3), "has_audio": p.has_audio},
        "filter": vf, "canvas_yuv": True, "scene": scene.result(),
        "people": n_people, "frames_people": [everyone.get(f, []) for f in range(n)],
    }
    d["persons"], d["kpts"], d["tracks"] = main_person(d)
    d["extract_s"] = round(time.perf_counter() - t0, 1)
    return d


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    """中值去尖刺 + 滑动平均；NaN 段保持 NaN。"""
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return y
    filled = np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok])
    out = uniform_filter1d(median_filter(filled, size=k, mode="nearest"), size=k, mode="nearest")
    out[~ok] = np.nan
    return out


def _edges(air: np.ndarray, f: int, reach: int, fallback: int) -> tuple[int, int]:
    """波峰 f 前后的离地、落地帧：两脚离地高度（较低那只脚）降到"波峰前（后）最低值 + 30% 落差"以下的第一帧。
    门槛按局部最低值算，不按绝对值：手持拍摄时地面线估不准，脚着地时离地高度也可能不是 0。找不到就用 f ∓ fallback。"""
    n = len(air)

    def find(rng, seg):
        seg = seg[np.isfinite(seg)]
        if not len(seg) or not np.isfinite(pk):
            return None
        thr = seg.min() + 0.3 * (pk - seg.min())
        return next((g for g in rng if np.isfinite(air[g]) and air[g] < thr), None)

    win = air[max(0, f - 2):f + 3]
    pk = np.nanmax(win) if np.isfinite(win).any() else np.nan
    up = find(range(f, max(-1, f - reach - 1), -1), air[max(0, f - reach):f + 1])
    land = find(range(f, min(n, f + reach + 1)), air[f:min(n, f + reach + 1)])
    return (max(0, f - fallback) if up is None else up), (min(n - 1, f + fallback) if land is None else land)


def apparent_gravity(hip: np.ndarray, jumps: list[dict], fps: float, body_px: float | None) -> tuple[float | None, int]:
    """每跳腾空段（up_t–land_t）的髋部高度拟合抛物线，取加速度中位数，按假定身高换成 m/s²。返回 (g, 拟合了几跳)。"""
    if not body_px:
        return None, 0
    gs = []
    for j in jumps:
        a, b = round(j["up_t"] * fps), round(j["land_t"] * fps)
        y = hip[a:b + 1]
        if b - a < 3 or not np.isfinite(y).all():
            continue
        gs.append(2 * np.polyfit(np.arange(a, b + 1) / fps, y, 2)[0])
    if len(gs) < 4:
        return None, len(gs)
    return float(np.median(gs)) * ASSUMED_HEIGHT_M / body_px, len(gs)


def arm_sync(kpts: list, jumps: list[dict], groups: list[dict]) -> float | None:
    """摇绳节奏和起跳对不对得上：最长那组里，手腕（相对肩膀）上下位置在"一跳的间隔"上的自相关。
    绳子每过脚一次手腕转一圈，数对了应该接近 1；一圈数成两跳（或两圈数成一跳）时是负的。
    2026-10-11 在 34 段 Pexels 上：正面、侧面能看清手的多在 0.5–0.9；逆光剪影、多人、只有三五跳的接近 0。"""
    if not groups:
        return None
    g = max(groups, key=lambda g: g["count"])
    fs = [j["frame"] for j in jumps if g["from"] <= j["t"] <= g["to"]]
    if len(fs) < 5:
        return None
    lag = int(round(float(np.median(np.diff(fs)))))
    y = (_fill(_series(kpts, [L_WRIST, R_WRIST], "mean"), 3) - _fill(_series(kpts, [L_SHOULDER, R_SHOULDER], "mean"), 3))[fs[0]:fs[-1] + 1]
    y = y[np.isfinite(y)]
    if len(y) < 2 * lag + 2:
        return None
    y = y - median_filter(y, size=31, mode="nearest")
    y = (y - y.mean()) / (y.std() + 1e-9)
    return round(float(np.dot(y[:-lag], y[lag:]) / len(y)), 2)


def person_track(d: dict, persons: list, kpts: list, params: dict | None = None, speed: float | None = None) -> dict:
    """一个人的计数和渲染用数据：jumps、groups、breaks、absent、body、summary、warnings、status。
    speed：慢放倍数，给了就按它判双摇（双人 PK 时两人共用一个，见 build_track），不给按这个人自己量的视在重力估。"""
    fps, n, H = d["fps"], d["frames"], d["height"]
    dbg: dict = {}
    r = count_jumps(kpts, fps, params, debug=dbg, frame_h=H)
    p = {**P, **(params or {})}
    # 每帧的锚点：脚的中点、地面线（这只脚附近一段时间里的最低位置）、人框左右边
    feet_x = np.full(n, np.nan)
    for f, k in enumerate(kpts):
        if k:
            a = np.asarray(k, np.float32).reshape(-1, 3)
            ok = [i for i in L_FOOT + R_FOOT if a[i, 2] >= 40]
            if ok:
                feet_x[f] = float(a[ok, 0].mean())
    low = np.fmax(_series(kpts, L_FOOT, "max"), _series(kpts, R_FOOT, "max"))
    T = r.get("period_s") or 0.4
    win = max(3, int(round(max(p["detrend_s"], p["detrend_T"] * T) * fps)) | 1)
    ok = np.isfinite(low)
    ground = np.full(n, np.nan)
    if ok.sum() >= 2:
        ground = percentile_filter(np.interp(np.arange(n), np.flatnonzero(ok), low[ok]), 85, size=win, mode="nearest")
        ground[~ok] = np.nan
    ground = np.fmin(_smooth(ground, 5), d.get("content_bottom", H))
    k = max(3, int(round(0.15 * fps)) | 1)
    feet_x = _smooth(feet_x, k)
    box = np.array([b if b else [np.nan] * 4 for b in persons], np.float64).reshape(n, 4)
    bx1, bx2 = _smooth(box[:, 0], k), _smooth(box[:, 2], k)
    lift_l, lift_r = dbg.get("lift_l"), dbg.get("lift_r")
    air = np.fmin(lift_l, lift_r) if lift_l is not None else np.full(n, np.nan)
    h = dbg.get("h", np.full(n, np.nan))
    head = _smooth(_series(kpts, [HEAD, NOSE], "min"), k)

    def iv(v, s=1.0):
        return None if not np.isfinite(v) else int(round(float(v) * s))

    body = [None if not kpts[f] else [iv(feet_x[f]), iv(ground[f]), iv(bx1[f]), iv(bx2[f]), iv(air[f], 1000), iv(h[f], 1000), iv(head[f])]
            for f in range(n)]

    def ground_at(f: int) -> tuple[float | None, float | None]:
        for g in [f] + [f + s * o for o in range(1, 16) for s in (-1, 1)]:
            if 0 <= g < n and np.isfinite(feet_x[g]) and np.isfinite(ground[g]):
                return round(float(feet_x[g]), 1), round(float(ground[g]), 1)
        return None, None

    def sole_at(g: int) -> float | None:
        """g 前后一帧里最低的脚底（离地、落地那一刻脚就在地上，比估出来的地面线准）。"""
        seg = low[max(0, g - 1):g + 2]
        return round(float(np.nanmax(seg)), 1) if np.isfinite(seg).any() else None

    reach = max(2, int(round(0.5 * T * fps)))
    fallback = max(1, int(round(0.25 * T * fps)))
    jumps = []
    for i, j in enumerate(r["jumps"]):
        f = j["frame"]
        up, land = _edges(air, f, reach, fallback)
        x, y = ground_at(land)
        y_land, y_up = sole_at(land), sole_at(up)
        if j.get("inferred"):  # 人被挡住时按节拍补上的跳：没有位置，成片里不画脚下的效果
            x = y = y_land = y_up = None
        jumps.append({**j, "total": i + 1, "up_t": round(up / fps, 3), "land_t": round(land / fps, 3), "x": x,
                      "y": y_land if y_land is not None else y, "y_up": y_up if y_up is not None else y})
    # 每一跳绳子过脚几次（单摇 1、双摇 2、三摇 3），见 jumprope.rope_passes；腾空时间按慢放倍数换回真实时间，所以先估慢放
    hip0 = _fill(_series(kpts, [L_HIP, R_HIP], "mean"), 3)
    g0, _ = apparent_gravity(hip0, jumps, fps, r.get("body_px"))
    speed0 = float(np.clip(np.sqrt(9.8 / max(g0, 0.05)), 1, 10)) if g0 is not None and g0 < SLOW_G else 1.0
    if speed is not None:
        speed0 = speed
    cand_lift = {c["frame"]: max(c.get("lift_l") or 0, c.get("lift_r") or 0) for c in dbg.get("cand", [])}
    lifts = [cand_lift.get(j["frame"]) for j in jumps]
    for j, n_pass in zip(jumps, rope_passes(kpts, fps, jumps, speed0, r.get("feet_visible", True), dbg.get("tall"), lifts)):
        j["passes"] = n_pass
    # 两组之间人不在画面里（镜头摇走、被挡住）的时间超过一半：是离开了画面，不算断绳（两组照样分开）
    breaks, absent = [], []
    for b, g, g2 in zip(r["breaks"], r["groups"], r["groups"][1:]):
        f0, f1 = int(g["to"] * fps) + 1, int(g2["from"] * fps)
        if f1 > f0 and sum(1 for f in range(f0, f1) if kpts[f]) < 0.5 * (f1 - f0):
            absent.append({"from": g["to"], "to": g2["from"]})
            continue
        x, y = ground_at(min(n - 1, int(round(b["t"] * fps))))
        breaks.append({"t": b["t"], "count": g["count"], "x": x, "y": y})
    hip = _fill(_series(kpts, [L_HIP, R_HIP], "mean"), 3)
    g_app, n_fit = apparent_gravity(hip, jumps, fps, r.get("body_px"))
    speed = round(float(np.clip(np.sqrt(9.8 / max(g_app, 0.05)), 1, 10)), 2) if g_app is not None and g_app < SLOW_G else 1.0
    slowmo = speed > 1.0
    span = sum(g["to"] - g["from"] for g in r["groups"])
    reps = sum(g["count"] - 1 for g in r["groups"])
    warnings = [r["warning"]] if r.get("warning") else []
    sync = arm_sync(kpts, r["jumps"], r["groups"])
    n_multi = (sum(1 for j in jumps if j.get("passes") == 2), sum(1 for j in jumps if j.get("passes") == 3))
    if sync is not None and sync < 0.2:
        warnings.append("摇绳的手和起跳的节奏对不上（手被挡住、多人入镜或一圈跳两下），计数可能不准")
    if r.get("bounce") == "double":
        warnings.append("一圈绳子跳两下（绳子在头顶时垫一步），按绳子过脚的次数计数，垫的那一步不算")
    if slowmo:
        warnings.append(f"像是慢动作视频（约 {speed:.1f} 倍慢放），跳速、用时按真实时间换算（按成人身高估的，大约数）")
    total = r.get("total", 0)
    status = {"ok": True, "code": "ok", "title": "", "message": ""} if total >= MIN_JUMPS else {
        "ok": False, "code": "no_jumps", "title": "没数到跳绳",
        "message": "请把手机固定放好，拍到全身（脚别出画），正面或侧面都行，连续跳 5 下以上再试一次。"}
    return {
        "jumps": jumps, "groups": r["groups"], "breaks": breaks, "absent": absent, "body": body,
        # bpm、real_s 是换回真实时间的跳速、用时；speed 是慢放倍数（1 = 正常速度），渲染器据此换算
        "summary": {"total": total, "best": r.get("best", 0), "groups": len(r["groups"]), "breaks": len(breaks),
                    "period_s": r.get("period_s"), "bpm": round(60 * reps / span * speed, 1) if span > 0 else None,
                    "real_s": round(span / speed, 1), "feet_visible": r.get("feet_visible"), "body_px": r.get("body_px"),
                    "slowmo": slowmo, "speed": speed, "apparent_g": None if g_app is None else round(g_app, 2), "g_fits": n_fit, "arm_sync": sync,
                    # 双摇、三摇的跳数（绳子过脚 2、3 次的跳）和绳子过脚的总次数
                    "doubles": n_multi[0], "triples": n_multi[1], "passes": sum(j.get("passes", 1) for j in jumps), "bounce": r.get("bounce", "single"),
                    # double：一半以上是双摇（成片按双摇计数）；mixed：夹着几个双摇；single：全是单摇
                    "style": "double" if n_multi[0] + n_multi[1] >= max(3, 0.5 * len(jumps)) else "mixed" if n_multi[0] + n_multi[1] else "single"},
        "warnings": warnings, "status": status,
    }


PK_SIDES = ("left", "right")


def build_track(d: dict, src: str | None = None, params: dict | None = None, mode: str = "single", pk_rule: str = "auto") -> dict:
    """d：extract() 的结果（或 tools/eval/jumprope_extract.py 的缓存）。返回渲染用的轨迹 JSON。

    mode：single 单人（多人入镜时挑在跳绳的那个，见 people.select）；pk 双人 PK（挑两个在跳的人，按画面左右排）；
    auto 两个人明显在一起跳（见 people.select_pair）时出 PK，否则单人。
    PK 的轨迹：mode = "pk"，players 是左右两人各自的 person_track（加 side），summary 是双方总数、双摇数、比法、得分、谁赢、反超了几次。
    pk_rule：auto（两人都以双摇为主时比双摇个数，否则比总跳数）/ jumps / doubles / passes，见 pk_points。
    """
    fps, n, H = d["fps"], d["frames"], d["height"]
    pr, vf = d.get("probe") or {}, d.get("filter")
    if src and (not vf or "has_audio" not in pr):  # 旧的关键点缓存没存这些
        pp = probe(src)
        vf = vf or canvas_filter(pp, d["width"], H, fps)
        pr = {"width": pp.width, "height": pp.height, "fps": round(pp.fps, 3), "duration": round(pp.duration, 3), "has_audio": pp.has_audio}
    common = {
        "sport": "jumprope", "version": 1,
        "width": d["width"], "height": H, "fps": fps, "frames": n, "duration": round(n / fps, 3),
        # canvas_yuv：画布滤镜后先转 yuv420p（见 video.FrameReader），渲染照做，解码帧缓存是画布尺寸的
        "source": {"path": str(Path(src).resolve()) if src else d.get("source"), "probe": pr, "filter": vf,
                   "canvas_yuv": bool(d.get("canvas_yuv"))},
        "scene": d.get("scene"),
    }
    pair, tracks, pk_ok = None, [], False
    if mode in ("pk", "auto") and d.get("frames_people"):
        pair, tracks, pk_ok = people.select_pair(d["frames_people"], fps, H)
    if pair and (mode == "pk" or pk_ok):
        fp = d["frames_people"]
        series = [(side, *people.series(fp, t, n)) for side, t in zip(PK_SIDES, pair)]
        first = [person_track(d, persons, kpts, params) for _, persons, kpts in series]
        speed = shared_speed(first)
        # 双摇要按真实腾空时间判，两人共用一个慢放倍数重算一遍，同一段视频两人的双摇标准一样
        players = [{"side": side, **person_track(d, persons, kpts, params, speed=speed)} for side, persons, kpts in series]
        apply_speed(players, speed)
        totals = [pl["summary"]["total"] for pl in players]
        doubles = [pl["summary"]["doubles"] + pl["summary"]["triples"] for pl in players]
        rule = pk_rule if pk_rule != "auto" else ("doubles" if all(pl["summary"]["style"] == "double" for pl in players) else "jumps")
        scores = pk_scores(players, rule)
        warnings = [f"{'左' if pl['side'] == 'left' else '右'}边的人：{w}" for pl in players for w in pl["warnings"]]
        if speed > 1:
            warnings.append(f"像是慢动作视频（约 {speed:.1f} 倍慢放），双方的跳速、用时都按真实时间换算（大约数）")
        ok = min(totals) >= MIN_JUMPS
        status = {"ok": True, "code": "ok", "title": "", "message": ""} if ok else {
            "ok": False, "code": "no_pk", "title": "没数到两个人一起跳",
            "message": "双人 PK 要两个人同时在画面里跳：并排站好、都拍到全身，手机固定，一起跳 5 下以上再试一次。"}
        return {**common, "mode": "pk", "players": players, "people": tracks,
                # rule：怎么比——jumps 比总跳数，doubles 比双摇个数（两人都以双摇为主时自动选），passes 比绳子过脚的总次数（双摇算 2）
                "summary": {"totals": totals, "doubles": doubles, "rule": rule, "scores": scores,
                            "winner": None if scores[0] == scores[1] else int(scores[1] > scores[0]),
                            "lead_changes": lead_changes(players, rule)},
                "warnings": warnings, "status": status}
    if mode == "pk":
        notes = ["只找到一个人在跳绳，按单人出片"]
    else:
        notes = ["画面里有两个人在一起跳，可以出双人 PK 版（--mode pk）"] if pk_ok else []
    persons, kpts, tracks = main_person(d)
    one = person_track(d, persons, kpts, params)
    one["warnings"] = one["warnings"] + notes
    # people：画面里每个人（连成轨迹后）各数到几跳、选了谁：第一条是选中的
    return {**common, "mode": "single", **one, "people": tracks, "pk_candidate": pk_ok}


def shared_speed(players: list[dict]) -> float:
    """双人同在一段视频里，慢放倍数只能有一个：双方各自量的视在重力按拟合的跳数加权平均。
    （各算各的会不一样：6390396 左边的人量出 1.3 倍、右边的人 1 倍，同一段视频两人跳速就没法比。）"""
    gs = [(pl["summary"]["apparent_g"], pl["summary"]["g_fits"]) for pl in players if pl["summary"].get("apparent_g") is not None]
    if not gs:
        return 1.0
    g = sum(v * w for v, w in gs) / max(1, sum(w for _, w in gs))
    return round(float(np.clip(np.sqrt(9.8 / max(g, 0.05)), 1, 10)), 2) if g < SLOW_G else 1.0


def apply_speed(players: list[dict], speed: float) -> None:
    """按共用的慢放倍数统一换算两人的跳速、用时；各自的慢动作提示换成一条（见 build_track）。"""
    for pl in players:
        sm = pl["summary"]
        old = sm.get("speed") or 1.0
        if sm.get("bpm") is not None:
            sm["bpm"] = round(sm["bpm"] / old * speed, 1)
        sm["real_s"] = round(sm["real_s"] * old / speed, 1)
        sm["speed"], sm["slowmo"] = speed, speed > 1.0
        pl["warnings"] = [w for w in pl["warnings"] if not w.startswith("像是慢动作视频")]


LEAD_MARGIN = 2  # 领先 2 个才算领先：两人差不多同时跳时，谁先落地谁就"领先 1 个"，每跳都在换（同 render/jumprope_pk_fx.js）


def pk_points(j: dict, rule: str) -> int:
    """一跳在这种比法下得几分：jumps 每跳 1 分；doubles 只有双摇（三摇）得 1 分；passes 按绳子过脚的次数（双摇 2、三摇 3）。
    和 render/jumprope_pk_fx.js 的 points 一致。"""
    n = j.get("passes", 1)
    return n if rule == "passes" else int(n >= 2) if rule == "doubles" else 1


def pk_scores(players: list[dict], rule: str) -> list[int]:
    return [sum(pk_points(j, rule) for j in pl["jumps"]) for pl in players]


def lead_changes(players: list[dict], rule: str = "jumps") -> int:
    """按时间顺序合并双方得分的跳，领先（至少 LEAD_MARGIN 分）的一方换了几次。"""
    events = sorted([(j["t"], k, pk_points(j, rule)) for k, pl in enumerate(players) for j in pl["jumps"] if pk_points(j, rule)])
    cnt, lead, changes = [0, 0], None, 0
    for _, k, w in events:
        cnt[k] += w
        d = cnt[0] - cnt[1]
        now = 0 if d >= LEAD_MARGIN else 1 if -d >= LEAD_MARGIN else lead
        if now != lead and lead is not None:
            changes += 1
        lead = now
    return changes


def analyze_jumprope(src: str, model: str, pose_model: str, device: str = "auto", fps: float = 30.0,
                     frame_cache: dict | None = None, mode: str = "single", pk_rule: str = "auto") -> tuple[dict, dict]:
    """返回 (轨迹 JSON, 关键点缓存)。frame_cache 见 extract，mode 见 build_track。"""
    from .detector import YoloxDetector
    from .pose import RTMPose

    t0 = time.perf_counter()
    det = YoloxDetector(model, device)
    pose = RTMPose(pose_model, device)
    t1 = time.perf_counter()
    d = extract(src, det, pose, fps, frame_cache=frame_cache)
    t2 = time.perf_counter()
    tr = build_track(d, src, mode=mode, pk_rule=pk_rule)
    tr["timing"] = {"load_models_s": round(t1 - t0, 2), "extract_s": round(t2 - t1, 2), "count_s": round(time.perf_counter() - t2, 3)}
    tr["meta"] = {"model": Path(model).stem, "pose_model": Path(pose_model).stem, "device": det.device}
    return tr, d
