"""颠球分析：视频 → 轨迹数据 JSON（格式见 docs/track-format.md）。运球也用这里的检测和选轨迹。"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

from . import __version__
from .detector import PERSON, SPORTS_BALL, YoloxDetector
from .events import analyze_track
from .scene import SceneStats
from .track import Cand, mark_static, ref_radius, select_track
from .verdict import verdict
from .pose import RTMPose, pack_kpts
from .video import FrameReader, cache_complete, canvas_filter, content_bottom, frame_cache_path, probe, remove_cache

CROP = 640


def _best_person(dets) -> tuple | None:
    ps = [d for d in dets if d.cls == PERSON and d.score >= 0.3]
    if not ps:
        return None
    d = max(ps, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
    return (d.x1, d.y1, d.x2, d.y2)


def zoom_window(person: tuple, width: int, top: float, bottom: float) -> tuple[int, int, int, int]:
    """放大复检的窗口：以人为中心的正方形，下边到脚下一点、往上盖过头顶，边长至少 CROP（再小就不是放大了）。
    只在真实画面里（横屏放进竖版时上下是模糊垫底）。"""
    x1, y1, x2, y2 = person
    h = y2 - y1
    side = min(max(float(CROP), ZOOM["size"] * h, (x2 - x1) * 1.6), float(width), bottom - top)
    cx = (x1 + x2) / 2
    X0 = float(np.clip(cx - side / 2, 0, width - side))
    Y0 = float(np.clip(y2 + ZOOM["below"] * h - side, top, bottom - side))
    return int(X0), int(Y0), int(X0 + side), int(Y0 + side)


# 窗口边长 = 人高 × size，下边在脚下 below × 人高；skip：整帧检测在窗口里已经有分数 ≥ skip 的球就不复检（省时间，None = 每帧都复检）
# r_lo：放大后检出的球半径小于 r_lo × 0.065 × 人高（球的直径约人高的 13%）的不要——放大后远处的小东西（地上的小球、
# 灯、鞋上的反光）也会检出来，混进去会把参考半径拉小（5586534 运球：74 个半径 8 的候选，球本来半径 137）
ZOOM = {"size": 1.3, "below": 0.12, "skip": 0.5, "r_lo": 0.25}


def _merge(balls: list[Cand], dets, src: int) -> None:
    """复检出来的球并进这一帧的候选：和已有的重复（中心距离 < 半个半径）时留分数高的。"""
    for d in dets:
        dup = [c for c in balls if np.hypot(d.cx - c.x, d.cy - c.y) <= 0.5 * max(c.r, 1)]
        if not dup:
            balls.append(Cand(d.cx, d.cy, d.r, d.score, src))
        elif d.score > dup[0].s:
            c = dup[0]
            c.x, c.y, c.r, c.s, c.src = d.cx, d.cy, d.r, d.score, src


def _smooth_persons(persons: list[tuple | None], win: int = 9) -> list[tuple | None]:
    arr = np.array([p if p is not None else (np.nan,) * 4 for p in persons], dtype=np.float64)
    out: list[tuple | None] = []
    for i in range(len(arr)):
        if np.isnan(arr[i, 0]):
            out.append(None)
            continue
        w = arr[max(0, i - win // 2): i + win // 2 + 1]
        out.append(tuple(np.nanmedian(w, axis=0).tolist()))
    return out


def _predict_positions(sel_pts: dict[int, Cand], n: int, fps: float, max_gap: int) -> dict[int, tuple[float, float]]:
    """需要复检的帧：轨迹段内部缺检的帧，以及段两头各外推 max_gap 帧。"""
    if not sel_pts:
        return {}
    frames = sorted(sel_pts)
    xs = np.array([sel_pts[f].x for f in frames])
    ys = np.array([sel_pts[f].y for f in frames])
    want: dict[int, tuple[float, float]] = {}
    for a, b in zip(frames, frames[1:]):
        if 1 < b - a <= 3 * max_gap:
            for f in range(a + 1, b):
                want[f] = (float(np.interp(f, frames, xs)), float(np.interp(f, frames, ys)))
    # 段两头：用最近几帧的速度外推
    segs, cur = [], [frames[0]]
    for f in frames[1:]:
        if f - cur[-1] > 3 * max_gap:
            segs.append(cur)
            cur = [f]
        else:
            cur.append(f)
    segs.append(cur)
    for seg in segs:
        for end, step in ((seg[-1], 1), (seg[0], -1)):
            nb = [f for f in seg if abs(f - end) <= 3]
            if len(nb) < 2:
                continue
            p0, p1 = sel_pts[min(nb) if step > 0 else max(nb)], sel_pts[end]
            df = abs(end - (min(nb) if step > 0 else max(nb))) or 1
            vx, vy = (p1.x - p0.x) / df, (p1.y - p0.y) / df
            for k in range(1, max_gap + 1):
                f = end + step * k
                if 0 <= f < n and f not in sel_pts:
                    want.setdefault(f, (p1.x + vx * k, p1.y + vy * k))
    return want


def postprocess(cands: list[list[Cand]], persons_raw: list[tuple | None], fps: float, height: int, sel=None,
                content_bottom: float | None = None, kpts: list | None = None):
    """候选框 → 轨迹 → 事件。sel 为空时重新选轨迹（评估脚本可以只重跑这一步）。"""
    if sel is None:
        r_ref = ref_radius(cands)
        sel = select_track(cands, fps, r_ref, mark_static(cands, fps, r_ref))
    track = {f: cands[f][j] for f, j in enumerate(sel) if j is not None}
    persons = _smooth_persons(persons_raw)
    ev = analyze_track(track, fps, len(cands), persons, height, content_bottom=content_bottom, kpts=kpts)
    return track, persons, ev


class OnlineRefiner:
    """第 1 遍里边检测边跟踪：球漏检或分数低时，马上在预测位置裁 640×640 原图复检，省掉第二次解码。"""

    def __init__(self, det: YoloxDetector, width: int, height: int, fps: float, max_gap: int):
        self.det, self.w, self.h, self.max_gap = det, width, height, max_gap
        self.last = None  # (帧号, x, y, vx, vy)
        self.rs: list[float] = []
        self.t_infer = 0.0
        self.n = 0

    def _r(self) -> float:
        return float(np.median(self.rs[-30:])) if self.rs else 40.0

    def step(self, f: int, frame: np.ndarray, balls: list[Cand]) -> None:
        pred = None
        if self.last and f - self.last[0] <= self.max_gap:
            df = f - self.last[0]
            pred = (self.last[1] + self.last[3] * df, self.last[2] + self.last[4] * df)
        gate = 3.0 * self._r() + (abs(self.last[3]) + abs(self.last[4])) * 2 if pred else 0
        near = [c for c in balls if pred and np.hypot(c.x - pred[0], c.y - pred[1]) < gate]
        best = max(near, key=lambda c: c.s) if near else None
        if pred and (best is None or best.s < 0.3):
            x0 = int(min(max(pred[0] - CROP / 2, 0), self.w - CROP))
            y0 = int(min(max(pred[1] - CROP / 2, 0), self.h - CROP))
            a = time.perf_counter()
            dets = self.det.detect(frame[y0:y0 + CROP, x0:x0 + CROP], classes=(SPORTS_BALL,), score_thr=0.05, offset=(x0, y0))
            self.t_infer += time.perf_counter() - a
            self.n += 1
            for d in dets:
                dup = [c for c in balls if np.hypot(d.cx - c.x, d.cy - c.y) <= 0.5 * max(c.r, 1)]
                if not dup:
                    balls.append(Cand(d.cx, d.cy, d.r, d.score, 1))
                elif d.score > dup[0].s:
                    c = dup[0]
                    c.x, c.y, c.r, c.s, c.src = d.cx, d.cy, d.r, d.score, 1
            near = [c for c in balls if np.hypot(c.x - pred[0], c.y - pred[1]) < gate]
            best = max(near, key=lambda c: c.s) if near else None
        if best is None and not pred:
            conf = [c for c in balls if c.s >= 0.4]
            best = max(conf, key=lambda c: c.s) if conf else None
            if best:
                self.last = (f, best.x, best.y, 0.0, 0.0)
                self.rs.append(best.r)
            return
        if best is not None:
            df = f - self.last[0]
            vx, vy = (best.x - self.last[1]) / df, (best.y - self.last[2]) / df
            self.last = (f, best.x, best.y, 0.5 * vx + 0.5 * self.last[3], 0.5 * vy + 0.5 * self.last[4])
            if best.s >= 0.3:
                self.rs.append(best.r)


_END = object()


def _put(q: queue.Queue, item, stop: threading.Event) -> bool:
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            pass
    return False


def _get(q: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        try:
            return q.get(timeout=0.1)
        except queue.Empty:
            pass
    return _END


def _pass1(reader: FrameReader, det: YoloxDetector, pose: RTMPose | None, online: OnlineRefiner | None, scene: SceneStats,
           band: tuple[float, float] | None = None, zoom: tuple[float, float] | None = None):
    """第 1 遍：整帧检测（人 + 球）+ 在线复检 + 关键点。

    三级流水线，各占一个线程：① 取解码好的帧、统计亮度、检测预处理 → ② 检测推理、在线复检、挑人 → ③ 关键点。
    推理时 GPU 在算、CPU 空着，这样 CPU 的活和 GPU 的活叠在一起做。每一步算的东西和先后顺序都没变
    （亮度统计、在线复检仍按帧号顺序），结果和逐帧串行跑完全一样。
    返回 (cands, persons_raw, kpts, 检测累计秒数, 关键点累计秒数)；累计秒数是各线程各自的用时，加起来会超过总用时。
    """
    stop = threading.Event()
    errors: list[BaseException] = []
    q_prep: queue.Queue = queue.Queue(maxsize=4)
    q_pose: queue.Queue = queue.Queue(maxsize=8)
    kp_by_frame: dict[int, list] = {}
    t_prep, t_pose, t_zoom = [0.0], [0.0], [0.0]

    def prep():
        try:
            for f, frame in enumerate(reader):
                scene.add(frame)
                a = time.perf_counter()
                p = det.prepare(frame)
                t_prep[0] += time.perf_counter() - a
                if not _put(q_prep, (f, frame, p), stop):
                    return
        except BaseException as e:  # noqa: BLE001  交给主线程抛
            errors.append(e)
            stop.set()
        finally:
            _put(q_prep, _END, stop)

    def keypoints():
        try:
            while (item := _get(q_pose, stop)) is not _END:
                f, frame, person = item
                a = time.perf_counter()
                kp_by_frame[f] = pack_kpts(pose(frame, [person])[0])
                t_pose[0] += time.perf_counter() - a
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            stop.set()

    threads = [threading.Thread(target=prep, daemon=True)]
    if pose:
        threads.append(threading.Thread(target=keypoints, daemon=True))
    for t in threads:
        t.start()
    cands: list[list[Cand]] = []
    persons_raw: list[tuple | None] = []
    t_det = 0.0
    try:
        while (item := _get(q_prep, stop)) is not _END:
            f, frame, prepared = item
            a = time.perf_counter()
            dets = det.detect(None, classes=(PERSON, SPORTS_BALL), score_thr=0.05, prepared=prepared)
            t_det += time.perf_counter() - a
            if band is not None:  # 只要真实画面里的（横屏放进竖版时上下是同一帧放大模糊垫底，里面也会检出人和球）
                dets = [d for d in dets if band[0] <= (d.y1 + d.y2) / 2 <= band[1]]
            balls = [Cand(d.cx, d.cy, d.r, d.score, 0) for d in dets if d.cls == SPORTS_BALL]
            person = _best_person(dets)
            if zoom is not None and person is not None:
                # 放大复检：整帧缩到 640 时球常常只剩 10–20 像素（竖版画布缩小 3 倍），暗处、逆光剪影里分数只有 0.1 上下。
                # 在人周围裁一块（边长约 1.3 倍人高）单独检测，球大一两倍（见 docs/juggling.md 的"放大复检"）
                X0, Y0, X1, Y1 = zoom_window(person, frame.shape[1], zoom[0], zoom[1])
                sk = ZOOM["skip"]
                if sk is None or not any(c.s >= sk and X0 <= c.x <= X1 and Y0 <= c.y <= Y1 for c in balls):
                    a = time.perf_counter()
                    zd = det.detect(frame[Y0:Y1, X0:X1], classes=(SPORTS_BALL,), score_thr=0.05, offset=(X0, Y0))
                    t_zoom[0] += time.perf_counter() - a
                    r_min = ZOOM["r_lo"] * 0.065 * (person[3] - person[1])
                    _merge(balls, [d for d in zd if d.r >= r_min], 2)
            if online:
                online.step(f, frame, balls)
            cands.append(balls)
            persons_raw.append(person)
            if pose and person:
                _put(q_pose, (f, frame, person), stop)
    except BaseException:
        stop.set()
        raise
    finally:
        if pose:
            _put(q_pose, _END, stop)
        if stop.is_set():
            reader.close()
        for t in threads:
            t.join()
    if errors:
        raise errors[0]
    kpts = [kp_by_frame.get(f) for f in range(len(cands))] if pose else None
    return cands, persons_raw, kpts, t_det + t_prep[0] + t_zoom[0], t_pose[0]


def analyze(
    src: str,
    model: str = "models/yolox_s.onnx",
    device: str = "auto",
    refine: str = "online",
    width: int = 1080,
    height: int = 1920,
    fps: float = 30.0,
    log=print,
    debug: dict | None = None,
    pose_model: str | None = "models/rtmpose-m_halpe26_256x192.onnx",
    frame_cache: dict | None = None,
    content_only: bool = False,
    zoom: bool | None = None,
) -> dict:
    """debug 传入一个 dict 时，会把每帧候选框、选中的下标、人框放进去，便于评估。
    content_only：只用真实画面里的检测（横屏放进竖版时，上下的模糊垫底里也会检出放大的人和球，最大的人会选到那里去）。
    颠球默认不开（打开会改变现有结果），篮球运球（combocam/dribble_track.py）打开。
    frame_cache 传入一个 dict 时，够条件就把解码帧存一份给渲染用（见 video.frame_cache_path），
    写进 frame_cache["path"]（不完整或没存时为 None）；用完由调用方删除（video.remove_cache）。"""
    t_start = time.perf_counter()
    p = probe(src)
    vf = canvas_filter(p, width, height, fps)
    det = YoloxDetector(model, device)
    # 人体关键点（判断落地、手接球用）；模型文件不在就跳过，计数退回只看轨迹和人框的旧规则
    pose = RTMPose(pose_model, device) if pose_model and Path(pose_model).exists() else None
    t_model = time.perf_counter()
    log(f"[analyze] {Path(src).name}: {p.width}x{p.height} {p.fps:.2f}fps {p.duration:.2f}s → 画布 {width}x{height}@{fps:g}，模型 {Path(model).stem} on {det.device}")

    # ---- 第 1 遍：整帧检测（人 + 球）；refine=online 时顺带做局部复检 ----
    max_gap = max(3, int(round(0.4 * fps)))
    online = OnlineRefiner(det, width, height, fps, max_gap) if refine == "online" else None
    scene = SceneStats(fps)
    t1 = time.perf_counter()
    cache = frame_cache_path(p, fps) if frame_cache is not None else None
    try:
        cb = content_bottom(p.width, p.height, width, height)
        band = (height - cb, cb) if content_only else None
        if zoom is None:
            zoom = os.environ.get("COMBOCAM_ZOOM", "1") != "0"
        zb = (height - cb, cb) if zoom else None
        cands, persons_raw, kpts, t_det, t_pose = _pass1(FrameReader(src, vf, width, height, cache=cache), det, pose, online, scene, band, zb)
    except BaseException:
        remove_cache(cache)
        raise
    n = len(cands)
    if cache and not cache_complete(cache, p, n):
        log("[analyze] 解码帧缓存没写全（内存盘满了？），渲染改为自己解码")
        remove_cache(cache)
        cache = None
    if frame_cache is not None:
        frame_cache["path"] = cache
    pass1_s = time.perf_counter() - t1
    extra = f"，在线复检 {online.n} 帧（推理 {online.t_infer:.1f}s）" if online else ""
    extra += f"，关键点 {t_pose:.1f}s" if pose else "，没有关键点模型"
    log(f"[analyze] 第 1 遍 {n} 帧，用时 {pass1_s:.1f}s（整帧推理 {t_det:.1f}s{extra}）")

    r_ref = ref_radius(cands)
    static = mark_static(cands, fps, r_ref)
    sel = select_track(cands, fps, r_ref, static)

    # ---- refine=offline：第 2 遍再解码一次，在轨迹预测位置附近原分辨率裁 640×640 复检 ----
    pass2_s, t_det2, n_refined = 0.0, 0.0, 0
    if online:
        t_det2, n_refined = online.t_infer, online.n
    if refine == "offline":
        sel_pts = {f: cands[f][j] for f, j in enumerate(sel) if j is not None}
        want = _predict_positions(sel_pts, n, fps, max_gap)
        # 置信度低的已选帧也复检一次
        for f, c in sel_pts.items():
            if c.s < 0.3:
                want.setdefault(f, (c.x, c.y))
        if want:
            t2 = time.perf_counter()
            for f, frame in enumerate(FrameReader(src, vf, width, height)):
                if f not in want:
                    continue
                px, py = want[f]
                x0 = int(min(max(px - CROP / 2, 0), width - CROP))
                y0 = int(min(max(py - CROP / 2, 0), height - CROP))
                a = time.perf_counter()
                dets = det.detect(frame[y0:y0 + CROP, x0:x0 + CROP], classes=(SPORTS_BALL,), score_thr=0.05, offset=(x0, y0))
                t_det2 += time.perf_counter() - a
                n_refined += 1
                for d in dets:
                    # 和第 1 遍重复的框不要重复加
                    if all(np.hypot(d.cx - c.x, d.cy - c.y) > 0.5 * max(c.r, 1) for c in cands[f]):
                        cands[f].append(Cand(d.cx, d.cy, d.r, d.score, 1))
                    else:
                        for c in cands[f]:
                            if np.hypot(d.cx - c.x, d.cy - c.y) <= 0.5 * max(c.r, 1) and d.score > c.s:
                                c.x, c.y, c.r, c.s, c.src = d.cx, d.cy, d.r, d.score, 1
            pass2_s = time.perf_counter() - t2
            log(f"[analyze] 第 2 遍复检 {n_refined} 帧，用时 {pass2_s:.1f}s（其中推理 {t_det2:.1f}s）")
            r_ref = ref_radius(cands)
            static = mark_static(cands, fps, r_ref)
            sel = select_track(cands, fps, r_ref, static)

    t3 = time.perf_counter()
    track, persons, ev = postprocess(cands, persons_raw, fps, height, sel, content_bottom=content_bottom(p.width, p.height, width, height),
                                     kpts=kpts)
    if debug is not None:
        debug["cands"] = [[(round(c.x, 1), round(c.y, 1), round(c.r, 1), round(c.s, 3), c.src) for c in fr] for fr in cands]
        debug["sel"] = sel
        debug["static"] = [s.tolist() for s in static]
        debug["persons"] = persons_raw
        if kpts is not None:
            debug["kpts"] = kpts
    post_s = time.perf_counter() - t3
    total_s = time.perf_counter() - t_start

    n_ball_det = sum(1 for fr in cands if fr)
    best_combo = max((c["count"] for c in ev.combos), default=0)
    status = verdict(n, len(track), sum(1 for p in persons_raw if p), best_combo)
    out = {
        "version": 1,
        "status": status,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": n,
        "duration": round(n / fps, 3),
        "ball": ev.ball,
        "touches": ev.touches,
        "apex": ev.apex,
        "combos": ev.combos,
        "highlight": ev.highlight,
        "bounces": ev.bounces,
        "summary": {
            "touches": len(ev.touches),
            "best_combo": best_combo,
            "best_height_m": max((a["height_m"] for a in ev.apex), default=None),
            "ball_detected_frames": n_ball_det,
            "track_frames": len(track),
        },
        "calib": ev.calib,
        "scene": scene.result(),
        "warnings": ev.warnings,
        "source": {
            "path": str(Path(src).resolve()),
            "probe": {"width": p.width, "height": p.height, "fps": round(p.fps, 3), "duration": round(p.duration, 3), "has_audio": p.has_audio},
            "filter": vf,
        },
        "meta": {
            "analyzer": f"combocam {__version__}",
            "model": Path(model).stem,
            "device": det.device,
            "refine": refine,
            "pose": Path(pose_model).stem if pose else None,
        },
        "timing": {
            "model_load_s": round(t_model - t_start, 2),
            "pass1_s": round(pass1_s, 2),
            "detect_infer_s": round(t_det, 2),
            "refine_infer_s": round(t_det2, 2),
            "refine_frames": n_refined,
            "pose_infer_s": round(t_pose, 2),
            "pass2_s": round(pass2_s, 2),
            "post_s": round(post_s, 2),
            "total_s": round(total_s, 2),
        },
    }
    log(f"[analyze] 触球 {len(ev.touches)} 次，最高连击 {best_combo}，落地反弹 {len(ev.bounces)} 次；总用时 {total_s:.1f}s")
    if not status["ok"]:
        log(f"[analyze] 不出片（{status['code']}）：{status['title']}。{status['message']}")
    for w in ev.warnings:
        log(f"[analyze] 注意：{w}")
    return out
