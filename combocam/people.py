"""多人入镜：逐帧所有人 → 跨帧连成轨迹 → 挑出在跳绳的那个人。

- 每帧：检测到的人里，高度不到最高那人 P["min_rel_h"] 的不要（远处路人），最多留 P["max_people"] 个，都跑关键点，
  再记一份衣服颜色（躯干区域的色调-饱和度直方图）。
- 连轨迹：相邻帧按人框重叠度贪心配对，衣服颜色还要像（直方图交 ≥ P["link_sim"]）——镜头摇到另一个人身上时，
  新的人常落在同一个位置，只看重叠度会接成同一个人；一时没配上的轨迹保留 P["max_gap_s"] 秒，回来时按中心距离、大小接上。
- 合并：镜头摇走又摇回来、被别人挡住时会断成几段。时间上不重叠、衣服颜色像（直方图交 ≥ P["merge_sim"]）、
  大小差不多的几段接成一条（按相似度从高到低贪心，接上后不能和已有的段重叠）。
- 选人：每条轨迹单独跑跳绳计数（combocam/jumprope.count_jumps）。跳数够格（≥ 3 且 ≥ 最多那条的 P["jump_gate"]）的
  算"在跳绳"，其中挑离镜头近、在画面里时间长的：分数 = 平均身高 × 出现时长占比。没人够格就按同样的分数挑。
  跳数只当门槛、不直接比大小：后面的人被前面的人挡着时脚点少，常常数少（4761426 前面穿白衣服的人数到 4、后面的人 8），
  拍视频的人想拍的是离镜头最近、在跳的那个。7207259 离镜头最近的是背对镜头站着的人（原来只按"最大的人"选，数到 0）。
"""

from __future__ import annotations

import cv2
import numpy as np

from .jumprope import count_jumps

P = {"min_rel_h": 0.3, "max_people": 6, "max_gap_s": 0.5, "iou": 0.2, "link_sim": 0.3, "keep_iou": 0.6, "cut_iou": 0.85, "suspect_n": 3,
     "merge_sim": 0.7, "min_track_s": 1.0,
     "jump_gate": 0.4, "min_jumps": 3, "pk_overlap": 0.5, "pk_jumps": 0.5, "pk_size": 0.5}
H_BINS, S_BINS = 8, 4


TORSO = (5, 6, 11, 12)  # 左右肩、左右髋（Halpe26 编号）


def appearance(frame: np.ndarray, box, kpts: np.ndarray | None = None) -> list[int]:
    """躯干的色调 × 饱和度直方图，归一化后 ×1000 取整。有关键点时取两肩、两髋围成的框（往里收 15%），
    不容易混进挡在前面、站在旁边的人；没有时取人框宽的中间 50%、高的 20%–60%。"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = frame.shape[:2]
    a, b = x1 + 0.25 * w, x2 - 0.25 * w
    c, d = y1 + 0.2 * h, y1 + 0.6 * h
    if kpts is not None:
        k = np.asarray(kpts, np.float32).reshape(-1, 3)[list(TORSO)]
        if (k[:, 2] >= 0.4).all():
            tx0, tx1, ty0, ty1 = k[:, 0].min(), k[:, 0].max(), k[:, 1].min(), k[:, 1].max()
            if tx1 - tx0 > 8 and ty1 - ty0 > 8:
                mx, my = 0.15 * (tx1 - tx0), 0.15 * (ty1 - ty0)
                a, b, c, d = tx0 + mx, tx1 - mx, ty0 + my, ty1 - my
    a, b, c, d = int(max(0, a)), int(min(W, b)), int(max(0, c)), int(min(H, d))
    if b - a < 4 or d - c < 4:
        return [0] * (H_BINS * S_BINS)
    crop = frame[c:d:2, a:b:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256]).ravel()
    return [int(round(v)) for v in 1000 * hist / max(1.0, hist.sum())]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def keep_boxes(boxes: list, params: dict | None = None) -> list[int]:
    """这一帧要跑关键点的人：按面积从大到小，高度不到最高那人 min_rel_h 的不要，最多 max_people 个。返回下标。"""
    p = {**P, **(params or {})}
    if not boxes:
        return []
    hmax = max(b[3] - b[1] for b in boxes)
    idx = [i for i in sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
           if boxes[i][3] - boxes[i][1] >= p["min_rel_h"] * hmax]
    return idx[:p["max_people"]]


def _sim(a, b) -> float:
    """两个直方图（×1000 的整数）的交，0–1；没有颜色数据时当作相同。"""
    if a is None or b is None or not np.any(a) or not np.any(b):
        return 1.0
    return float(np.minimum(np.asarray(a, np.float64), np.asarray(b, np.float64)).sum()) / 1000


def link(frames: list[list[dict]], fps: float, params: dict | None = None) -> list[dict]:
    """frames[f] = [{"box", "kpts", "hist"}, ...]。返回轨迹 [{"frames": {f: 下标}}]（按出现先后）。

    衣服颜色对不上（< link_sim）但人框重叠很多（≥ keep_iou）时照样接上：颜色量错的情况不少——单帧关键点出错让躯干取错地方，
    肩膀出画时躯干时而按关键点、时而按人框取（6326809 从第 215 帧起一直只有 0.2，人框重叠 0.97，是同一个人连续的镜头）。
    人框也明显变了（重叠 < cut_iou）才记为可疑，连续 P["suspect_n"] 帧可疑才是换了人（剪辑切到另一个镜头、
    镜头摇到另一个人身上），从第一帧可疑处拆成新轨迹。
    """
    p = {**P, **(params or {})}
    max_gap = max(1, int(round(p["max_gap_s"] * fps)))
    tracks: list[dict] = []

    def take(t: dict, f: int, q: dict, qi: int) -> None:
        moved = _iou(t["box"], q["box"]) < p["cut_iou"] if t.get("box") is not None else False
        t["frames"][f] = qi
        t["box"], t["last"] = q["box"], f
        h = np.asarray(q["hist"], np.float64) if q.get("hist") else None
        if h is not None and t.get("hist") is not None and _sim(t["hist"], h) < p["link_sim"] and (moved or t.get("suspect")):
            t.setdefault("suspect", []).append((f, qi, h))
            return
        t["suspect"] = []
        if h is not None:  # 衣服颜色按最近几帧慢慢更新（光线会变）
            t["hist"] = h if t.get("hist") is None else 0.8 * t["hist"] + 0.2 * h

    def split(t: dict) -> dict:
        sus = t.pop("suspect")
        for f, _, _ in sus:
            del t["frames"][f]
        t["suspect"] = []
        t["last"] = max(t["frames"]) if t["frames"] else sus[0][0] - 1
        nt = {"frames": {f: qi for f, qi, _ in sus}, "box": t["box"], "last": sus[-1][0],
              "hist": np.mean([h for _, _, h in sus], axis=0), "suspect": []}
        t["box"] = frames[t["last"]][t["frames"][t["last"]]]["box"] if t["frames"] else t["box"]
        return nt

    for f, people in enumerate(frames):
        live = [t for t in tracks if f - t["last"] <= max_gap]
        # 先配颜色对得上的，再按重叠度；颜色对不上的只在重叠很多时才配（见上）
        pairs = []
        for ti, t in enumerate(live):
            for qi, q in enumerate(people):
                iou = _iou(t["box"], q["box"])
                same = _sim(t.get("hist"), q.get("hist")) >= p["link_sim"]
                if iou >= p["iou"] and (same or iou >= p["keep_iou"]):
                    pairs.append((same, iou, ti, qi))
        pairs.sort(reverse=True)
        used_t, used_q = set(), set()
        for _, iou, ti, qi in pairs:
            if ti in used_t or qi in used_q:
                continue
            used_t.add(ti)
            used_q.add(qi)
            take(live[ti], f, people[qi], qi)
        # 没配上的：离开不久的轨迹按中心距离（不到身高一半）、大小（0.7–1.4 倍）、颜色接上，否则新开一条
        for qi, q in enumerate(people):
            if qi in used_q:
                continue
            b = q["box"]
            bh = b[3] - b[1]
            best, bd = None, 0.5
            for ti, t in enumerate(live):
                if ti in used_t or _sim(t.get("hist"), q.get("hist")) < p["link_sim"]:
                    continue
                tb = t["box"]
                th = tb[3] - tb[1]
                if not 0.7 <= bh / max(th, 1) <= 1.4:
                    continue
                dist = np.hypot((b[0] + b[2] - tb[0] - tb[2]) / 2, (b[1] + b[3] - tb[1] - tb[3]) / 2) / max(th, 1)
                if dist < bd:
                    best, bd = ti, dist
            if best is not None:
                used_t.add(best)
                take(live[best], f, q, qi)
            else:
                t = {"frames": {}, "suspect": [], "box": None}
                take(t, f, q, qi)
                tracks.append(t)
        for t in list(tracks):
            if len(t.get("suspect", [])) >= p["suspect_n"]:
                tracks.append(split(t))
    return [{"frames": t["frames"]} for t in tracks if t["frames"]]


def _hist(frames, t) -> np.ndarray:
    hs = [frames[f][i]["hist"] for f, i in t["frames"].items() if frames[f][i].get("hist")]
    return np.median(np.asarray(hs, np.float64), axis=0) if hs else np.zeros(H_BINS * S_BINS)


def _height(frames, t) -> float:
    return float(np.median([frames[f][i]["box"][3] - frames[f][i]["box"][1] for f, i in t["frames"].items()]))


def merge(frames: list[list[dict]], tracks: list[dict], params: dict | None = None) -> list[dict]:
    """时间上不重叠、衣服颜色像、大小差不多的几段轨迹接成一条（镜头摇走又摇回来、被挡住）。"""
    p = {**P, **(params or {})}
    groups = [{"frames": dict(t["frames"]), "segs": [(min(t["frames"]), max(t["frames"]))], "hist": _hist(frames, t),
               "h": _height(frames, t)} for t in tracks]
    while True:
        best = None
        for i, a in enumerate(groups):
            for j, b in enumerate(groups):
                # 时间上要分开：两边的每一段（起止帧区间）都不能重叠
                if j <= i or any(s0 <= e1 and s1 <= e0 for s0, e0 in a["segs"] for s1, e1 in b["segs"]):
                    continue
                if not 0.6 <= a["h"] / max(b["h"], 1) <= 1.6:
                    continue
                sim = float(np.minimum(a["hist"], b["hist"]).sum()) / 1000
                if sim >= p["merge_sim"] and (best is None or sim > best[0]):
                    best = (sim, i, j)
        if best is None:
            return [{"frames": g["frames"]} for g in groups]
        _, i, j = best
        a, b = groups[i], groups[j]
        na, nb = len(a["frames"]), len(b["frames"])
        groups[i] = {"frames": {**a["frames"], **b["frames"]}, "segs": a["segs"] + b["segs"],
                     "hist": (a["hist"] * na + b["hist"] * nb) / (na + nb),
                     "h": (a["h"] * na + b["h"] * nb) / (na + nb)}
        del groups[j]


def series(frames: list[list[dict]], t: dict, n: int) -> tuple[list, list]:
    """一条轨迹逐帧的 (人框, 关键点)，没有的帧为 None。"""
    persons, kpts = [None] * n, [None] * n
    for f, i in t["frames"].items():
        persons[f] = frames[f][i]["box"]
        kpts[f] = frames[f][i]["kpts"]
    return persons, kpts


def rank(frames: list[list[dict]], fps: float, frame_h: int, params: dict | None = None) -> list[dict]:
    """连轨迹、合并、每条单独计数，按"在跳绳、离镜头近、在画面里时间长"排序（见模块说明）。每行带 track。"""
    p = {**P, **(params or {})}
    n = len(frames)
    tracks = merge(frames, link(frames, fps, p), p)
    tracks = [t for t in tracks if len(t["frames"]) >= p["min_track_s"] * fps] or tracks
    rows = []
    for t in tracks:
        _, kp = series(frames, t, n)
        r = count_jumps(kp, fps, frame_h=frame_h)
        fs = sorted(t["frames"])
        rows.append({"track": t, "jumps": r.get("total", 0), "height": _height(frames, t), "from": fs[0], "to": fs[-1],
                     "frames": len(fs), "x": float(np.median([(frames[f][i]["box"][0] + frames[f][i]["box"][2]) / 2
                                                             for f, i in t["frames"].items()]))})
    if not rows:
        return []
    gate = max(p["min_jumps"], p["jump_gate"] * max(r["jumps"] for r in rows))
    for r in rows:
        r["jumping"] = r["jumps"] >= gate
        r["score"] = round(r["height"] * r["frames"] / n, 1)
    rows.sort(key=lambda r: (not r["jumping"], -r["score"], -r["jumps"]))
    return rows


def _summary(rows: list[dict]) -> list[dict]:
    return [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k != "track"} for r in rows]


def select(frames: list[list[dict]], fps: float, frame_h: int, params: dict | None = None) -> tuple[dict | None, list[dict]]:
    """返回 (选中的轨迹, 各轨迹的摘要（按分数从高到低）)。"""
    rows = rank(frames, fps, frame_h, params)
    return (rows[0]["track"] if rows else None), _summary(rows)


def select_pair(frames: list[list[dict]], fps: float, frame_h: int, params: dict | None = None) -> tuple[list | None, list[dict], bool]:
    """双人 PK：排名前两条、都在跳的轨迹，按画面左右排好。返回 ([左, 右] 或 None, 各轨迹摘要, 像不像两人一起跳)。

    "像"（auto 模式据此自动出 PK）：两人都在跳、同时在画面里的时间占短的那条的一半以上、跳数差不到一倍、个子差不到一倍——
    排除健身课后排也在跳的小人（4761432 前排 16 跳、身高 1268，后排 11 跳、身高 531）。
    """
    p = {**P, **(params or {})}
    rows = rank(frames, fps, frame_h, p)
    summary = _summary(rows)
    jumping = [r for r in rows if r["jumping"]]
    if len(jumping) < 2:
        return None, summary, False
    a, b = jumping[0], jumping[1]
    both = len(set(a["track"]["frames"]) & set(b["track"]["frames"]))
    together = (both >= p["pk_overlap"] * min(a["frames"], b["frames"])
                and min(a["jumps"], b["jumps"]) >= p["pk_jumps"] * max(a["jumps"], b["jumps"])
                and min(a["height"], b["height"]) >= p["pk_size"] * max(a["height"], b["height"]))
    left, right = sorted((a, b), key=lambda r: r["x"])
    return [left["track"], right["track"]], summary, together
