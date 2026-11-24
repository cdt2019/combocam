"""篮球运球计数回归测试：模拟的球心轨迹（带抖动）和人框、膝盖，不需要 GPU。

球在手和地面之间匀速往返（运球时手往下推，球下落和弹起都接近匀速），碰地那一帧是 y 的尖角；
人框高 1000 像素，球半径按人高的 0.065 倍；30 fps。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from combocam.dribble import ball_series, classify_moves, count_dribbles, facing  # noqa: E402
from combocam.dribble_track import build_track  # noqa: E402
from combocam.pose import L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER  # noqa: E402

FPS = 30.0
BOX = [300.0, 700.0, 700.0, 1700.0]  # 人框 x1, y1, x2, y2（脚底 1700）
R = 0.065 * (BOX[3] - BOX[1])
FLOOR = BOX[3] - R  # 球碰地时的球心
HAND = BOX[1] + 0.55 * (BOX[3] - BOX[1])  # 手（腰）的高度
KNEE_Y = BOX[1] + 0.75 * (BOX[3] - BOX[1])


def path(groups: list[int], period: float = 0.55, rest: float = 2.0, lead: float = 0.8, seed: int = 0):
    """每组连续运 n 下，组间球拿在手上 rest 秒。返回 (x, y, r, 真值触地帧)。"""
    rng = np.random.default_rng(seed)
    t, hits = lead, []
    for k, m in enumerate(groups):
        for _ in range(m):
            hits.append(t + period / 2)
            t += period
        t += rest if k + 1 < len(groups) else 0
    n = int((t + lead) * FPS)
    ts = np.arange(n) / FPS
    y = np.full(n, HAND)
    for h in hits:  # 每一下：从手到地再回到手，各用半个周期
        d = np.abs(ts - h)
        on = d < period / 2
        y[on] = FLOOR - (FLOOR - HAND) * d[on] / (period / 2)
    y += rng.normal(0, 1.5, n)
    x = 560 + rng.normal(0, 2, n)
    r = R + rng.normal(0, 1, n)
    return x, y, r, [int(round(h * FPS)) for h in hits]


def people(n: int):
    k = np.zeros((26, 3), np.float32)
    k[L_KNEE] = [450, KNEE_Y, 90]
    k[R_KNEE] = [550, KNEE_Y, 90]
    return [list(BOX)] * n, [k.reshape(-1).tolist()] * n


def run(x, y, r, params=None, bottom=None):
    persons, kpts = people(len(y))
    return count_dribbles(x, y, r, R, FPS, persons, kpts, params, bottom)


def near(pred, truth, tol=2):
    return len(pred) == len(truth) and all(abs(a - b) <= tol for a, b in zip(pred, truth))


def test_counts_each_floor_contact():
    x, y, r, truth = path([12])
    res = run(x, y, r)
    assert res["total"] == 12 and res["best"] == 12
    assert near([h["frame"] for h in res["dribbles"]], truth)
    assert abs(res["period_s"] - 0.55) < 0.05


def test_groups_split_on_pause():
    x, y, r, truth = path([8, 5], rest=2.5)
    res = run(x, y, r)
    assert [g["count"] for g in res["groups"]] == [8, 5]
    assert res["total"] == 13 and res["best"] == 8


def test_short_runs():
    """停下来以后只运两下也算一组（2026-10-11 定的规则）；只有一下的不算。"""
    x, y, r, _ = path([2, 6], rest=2.5)
    assert [g["count"] for g in run(x, y, r)["groups"]] == [2, 6]
    x, y, r, _ = path([1, 6], rest=2.5)
    assert [g["count"] for g in run(x, y, r)["groups"]] == [6]


def test_hand_turnaround_not_counted():
    """球在手里上下晃（慢、高处）不算：低于膝盖 + 急转两个条件都不满足。"""
    n = 200
    ts = np.arange(n) / FPS
    y = HAND + 40 * np.sin(2 * np.pi * ts / 0.6)
    res = run(np.full(n, 560.0), y, np.full(n, R))
    assert res["total"] == 0


def test_blurred_contact_inferred():
    """触地前后几帧球糊掉没检测到：用前后两段直线的交点补上。"""
    x, y, r, truth = path([10], seed=1)
    for h in truth:
        y[h - 1:h + 2] = np.nan
    res = run(x, y, r)
    assert near([h["frame"] for h in res["dribbles"]], truth)
    assert sum(h["inferred"] for h in res["dribbles"]) == 10


def test_long_gap_bridged():
    """更长的漏检（球碰地时半个出了画面、被身体挡住，约 0.3 秒）也算一下。"""
    x, y, r, truth = path([8], period=0.8, seed=2)
    for h in truth[2:5]:
        y[h - 4:h + 5] = np.nan
    res = run(x, y, r)
    assert near([h["frame"] for h in res["dribbles"]], truth, tol=3)


def test_far_from_player_not_counted():
    """人框左右各扩 0.6 个框宽以外的球（别人的球、滚走的球）不算。"""
    x, y, r, _ = path([8])
    res = run(x + 900, y, r)
    assert res["total"] == 0


def test_cut_by_frame_bottom():
    """球碰地时被画面底边切掉一截：检测框只有露出来的部分（球心偏高、半径偏小），按上沿修正后照样数到。"""
    x, y, r, truth = path([10], seed=3)
    bottom = FLOOR + 0.3 * R  # 真实画面底边，比球碰地时的底边高
    top = y - r
    cut = y + r > bottom
    r2, y2 = r.copy(), y.copy()
    r2[cut] = (bottom - top[cut]) / 2
    y2[cut] = top[cut] + r2[cut]
    res = run(x, y2, r2, bottom=bottom)
    assert near([h["frame"] for h in res["dribbles"]], truth)


def test_ball_series_drops_tiny_detections():
    """比按人框估的球小很多的检测（远处别的球、手、鞋上的反光）不进选轨迹。"""
    x, y, r, truth = path([8], seed=4)
    rng = np.random.default_rng(5)
    cands = []
    for f in range(len(y)):
        fr = [[float(x[f]), float(y[f]), float(r[f]), 0.9, 0]]
        if f % 3 == 0:
            fr.append([float(150 + rng.normal(0, 3)), float(1650 + rng.normal(0, 3)), 0.15 * R, 0.95, 0])
        cands.append(fr)
    persons, _ = people(len(y))
    xs, ys, rs, r_ref = ball_series(cands, FPS, persons)
    assert np.nanmax(np.abs(xs - x)) < 1e-6
    res = run(xs, ys, rs)
    assert res["total"] == 8


def test_build_track_format():
    x, y, r, truth = path([6, 7], rest=2.5, seed=6)
    persons, kpts = people(len(y))
    d = {"fps": FPS, "width": 1080, "height": 1920, "frames": len(y), "scene": None, "source": {},
         "cands": [[[float(a), float(b), float(c), 0.9, 0]] for a, b, c in zip(x, y, r)], "persons": persons, "kpts": kpts,
         "content_bottom": 1920.0}
    tr = build_track(d)
    assert tr["sport"] == "dribble" and tr["status"]["ok"]
    assert tr["summary"]["total"] == 13 and tr["summary"]["best"] == 7 and tr["summary"]["groups"] == 2
    assert [c["count"] for c in tr["combos"]] == [6, 7]
    assert len(tr["touches"]) == 13 and all(t["part"] == "floor" for t in tr["touches"])
    # 弹起高度：手（腰）到地面，约 0.45 倍人高 → 按球半径 0.12 m 换算
    h = np.median([a["height_m"] for a in tr["apex"]])
    expect = (FLOOR + R - HAND - R) / (R / 0.12)
    assert abs(h - expect) < 0.1
    assert tr["labels"]["unit"] == "下"


def test_build_track_no_dribble():
    n = 120
    persons, kpts = people(n)
    d = {"fps": FPS, "width": 1080, "height": 1920, "frames": n, "scene": None, "source": {},
         "cands": [[] for _ in range(n)], "persons": persons, "kpts": kpts}
    tr = build_track(d)
    assert not tr["status"]["ok"] and tr["status"]["code"] == "no_dribble"


# ---------- 动作：单手 / 交叉 / 胯下 ----------
# 面对镜头：人的左边在画面右边。两腿：左腿 x 600（髋 560、膝 590、踝 610），右腿 x 400（髋 440、膝 410、踝 390）
LEGS = {"L": (560.0, 590.0, 610.0), "R": (440.0, 410.0, 390.0)}
HIP_Y, KNEE_Y2, ANKLE_Y = 1200.0, 1450.0, 1680.0


def body(n: int, front: bool = True):
    k = np.zeros((26, 3), np.float32)
    s = 1 if front else -1  # 背对镜头时左右在画面上对调
    for side, (hip, kn, an) in LEGS.items():
        c = 500 + s * ((hip if side == "L" else hip) - 500)
        hi, ki, ai, si = (L_HIP, L_KNEE, L_ANKLE, L_SHOULDER) if side == "L" else (R_HIP, R_KNEE, R_ANKLE, R_SHOULDER)
        k[hi] = [c, HIP_Y, 90]
        k[ki] = [500 + s * (kn - 500), KNEE_Y2, 90]
        k[ai] = [500 + s * (an - 500), ANKLE_Y, 90]
        k[si] = [500 + s * (1.3 * (hip - 500)), 950, 90]
    return [list(BOX)] * n, [k.reshape(-1).tolist()] * n


def moves_path(seq: str, period: float = 0.6, seed: int = 0):
    """seq：每一下的动作，S 同一只手、C 体前交叉、L 胯下、B 背后；从人的右手（画面左边）开始。
    手托球在腰侧（左手 x 700、右手 x 300），触地点：同手在手下面，交叉在两脚前面（画面上比脚低一点），胯下在两脚连线上。
    胯下那一下触地后 2–5 帧球在小腿后面，检测不到。背后那一下从手到落地、再弹到另一只手的前一小段都在身后，整段检测不到
    （触地在身后、画面上比脚高）。"""
    rng = np.random.default_rng(seed)
    hand_x = {"L": 700.0, "R": 300.0}
    side, t, keys = "R", 0.6, []
    for m in seq:
        nxt = side if m == "S" else ("L" if side == "R" else "R")
        keys.append((t, side, nxt, m))
        side, t = nxt, t + period
    n = int((t + 0.6) * FPS)
    ts = np.arange(n) / FPS
    x, y = np.full(n, hand_x["R"]), np.full(n, HAND)
    hits, hidden = [], []
    for t0, a, b, m in keys:
        th = t0 + period / 2
        floor = FLOOR + (0.8 * R if m == "C" else 0.0)
        bx = hand_x[a] if m == "S" else 500.0
        for i in np.flatnonzero((ts >= t0) & (ts < t0 + period)):
            u = (ts[i] - t0) / period
            if u < 0.5:
                x[i] = hand_x[a] + (bx - hand_x[a]) * u / 0.5
                y[i] = HAND + (floor - HAND) * u / 0.5
            else:
                x[i] = bx + (hand_x[b] - bx) * (u - 0.5) / 0.5
                y[i] = floor - (floor - HAND) * (u - 0.5) / 0.5
        f = int(round(th * FPS))
        hits.append(f)
        if m == "L":
            hidden += list(range(f + 2, f + 6))
        if m == "B":
            # 从手往下走一小段就躲到身后，弹起来快到另一只手才出来
            a0, a1 = int(round((t0 + 0.12 * period) * FPS)), int(round((t0 + 0.82 * period) * FPS))
            hidden += list(range(a0, a1))
            sel = (ts >= t0) & (ts < t0 + period)
            y[sel] = np.minimum(y[sel], FLOOR - 3.5 * R)  # 触地在身后，画面上的位置比脚高
    x[ts >= t] = hand_x[side]  # 最后一下之后球留在最后那只手上
    x += rng.normal(0, 1.5, n)
    y += rng.normal(0, 1.5, n)
    r = R + rng.normal(0, 0.8, n)
    y[hidden] = np.nan
    return x, y, r, hits


def run_moves(seq: str, front: bool = True, seed: int = 0):
    x, y, r, hits = moves_path(seq, seed=seed)
    persons, kpts = body(len(y), front)
    res = count_dribbles(x, y, r, R, FPS, persons, kpts, {"min_group": 1})
    return res, hits


def test_facing():
    assert facing(body(50, True)[1]) == "front"
    assert facing(body(50, False)[1]) == "back"


def test_moves_same_cross_legs():
    seq = "SSCLCLSCC"
    res, hits = run_moves(seq)
    assert res["view"] == "front"
    got = {h["frame"]: h for h in res["dribbles"]}
    assert len(got) == len(seq)
    want = {"S": "same", "C": "cross", "L": "legs"}
    for f, m in zip(hits, seq):
        h = got[min(got, key=lambda g: abs(g - f))]
        assert h["move"] == want[m], (f, m, h)
    assert res["moves"] == {"same": 3, "cross": 4, "legs": 2, "behind": 0}


def test_moves_hands():
    res, _ = run_moves("SC")
    a, b = res["dribbles"]
    assert (a["hand_from"], a["hand_to"]) == ("R", "R")
    assert (b["hand_from"], b["hand_to"]) == ("R", "L")


def test_moves_not_judged_from_behind():
    """背对镜头时左右手对不上、腿挡球的关系也反过来：不判动作。"""
    res, _ = run_moves("SCLC", front=False)
    assert res["view"] == "back"
    assert all(h["move"] is None for h in res["dribbles"])


def test_frame_edge_is_not_leg():
    """触地时球出了真实画面底边（横屏素材），之后几帧没检测到：不能当成被腿挡住的胯下。"""
    x, y, r, hits = moves_path("SCCS", seed=3)
    bottom = FLOOR + 0.5 * R
    for f in hits:
        y[f - 2:f + 5] = np.nan  # 触地前后都出了画面
    persons, kpts = body(len(y))
    res = count_dribbles(x, y, r, R, FPS, persons, kpts, {"min_group": 1}, bottom, 1080.0)
    mv = [h["move"] for h in res["dribbles"]]
    assert "legs" not in mv and mv.count("cross") == 2


def test_classify_handles_empty():
    assert classify_moves([], np.array([]), np.array([]), np.array([]), FPS, np.zeros((0, 4)), []) is None


def test_behind_the_back():
    """背后运球：从右手躲到身后，在身后触地，从左手那边出来。触地看不见，也照样算一下、判成背后。"""
    seq = "SCSBSC"
    res, hits = run_moves(seq, seed=7)
    got = sorted(h["frame"] for h in res["dribbles"])
    assert len(got) == len(seq), got
    mv = {h["frame"]: h for h in res["dribbles"]}
    b = mv[min(mv, key=lambda g: abs(g - hits[3]))]
    assert b["move"] == "behind" and abs(b["frame"] - hits[3]) <= 4
    assert (b["hand_from"], b["hand_to"]) == ("L", "R") or (b["hand_from"], b["hand_to"]) == ("R", "L")
    assert res["moves"]["behind"] == 1 and res["moves"]["cross"] == 2


def test_behind_not_from_behind_view():
    """背对镜头时不找背后运球（球在身前被身体挡住也是这个样子）。"""
    res, _ = run_moves("SBS", front=False, seed=8)
    assert all(h.get("move") != "behind" for h in res["dribbles"])


# ---------- 暗光片段：球离得远显得小、一边慢、节奏不匀 ----------
def vpath(bounces: list[tuple[float, float, float]], n: int, r: float = R):
    """每一下 (触地时刻 s, 下落速度, 弹起速度)，速度单位 球半径/秒；两下之间球回到手的高度停住。"""
    ts = np.arange(n) / FPS
    y = np.full(n, HAND)
    for t0, vd, vu in bounces:
        dn = (ts <= t0) & (ts >= t0 - (FLOOR - HAND) / (vd * r))
        up = (ts > t0) & (ts <= t0 + (FLOOR - HAND) / (vu * r))
        y[dn] = np.maximum(y[dn], FLOOR - vd * r * (t0 - ts[dn]))
        y[up] = np.maximum(y[up], FLOOR - vu * r * (ts[up] - t0))
    return np.full(n, 560.0), y, np.full(n, r)


def test_one_side_slower():
    """拍下去快、弹起来慢（斜着拍、球在最低处停了一下）：慢的那边够 v_min 的一半就算。"""
    x, y, r = vpath([(1.0, 12, 12), (2.0, 12, 3.6), (3.0, 12, 12), (4.0, 12, 2.0)], 150)
    res = run(x, y, r, {"min_group": 1})
    got = [h["frame"] for h in res["dribbles"]]
    assert len(got) == 3 and all(abs(a - b) <= 2 for a, b in zip(got, [30, 60, 90])), got


def test_local_radius():
    """后半段球离得远、或者下半截在暗处没框进去，检测框只有整段中位数的七成：同样的动作在画面上折成的"半径/秒"变大，
    门槛按附近的球大小算，慢一点的那几下也数得到。"""
    n = 240
    x, y1, _ = vpath([(t, 12, 12) for t in (0.6, 1.4, 2.2, 3.0)], n)
    _, y2, _ = vpath([(t, 5, 5) for t in (4.6, 5.4, 6.2, 7.0)], n)  # 按整段半径算只有 5 个半径/秒，不够 v_min
    y = np.where(np.arange(n) < 120, y1, y2)
    r = np.where(np.arange(n) < 120, R, 0.7 * R)
    persons, kpts = people(n)
    res = count_dribbles(x, y, r, R, FPS, persons, kpts, {"min_group": 1})
    assert len(res["dribbles"]) == 8, [h["frame"] for h in res["dribbles"]]
    res = count_dribbles(x, y, r, R, FPS, persons, kpts, {"min_group": 1, "local_r": False})
    assert len(res["dribbles"]) == 4


def test_uneven_rhythm_short_group():
    """暗处漏掉几下、换动作时快慢不一（10341426）：间隔 0.73 秒、1.67 秒的三下还算一组——只有一个间隔时不按它定节奏。"""
    x, y, r = vpath([(1.0, 12, 12), (1.73, 12, 12), (3.4, 12, 12)], 130)
    res = run(x, y, r)
    assert [g["count"] for g in res["groups"]] == [3]


# ---------- 触地在画面底边外 ----------
def test_duplicate_frames_removed():
    """25 帧的视频转成 30 帧：每 6 帧重复一帧（人框、关键点和上一帧一模一样）。去掉重复帧在原视频的帧上数，
    帧号、时间要换回成片的。（重复帧夹在触地前后让速度量成 0 的真实例子是 6070824 第 278 帧，这里只验证换算。）"""
    n = 180
    src = [int(k * 25 / 30) for k in range(n)]  # 第 k 帧显示原视频第几帧
    ts = np.array(src) / 25.0
    hits_t = [1.0, 1.6, 2.2, 2.8, 3.4, 4.0, 4.6, 5.2]
    y = np.full(n, HAND)
    for t0 in hits_t:
        d = np.abs(ts - t0)
        on = d < 0.3
        y[on] = FLOOR - (FLOOR - HAND) * d[on] / 0.3
    x, r = np.full(n, 560.0), np.full(n, R)
    persons, kpts = people(n)
    persons = [[b[0] + 0.01 * s_, b[1], b[2], b[3]] for b, s_ in zip(persons, src)]  # 同一原视频帧的人框一模一样
    kp = np.array(kpts[0], np.float32)
    kpts = [(kp + 0.001 * s_).tolist() for s_ in src]
    res = count_dribbles(x, y, r, R, FPS, persons, kpts)
    got = [h["frame"] for h in res["dribbles"]]
    want = [int(round(t0 * 30)) for t0 in hits_t]
    assert len(got) == 8 and all(abs(a - b) <= 2 for a, b in zip(got, want)), (got, want)
    assert abs(res["dribbles"][3]["t"] - 2.8) < 0.08


def test_contact_below_frame_bottom():
    """触地在画面底边外：球的下半截出了画面，触地前后几帧检测不到、看得见的最低点只比底边高一点。
    弹起高度按两段直线的交点算，不按看得见的最低点。"""
    x, y, r, truth = path([10], period=0.5, seed=9)
    low = y > FLOOR - 1.6 * R  # 触地前后球的下半截出了画面：看不见
    y2 = np.where(low, np.nan, y)
    # prom_r 5：看得见的弹起高度（4.3 个半径）不够，按触地点算（5.9 个半径）够
    res = count_dribbles(x, y2, r, R, FPS, *people(len(y)), {"min_group": 1, "prom_r": 5.0}, FLOOR + R - 1.6 * R)
    got = [h["frame"] for h in res["dribbles"]]
    assert len(got) == 10 and all(abs(a - b) <= 2 for a, b in zip(got, truth)), (got, truth)


def test_dup_frames_pattern():
    from combocam.dribble import dup_frames
    n = 60
    src = [int(k * 25 / 30) for k in range(n)]
    persons = [[float(s_), 0.0, 1.0, 1.0] for s_ in src]
    dup = dup_frames(persons, None, n)
    assert dup.sum() == n - len(set(src)) and not (dup[1:] & dup[:-1]).any()
    # 定格画面（连着好几帧一模一样）、构造的数据（每帧都一样）：不当帧率换算的重复帧
    assert not dup_frames([[0.0, 0.0, 1.0, 1.0]] * n, None, n).any()
    frozen = [[float(min(k, 20)), 0.0, 1.0, 1.0] for k in range(n)]
    assert not dup_frames(frozen, None, n).any()
