"""跳绳计数回归测试：按自由落体模拟的人体关键点（带抖动），不需要 GPU。

腾空时髋部和脚按 g = 9.8 m/s² 走抛物线，着地时屈膝下沉；身高 1000 像素按 1.65 m 换算。
慢动作版本把同一段动作按 2 倍时间采样，用来测视在重力估慢放倍数。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from combocam.jumprope import count_jumps  # noqa: E402
from combocam.jumprope_track import build_track  # noqa: E402
from combocam.pose import (HEAD, L_ANKLE, L_BIG_TOE, L_HEEL, L_HIP, L_KNEE, L_SHOULDER, L_SMALL_TOE, L_WRIST, NOSE,  # noqa: E402
                         R_ANKLE, R_BIG_TOE, R_HEEL, R_HIP, R_KNEE, R_SHOULDER, R_SMALL_TOE, R_WRIST)

FPS = 30.0
TALL = 1000.0  # 头顶到脚底（像素）
PX_PER_M = TALL / 1.65
SOLE = 1700.0


def plan(n_jumps: list[int], period: float = 0.42, air: float = 0.22, rest: float = 1.3, lead: float = 1.0) -> tuple[list, float]:
    """每组 n 跳，组间站着休息 rest 秒。返回 [(离地时刻, 腾空时长)]、总时长。"""
    t, out = lead, []
    for k, m in enumerate(n_jumps):
        for _ in range(m):
            out.append((t, air))
            t += period
        if k + 1 < len(n_jumps):
            t += rest
    return out, t + lead


def lift_at(jumps: list, t: float) -> tuple[float, float]:
    """(离地高度 m, 屈膝下沉 m)。"""
    for t0, air in jumps:
        if t0 <= t <= t0 + air:
            s = t - t0
            return 9.8 * air / 2 * s - 9.8 * s * s / 2, 0.0
    # 着地时屈膝：落地后到下一次起跳之间下沉一点
    prev = [t0 + air for t0, air in jumps if t0 + air < t]
    nxt = [t0 for t0, air in jumps if t0 > t]
    if prev and nxt and nxt[0] - prev[-1] < 0.4:
        a, b = prev[-1], nxt[0]
        return 0.0, 0.03 * np.sin(np.pi * (t - a) / (b - a))
    return 0.0, 0.0


def make_kpts(jumps: list, t_end: float, slow: float = 1.0, seed: int = 0, rope=None) -> list:
    """rope(t)：可选，绳子转到第几圈（实数），手腕按它画小圆——一圈手腕上下一次。"""
    rng = np.random.default_rng(seed)
    n = int(t_end * slow * FPS)
    out = []
    for f in range(n):
        z, dip = lift_at(jumps, f / FPS / slow)
        wy = 0.0 if rope is None else 40 * np.cos(2 * np.pi * rope(f / FPS / slow))
        up = (z - dip) * PX_PER_M  # 身体整体上移（像素）
        feet = z * PX_PER_M
        k = np.zeros((26, 3))
        pts = {HEAD: (540, SOLE - TALL - up), NOSE: (540, SOLE - TALL + 60 - up),
               L_SHOULDER: (480, SOLE - 820 - up), R_SHOULDER: (600, SOLE - 820 - up),
               L_WRIST: (430, SOLE - 520 - up + wy), R_WRIST: (650, SOLE - 520 - up + wy),
               L_HIP: (500, SOLE - 520 - up), R_HIP: (580, SOLE - 520 - up),
               L_KNEE: (505, SOLE - 270 - (up + feet) / 2), R_KNEE: (575, SOLE - 270 - (up + feet) / 2),
               L_ANKLE: (505, SOLE - 40 - feet), R_ANKLE: (575, SOLE - 40 - feet)}
        for i, x in ((L_BIG_TOE, 490), (L_SMALL_TOE, 480), (L_HEEL, 510), (R_BIG_TOE, 590), (R_SMALL_TOE, 600), (R_HEEL, 570)):
            pts[i] = (x, SOLE - feet)
        for i, (x, y) in pts.items():
            k[i] = (x + rng.normal(0, 1.5), y + rng.normal(0, 1.5), 90)
        out.append([int(round(v)) for v in k.ravel()])
    return out


def pose_data(kpts: list) -> dict:
    n = len(kpts)
    return {"fps": FPS, "width": 1080, "height": 1920, "frames": n, "content_bottom": 1920.0, "kpts": kpts,
            "persons": [[400.0, 650.0, 700.0, 1710.0]] * n, "filter": "fps=30,scale=1080:1920", "probe": {"has_audio": False}}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_counts_jumps_and_breaks(seed):
    jumps, t_end = plan([12, 8])
    r = count_jumps(make_kpts(jumps, t_end, seed=seed), FPS)
    assert r["total"] == 20
    assert [g["count"] for g in r["groups"]] == [12, 8]
    assert len(r["breaks"]) == 1
    assert abs(r["groups"][0]["bpm"] - 60 / 0.42) < 8


def test_standing_still_is_not_counted():
    jumps, t_end = plan([0], lead=3.0)
    r = count_jumps(make_kpts([], t_end), FPS)
    assert r["total"] == 0


@pytest.mark.parametrize("slow", [1.0, 2.0])
def test_slow_motion_is_detected_from_gravity(slow):
    jumps, t_end = plan([15])
    tr = build_track(pose_data(make_kpts(jumps, t_end, slow=slow)))
    sm = tr["summary"]
    assert sm["total"] == 15
    if slow == 1.0:
        assert not sm["slowmo"] and sm["speed"] == 1.0
    else:
        assert sm["slowmo"] and abs(sm["speed"] - slow) < 0.35
    # 换回真实时间的跳速和模拟的一致
    assert abs(sm["bpm"] - 60 / 0.42) < 10
    # 每跳的离地、落地时刻落在模拟的腾空段附近
    for j, (t0, air) in zip(tr["jumps"], jumps):
        assert abs(j["up_t"] - t0 * slow) < 0.1 * slow + 0.05
        assert abs(j["land_t"] - (t0 + air) * slow) < 0.1 * slow + 0.05


# ---- 多人入镜（combocam/people.py）----
HIST_A = [1000] + [0] * 31  # 两种衣服颜色
HIST_B = [0] * 31 + [1000]


def shifted(kpts: list, dx: float, dy: float = 0.0) -> list:
    out = []
    for k in kpts:
        a = np.asarray(k, np.float64).reshape(-1, 3)
        a[:, 0] += dx
        a[:, 1] += dy
        out.append([int(round(v)) for v in a.ravel()])
    return out


def person(k: list, hist: list) -> dict:
    a = np.asarray(k, np.float64).reshape(-1, 3)
    return {"box": [float(a[:, 0].min() - 40), float(a[:, 1].min() - 20), float(a[:, 0].max() + 40), float(a[:, 1].max() + 10)],
            "kpts": k, "hist": hist}


def test_picks_the_jumper_not_the_biggest_person():
    jumps, t_end = plan([15])
    jumper = make_kpts(jumps, t_end, seed=3)
    # 离镜头更近（大一圈）、站着不动的人
    big = [[int(round(v * (1.25 if i % 3 < 2 else 1))) for i, v in enumerate(k)] for k in shifted(make_kpts([], t_end, seed=4), -380, -420)]
    frames = [[person(b, HIST_B), person(j, HIST_A)] for b, j in zip(big, jumper)]
    d = {**pose_data(jumper), "frames_people": frames}
    tr = build_track(d)
    assert tr["summary"]["total"] == 15
    assert tr["people"][0]["jumps"] == 15 and tr["people"][1]["jumps"] == 0


def test_rejoins_jumper_after_leaving_frame():
    jumps, t_end = plan([30])
    jumper = make_kpts(jumps, t_end, seed=5)
    other = shifted(make_kpts([], t_end, seed=6), 400, 0)  # 镜头摇过去以后，另一个人在画面里的位置不同
    n = len(jumper)
    gone = range(int(4.0 * FPS), int(6.5 * FPS))  # 镜头摇到另一个人身上 2.5 秒
    frames = [[person(other[f], HIST_B)] if f in gone else [person(jumper[f], HIST_A)] for f in range(n)]
    tr = build_track({**pose_data(jumper), "frames_people": frames})
    assert len(tr["people"]) == 2 and tr["people"][0]["jumps"] > tr["people"][1]["jumps"]
    assert tr["breaks"] == [] and len(tr["absent"]) == 1  # 离开画面不算断绳
    visible = sum(1 for t0, air in jumps if int((t0 + air / 2) * FPS) not in gone)
    assert visible - 2 <= tr["summary"]["total"] <= visible


def test_single_frame_color_glitch_does_not_split_track():
    # 6326809：单帧关键点出错，躯干取错地方，颜色突然不像；前后都对，不能断成两条
    jumps, t_end = plan([20])
    jumper = make_kpts(jumps, t_end, seed=7)
    frames = [[person(k, HIST_B if f == 150 else HIST_A)] for f, k in enumerate(jumper)]
    tr = build_track({**pose_data(jumper), "frames_people": frames})
    assert len(tr["people"]) == 1 and tr["summary"]["total"] == 20


def test_hard_cut_to_another_person_splits_track():
    # 剪辑切到另一个人：人框还重叠不少（约 0.66），但位置变了、衣服颜色不同，要拆开，不能接着数
    jumps, t_end = plan([24])
    jumper = make_kpts(jumps, t_end, seed=8)
    other = shifted(make_kpts([], t_end, seed=9), 150, 0)  # 人框重叠约 0.66
    cut = int(5.0 * FPS)
    frames = [[person(jumper[f], HIST_A)] if f < cut else [person(other[f], HIST_B)] for f in range(len(jumper))]
    tr = build_track({**pose_data(jumper), "frames_people": frames})
    assert len(tr["people"]) == 2
    assert tr["people"][0]["to"] < cut and tr["people"][1]["from"] >= cut
    assert tr["people"][1]["jumps"] == 0


# ---- 双人 PK ----
def two_jumpers(n_left: int, n_right: int, scale_right: float = 1.0):
    jl, tl_ = plan([n_left], period=0.42)
    jr, tr_ = plan([n_right], period=0.40)
    t_end = max(tl_, tr_)
    left = shifted(make_kpts(jl, t_end, seed=10), -280)
    right = make_kpts(jr, t_end, seed=11)
    if scale_right != 1.0:  # 后排的小人：以脚底为基准缩小
        right = [[int(round(v if i % 3 == 2 else (v - 540) * scale_right + 540 + 300 if i % 3 == 0 else (v - SOLE) * scale_right + SOLE))
                  for i, v in enumerate(k)] for k in right]
    else:
        right = shifted(right, 280)
    frames = [[person(a, HIST_A), person(b, HIST_B)] for a, b in zip(left, right)]
    return {**pose_data(left), "frames_people": frames}


def test_pk_counts_both_players_left_to_right():
    d = two_jumpers(14, 18)
    tr = build_track(d, mode="pk")
    assert tr["mode"] == "pk" and [pl["side"] for pl in tr["players"]] == ["left", "right"]
    assert tr["summary"]["totals"] == [14, 18] and tr["summary"]["winner"] == 1
    assert tr["status"]["ok"]
    # auto：两人一样大、一起跳，自动出 PK
    assert build_track(d, mode="auto")["mode"] == "pk"


def test_auto_stays_single_when_second_jumper_is_small_background_person():
    # 健身课：前排的人跳，后排的小人也在跳（4761432）——不是 PK
    d = two_jumpers(16, 16, scale_right=0.4)
    tr = build_track(d, mode="auto")
    assert tr["mode"] == "single" and tr["summary"]["total"] == 16
    assert not tr["pk_candidate"]


def test_lead_changes_ignore_alternating_landings():
    from combocam.jumprope_track import lead_changes

    def pl(ts):
        return {"jumps": [{"t": t} for t in ts]}
    # 两人同步跳，谁先落地每跳都在换：不算换领先
    assert lead_changes([pl([i * 0.4 for i in range(20)]), pl([i * 0.4 + 0.05 for i in range(20)])]) == 0
    # 左边先领先 3 个，后来右边加速反超到领先 2 个以上：换 1 次
    left = [i * 0.4 for i in range(20)]
    right = [1.2 + i * 0.3 for i in range(26)]
    assert lead_changes([pl(left), pl(right)]) == 1


# ---- 遮挡 ----
def test_person_hidden_mid_run_keeps_counting():
    # 有人从镜头前走过，跳绳的人 0.8 秒看不见：不算断绳，按节拍补上看不见的那两跳
    jumps, t_end = plan([24])
    kp = make_kpts(jumps, t_end, seed=12)
    a, b = int(4.0 * FPS), int(4.8 * FPS)
    kp = [None if a <= f < b else k for f, k in enumerate(kp)]
    r = count_jumps(kp, FPS)
    assert r["groups"] and len(r["groups"]) == 1 and r["breaks"] == []
    assert r["inferred"] >= 1 and abs(r["total"] - 24) <= 1


def test_feet_hidden_mid_run_still_counts():
    # 脚被挡住两秒（关键点置信度为 0）：只看髋部，照样数
    jumps, t_end = plan([24])
    kp = make_kpts(jumps, t_end, seed=13)
    a, b = int(3.0 * FPS), int(5.0 * FPS)
    for f in range(a, b):
        arr = np.asarray(kp[f]).reshape(-1, 3)
        arr[[L_ANKLE, R_ANKLE, L_BIG_TOE, R_BIG_TOE, L_SMALL_TOE, R_SMALL_TOE, L_HEEL, R_HEEL], 2] = 0
        kp[f] = arr.ravel().tolist()
    assert count_jumps(kp, FPS)["total"] == 24


# ---- 双摇 ----
def rope_turns(jumps: list, per_jump: list[int]):
    """绳子转的圈数随时间：第 k 跳离地到下一跳离地之间转 per_jump[k] 圈（匀速）。"""
    starts = [t0 for t0, _ in jumps]

    def f(t):
        if t <= starts[0]:
            return (t - starts[0]) / (starts[1] - starts[0])
        for k in range(len(starts)):
            end = starts[k + 1] if k + 1 < len(starts) else starts[k] + (starts[k] - starts[k - 1])
            if t < end:
                return sum(per_jump[:k]) + per_jump[k] * (t - starts[k]) / (end - starts[k])
        return sum(per_jump) + (t - end) / (end - starts[-1])
    return f


def test_double_unders_after_singles():
    # 先 8 个单摇（一跳 0.42 秒、腾空 0.22 秒，绳子一跳一圈），再 6 个双摇（一跳 0.62 秒、腾空 0.40 秒，一跳两圈）
    singles, t1 = plan([8], period=0.42, air=0.22, lead=1.0)
    doubles, t_end = plan([6], period=0.62, air=0.40, lead=t1 - 1.0 + 0.42)
    jumps = singles + doubles
    per = [1] * 8 + [2] * 6
    kp = make_kpts(jumps, t_end, seed=14, rope=rope_turns(jumps, per))
    tr = build_track(pose_data(kp))
    sm = tr["summary"]
    assert sm["total"] == 14
    passes = [j["passes"] for j in tr["jumps"]]
    assert passes[:7] == [1] * 7 and passes[9:13] == [2] * 4  # 交界处的一跳允许判错
    assert 5 <= sm["doubles"] <= 7 and sm["style"] == "mixed"


def test_singles_with_high_jumps_are_not_doubles():
    # 跳得高、腾空久，但绳子一跳只转一圈：不是双摇
    jumps, t_end = plan([12], period=0.62, air=0.40)
    kp = make_kpts(jumps, t_end, seed=15, rope=rope_turns(jumps, [1] * 12))
    sm = build_track(pose_data(kp))["summary"]
    assert sm["total"] == 12 and sm["doubles"] == 0 and sm["style"] == "single"


# ---- 一圈跳两下 ----
def test_double_bounce_counts_rope_passes():
    # 每 0.3 秒跳一下、一样高，绳子两下才转一圈（绳子在头顶时垫一下）：按过绳算，只有一半
    jumps, t_end = plan([24], period=0.3, air=0.18)
    kp = make_kpts(jumps, t_end, seed=16, rope=rope_turns(jumps, [0.5] * 24))
    r = count_jumps(kp, FPS)
    assert r["bounce"] == "double"
    assert r["total"] == 12
    # 留下的是过绳的那几下（离地时绳子转到整圈，手腕在最低点附近）
    kept = [j["frame"] for j in r["jumps"]]
    pass_frames = [int(round((t0 + air / 2) * FPS)) for k, (t0, air) in enumerate(jumps) if k % 2 == 0]
    assert all(min(abs(f - g) for g in pass_frames) <= 2 for f in kept)


def test_normal_single_bounce_is_untouched():
    jumps, t_end = plan([20], period=0.42, air=0.22)
    kp = make_kpts(jumps, t_end, seed=17, rope=rope_turns(jumps, [1] * 20))
    r = count_jumps(kp, FPS)
    assert r["bounce"] == "single" and r["total"] == 20


# ---- 双人 PK 的双摇 ----
def pk_pair(left_spec, right_spec):
    """left_spec / right_spec：(单摇几个, 双摇几个)。单摇一跳 0.42 秒、腾空 0.22 秒；双摇一跳 0.62 秒、腾空 0.40 秒，一跳两圈。"""
    def one(spec, seed, dx):
        n1, n2 = spec
        s1, t1 = plan([n1], period=0.42, air=0.22, lead=1.0) if n1 else ([], 1.0)
        s2, t_end = plan([n2], period=0.62, air=0.40, lead=(t1 - 1.0 + 0.42) if n1 else 1.0) if n2 else ([], t1)
        jumps = s1 + s2
        kp = make_kpts(jumps, max(t_end, 12.0), seed=seed, rope=rope_turns(jumps, [1] * n1 + [2] * n2))
        return shifted(kp, dx)
    a, b = one(left_spec, 20, -280), one(right_spec, 21, 280)
    n = min(len(a), len(b))
    frames = [[person(a[f], HIST_A), person(b[f], HIST_B)] for f in range(n)]
    return {**pose_data(a[:n]), "frames_people": frames}


def test_pk_double_unders_auto_scores_doubles():
    tr = build_track(pk_pair((0, 14), (0, 11)), mode="pk")
    sm = tr["summary"]
    assert sm["rule"] == "doubles"
    assert abs(sm["doubles"][0] - 14) <= 1 and abs(sm["doubles"][1] - 11) <= 1
    assert sm["scores"] == sm["doubles"] and sm["winner"] == 0


def test_pk_mixed_scores_total_jumps_and_reports_doubles():
    tr = build_track(pk_pair((10, 5), (16, 0)), mode="pk")
    sm = tr["summary"]
    assert sm["rule"] == "jumps"
    assert sm["totals"] == [15, 16] and sm["scores"] == [15, 16]
    assert 4 <= sm["doubles"][0] <= 6 and sm["doubles"][1] == 0
    # 换成比过绳次数：双摇算 2，左边反超
    sp = build_track(pk_pair((10, 5), (16, 0)), mode="pk", pk_rule="passes")["summary"]
    assert sp["rule"] == "passes" and sp["scores"][0] == 15 + sp["doubles"][0] and sp["winner"] == 0
