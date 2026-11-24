"""计数逻辑回归测试：真值轨迹 + 模拟检测噪声（漏检、抖动、误检、场边静止球），不需要 GPU。"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from make_synthetic import CX, FPS, GROUND, PX_PER_M, R, ball_at, build_plan  # noqa: E402

from combocam.analyze import postprocess  # noqa: E402
from combocam.track import Cand  # noqa: E402


def simulate(seed: int, recall: float, noise_px: float, fp_rate: float):
    flights, touches, bounces, t_end = build_plan(7)
    rng = np.random.default_rng(seed)
    n = int(t_end * FPS)
    cands, persons = [], []
    for i in range(n):
        x, y = ball_at(flights, i / FPS)
        fr = []
        if rng.random() < recall:
            fr.append(Cand(x + rng.normal(0, noise_px), y + rng.normal(0, noise_px), R * rng.uniform(0.95, 1.08), rng.uniform(0.15, 0.95)))
        if rng.random() < fp_rate:
            fr.append(Cand(rng.uniform(0, 1080), rng.uniform(0, 1920), R * rng.uniform(0.5, 1.5), rng.uniform(0.05, 0.3)))
        fr.append(Cand(150 + rng.normal(0, 1), GROUND + 140 - R + rng.normal(0, 1), R, 0.9))  # 场边静止的球
        cands.append(fr)
        persons.append((CX - 200, GROUND - 1.85 * PX_PER_M, CX + 200, GROUND))
    return cands, persons, touches, bounces


def match(pred, truth, tol=2.5 / FPS):
    used, tp = set(), 0
    for p in pred:
        best = min(((abs(p - t), i) for i, t in enumerate(truth) if i not in used and abs(p - t) <= tol), default=None)
        if best:
            used.add(best[1])
            tp += 1
    return tp, len(pred) - tp


@pytest.mark.parametrize("seed,recall,noise,fp", [(1, 0.95, 1.5, 0.05), (2, 0.8, 3.0, 0.1), (3, 0.65, 4.0, 0.15)])
def test_touch_count_and_combos(seed, recall, noise, fp):
    cands, persons, touches, bounces = simulate(seed, recall, noise, fp)
    _, _, ev = postprocess(cands, persons, FPS, 1920)
    tp, fp_n = match([t["t"] for t in ev.touches], [t["t"] for t in touches])
    assert tp >= len(touches) - 1, f"漏检太多：{tp}/{len(touches)}"
    assert fp_n <= 1, f"误报 {fp_n}"
    assert [c["count"] for c in ev.combos] in ([14, 8], [13, 8], [14, 7]), ev.combos
    # 落地反弹不计数，且至少识别出两段之间那次落地
    assert any(b["t"] > touches[13]["t"] and b["t"] < touches[14]["t"] for b in ev.bounces)


def test_height_scale_from_ball_size():
    cands, persons, touches, _ = simulate(4, 0.95, 1.5, 0.0)
    _, _, ev = postprocess(cands, persons, FPS, 1920)
    assert abs(ev.calib["px_per_m"] / PX_PER_M - 1) < 0.1
    assert ev.calib["px_per_m_gravity"] and abs(ev.calib["px_per_m_gravity"] / PX_PER_M - 1) < 0.15


def test_static_ball_only_gives_no_touches():
    cands = [[Cand(500, 1500, 60, 0.9)] for _ in range(300)]
    _, _, ev = postprocess(cands, [None] * 300, FPS, 1920)
    assert ev.touches == []


def test_verdict():
    """不出片的判定：没球、没人、一个都没颠起来；颠到 1 个就出片。"""
    from combocam.verdict import verdict

    assert verdict(300, 290, 300, 12)["ok"]
    assert verdict(300, 290, 300, 1)["ok"]
    assert verdict(300, 10, 300, 0)["code"] == "no_ball"
    assert verdict(300, 290, 20, 5)["code"] == "no_person"
    v = verdict(300, 290, 300, 0)
    assert v["code"] == "no_juggling" and v["title"] and v["message"] and v["tips"]


def test_simulated_juggling_is_accepted():
    from combocam.verdict import verdict

    cands, persons, touches, _ = simulate(1, 0.95, 1.5, 0.05)
    track, _, ev = postprocess(cands, persons, FPS, 1920)
    best = max(c["count"] for c in ev.combos)
    assert verdict(len(cands), len(track), sum(1 for p in persons if p), best)["ok"]


# ---------- 2026-10-12：球没碰到人的被动反弹、从地上挑起（第 4 批留出集） ----------
from combocam.events import analyze_track  # noqa: E402
from combocam.pose import (HEAD, L_ANKLE, L_BIG_TOE, L_ELBOW, L_HEEL, L_HIP, L_KNEE, L_SHOULDER, L_SMALL_TOE, L_WRIST,  # noqa: E402
                         NECK, NOSE, R_ANKLE, R_BIG_TOE, R_ELBOW, R_HEEL, R_HIP, R_KNEE, R_SHOULDER, R_SMALL_TOE, R_WRIST)

BR, SOLE_Y, PX = 40.0, 1700.0, 40.0 / 0.11  # 球半径（像素）、脚底、每米像素
GPX = 9.8 * PX  # 重力（像素/秒²）


def _standing(n: int) -> tuple[list, list]:
    k = np.zeros((26, 3), np.float32)
    pts = {L_BIG_TOE: (470, SOLE_Y), R_BIG_TOE: (530, SOLE_Y), L_SMALL_TOE: (465, SOLE_Y - 5), R_SMALL_TOE: (535, SOLE_Y - 5),
           L_HEEL: (480, SOLE_Y - 8), R_HEEL: (520, SOLE_Y - 8), L_ANKLE: (480, SOLE_Y - 20), R_ANKLE: (520, SOLE_Y - 20),
           L_KNEE: (480, 1450), R_KNEE: (520, 1450), L_HIP: (480, 1220), R_HIP: (520, 1220), L_SHOULDER: (450, 950),
           R_SHOULDER: (550, 950), NECK: (500, 930), NOSE: (500, 860), HEAD: (500, 800), L_ELBOW: (430, 1100),
           R_ELBOW: (570, 1100), L_WRIST: (430, 1230), R_WRIST: (570, 1230)}
    for i, (x, y) in pts.items():
        k[i] = [x, y, 90]
    return [k.reshape(-1).astype(int).tolist()] * n, [(420.0, 780.0, 580.0, SOLE_Y)] * n


def _bounce_track(x: float, fall_px: float, ratio: float, n_up: int = 18) -> dict:
    """球从 fall_px 高处落到脚底那条线（球底贴地），弹回 ratio 倍高度。"""
    yc = SOLE_Y - BR
    t_fall = np.sqrt(2 * fall_px / GPX)
    v_up = np.sqrt(2 * GPX * ratio * fall_px)
    out, f = {}, 0
    for t in np.arange(0, t_fall, 1 / 30):
        out[f] = Cand(x, yc - fall_px + 0.5 * GPX * t * t, BR, 0.9)
        f += 1
    for k in range(1, n_up + 1):
        t = k / 30
        out[f] = Cand(x, yc - (v_up * t - 0.5 * GPX * t * t), BR, 0.9)
        f += 1
    return out


def test_passive_bounce_away_from_body_not_counted():
    """球在人前面（离脚 10 个球半径）落地、弹回一半高：没碰到人，不算触球（以前算成触球：地面线在脚底，球底正好贴地）。"""
    tr = _bounce_track(980.0, 600.0, 0.45)
    kpts, persons = _standing(max(tr) + 1)
    ev = analyze_track(tr, 30.0, max(tr) + 1, persons, 1920, kpts=kpts)
    assert not ev.touches and len(ev.bounces) == 1


def test_foot_touch_next_to_foot_still_counted():
    """同样高度落到脚边、被脚弹得比落下还高：是脚触球，照样算。"""
    tr = _bounce_track(540.0, 600.0, 1.2)
    kpts, persons = _standing(max(tr) + 1)
    ev = analyze_track(tr, 30.0, max(tr) + 1, persons, 1920, kpts=kpts)
    assert len(ev.touches) == 1 and ev.touches[0]["part"] == "foot"


# ---------- 2026-10-12：放大复检的窗口（analyze.zoom_window） ----------
def test_zoom_window():
    from combocam.analyze import CROP, zoom_window
    # 竖版画布上的全身人（高 1200）：窗口边长 1.3 × 1200 = 1560，下边在脚下一点，正方形、不出画面
    x0, y0, x1, y1 = zoom_window((400.0, 500.0, 700.0, 1700.0), 1080, 0.0, 1920.0)
    assert x1 - x0 == y1 - y0 == 1080  # 被画面宽度限制
    assert y1 >= 1700 and y0 >= 0 and x0 >= 0 and x1 <= 1080
    # 横屏放进竖版：只在真实画面（656–1264，高 608）里；人小时窗口取 CROP，但不超过真实画面的高
    x0, y0, x1, y1 = zoom_window((500.0, 900.0, 560.0, 1100.0), 1080, 656.0, 1264.0)
    assert x1 - x0 == y1 - y0 == min(CROP, 608) and 656 <= y0 and y1 <= 1264 and x0 <= 500 and x1 >= 560
