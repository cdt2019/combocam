"""生成带真值的合成颠球轨迹（和视频）。tests/test_events.py 用这里的轨迹函数测计数逻辑。

注意：画出来的视频里的球太"假"，YOLOX 基本认不出来，不能用来跑整条流水线或测检测准确率，只用于看画面和调试。

画面：竖版 1080×1920，人形剪影站在中间，足球按真实重力运动；
场边放一个静止的干扰球；两段连击之间球落地反弹两次再被挑起。
输出 <out>.mp4 和 <out>.truth.json。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1080, 1920, 30
PX_PER_M = 590.0
G = 9.81 * PX_PER_M
R = 0.11 * PX_PER_M  # 球半径（像素）
GROUND = 1560.0
CX = 540.0


class Flight:
    def __init__(self, t0, x0, y0, vx, vy, t1):
        self.t0, self.x0, self.y0, self.vx, self.vy, self.t1 = t0, x0, y0, vx, vy, t1

    def at(self, t):
        d = t - self.t0
        return self.x0 + self.vx * d, self.y0 + self.vy * d + 0.5 * G * d * d


def flight_between(t0, p0, p1, h_m):
    """从 p0 以"最高点比 p0 高 h_m 米"的速度起跳，落到 p1 的高度。"""
    h_m = max(h_m, (p0[1] - p1[1]) / PX_PER_M + 0.2)  # 最高点至少比落点高 20 cm
    vy = -math.sqrt(2 * G * h_m * PX_PER_M)
    # y0 + vy d + 0.5 G d^2 = y1
    a, b, c = 0.5 * G, vy, p0[1] - p1[1]
    d = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return Flight(t0, p0[0], p0[1], (p1[0] - p0[0]) / d, vy, t0 + d)


def build_plan(seed: int = 7):
    rng = np.random.default_rng(seed)
    flights, touches, bounces = [], [], []
    rest_y = GROUND - R
    t = 1.0
    x_rest = CX + 60

    def juggle(t, start, n_touch, combo):
        p = start
        for k in range(n_touch):
            part = "foot"
            roll = rng.random()
            if k % 5 == 3:
                part, hy = "thigh", 0.62
            elif k == n_touch - 3 and combo == 0:
                part, hy = "head", 1.72
            else:
                hy = 0.28 + rng.random() * 0.08
            side = -1 if k % 2 == 0 else 1
            q = (CX + side * (30 + 20 * roll), GROUND - hy * PX_PER_M)
            h = 0.5 + rng.random() * 0.9 if part != "head" else 0.35
            if p is start:
                h = 0.55  # 挑球
            fl = flight_between(t, p, q, h)
            flights.append(fl)
            t = fl.t1
            touches.append({"t": round(t, 4), "x": round(q[0], 1), "y": round(q[1], 1), "part": part, "combo": combo})
            p = q
        return t, p

    # 第一段：挑起 → 14 次触球 → 落地弹两次 → 滚停
    t, p = juggle(t, (x_rest, rest_y), 14, 0)
    fl = flight_between(t, p, (p[0] + 160, rest_y), 0.9)
    flights.append(fl)
    t = fl.t1
    vx, vy = fl.vx * 0.7, -(fl.vy + G * (fl.t1 - fl.t0)) * 0.55
    x = p[0] + 160
    for _ in range(2):
        bounces.append({"t": round(t, 4), "x": round(x, 1), "y": round(rest_y, 1)})
        d = 2 * -vy / G
        flights.append(Flight(t, x, rest_y, vx, vy, t + d))
        x += vx * d
        t += d
        vy *= 0.5
        vx *= 0.8
    bounces.append({"t": round(t, 4), "x": round(x, 1), "y": round(rest_y, 1)})
    # 滚 0.6 s 停住，再静止 1.2 s
    flights.append(Flight(t, x, rest_y, vx * 0.5, 0.0, t + 0.6))
    x += vx * 0.5 * 0.6
    t += 0.6
    flights.append(Flight(t, x, rest_y, 0.0, 0.0, t + 1.2))
    t += 1.2
    # 第二段：挑起 → 8 次触球 → 落地
    t, p = juggle(t, (x, rest_y), 8, 1)
    fl = flight_between(t, p, (p[0] - 120, rest_y), 0.7)
    flights.append(fl)
    t = fl.t1
    bounces.append({"t": round(t, 4), "x": round(p[0] - 120, 1), "y": round(rest_y, 1)})
    flights.append(Flight(t, p[0] - 120, rest_y, -60.0, 0.0, t + 1.5))
    t += 1.5
    return flights, touches, bounces, t


def ball_at(flights, t):
    for fl in flights:
        if fl.t0 <= t <= fl.t1:
            x, y = fl.at(t)
            return x, min(y, GROUND - R)
    fl = flights[0] if t < flights[0].t0 else flights[-1]
    x, y = fl.at(min(max(t, fl.t0), fl.t1))
    return x, min(y, GROUND - R)


def draw_ball(img, x, y, ang, r=R):
    xi, yi, ri = int(round(x)), int(round(y)), int(round(r))
    cv2.circle(img, (xi, yi), ri, (236, 238, 240), -1, cv2.LINE_AA)
    for i in range(5):
        a = ang + i * 2 * math.pi / 5
        px, py = x + math.cos(a) * r * 0.72, y + math.sin(a) * r * 0.66
        pts = np.array([[px + math.cos(a + k * 2 * math.pi / 5) * r * 0.26, py + math.sin(a + k * 2 * math.pi / 5) * r * 0.26] for k in range(5)])
        cv2.fillPoly(img, [pts.astype(np.int32)], (30, 30, 34), cv2.LINE_AA)
    pts = np.array([[x + math.cos(ang * 0.5 + k * 2 * math.pi / 5) * r * 0.3, y + math.sin(ang * 0.5 + k * 2 * math.pi / 5) * r * 0.3] for k in range(5)])
    cv2.fillPoly(img, [pts.astype(np.int32)], (30, 30, 34), cv2.LINE_AA)
    # 明暗
    yy, xx = np.ogrid[-ri:ri + 1, -ri:ri + 1]
    m = xx * xx + yy * yy <= ri * ri
    shade = np.clip(1.15 - 0.55 * ((xx + 0.35 * ri) ** 2 + (yy + 0.4 * ri) ** 2) ** 0.5 / ri, 0.45, 1.15)
    y0, y1, x0, x1 = yi - ri, yi + ri + 1, xi - ri, xi + ri + 1
    if y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1]:
        return
    roi = img[y0:y1, x0:x1].astype(np.float32)
    roi[m] *= shade[m][:, None]
    img[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def draw_background(rng):
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        if y < GROUND - 260:
            c = np.array([120, 110, 95]) + (y / H) * np.array([40, 40, 30])  # 墙
        else:
            k = (y - GROUND + 260) / (H - GROUND + 260)
            c = np.array([60, 140, 70]) * (1.0 - 0.4 * k)  # 草地
        img[y] = c
    cv2.line(img, (0, int(GROUND - 260)), (W, int(GROUND - 260)), (200, 205, 210), 6)
    noise = rng.normal(0, 6, (H, W, 1))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def draw_person(img, t, touch_near):
    skin, shirt, pants, shoe = (140, 170, 215), (40, 40, 200), (50, 45, 40), (240, 240, 240)
    hip = (int(CX), int(GROUND - 0.92 * PX_PER_M))
    sh_y = int(GROUND - 1.42 * PX_PER_M)
    # 躯干、头
    cv2.rectangle(img, (int(CX - 95), sh_y), (int(CX + 95), hip[1] + 20), shirt, -1)
    cv2.ellipse(img, (int(CX), sh_y - 115), (68, 85), 0, 0, 360, skin, -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(CX), sh_y - 150), (72, 50), 0, 180, 360, (30, 30, 35), -1, cv2.LINE_AA)
    cv2.rectangle(img, (int(CX - 28), sh_y - 40), (int(CX + 28), sh_y + 5), skin, -1)
    # 手臂
    for s in (-1, 1):
        cv2.line(img, (int(CX + s * 90), sh_y + 20), (int(CX + s * 165), sh_y + 260), shirt, 52, cv2.LINE_AA)
        cv2.line(img, (int(CX + s * 165), sh_y + 260), (int(CX + s * 195), sh_y + 470), skin, 40, cv2.LINE_AA)
    # 腿：触球那条腿抬向球
    for s in (-1, 1):
        hipp = (int(CX + s * 45), hip[1])
        foot = (int(CX + s * 75), int(GROUND - 25))
        knee = (int(CX + s * 60), int(GROUND - 0.48 * PX_PER_M))
        if touch_near and touch_near[0] == s:
            k, tx, ty = touch_near[1], touch_near[2], touch_near[3]
            foot = (int(foot[0] + (tx - foot[0]) * k), int(foot[1] + (ty + R - foot[1]) * k))
            knee = (int(knee[0] + (tx - knee[0]) * 0.4 * k), int(knee[1] - 120 * k))
        cv2.line(img, hipp, knee, pants, 70, cv2.LINE_AA)
        cv2.line(img, knee, foot, skin, 48, cv2.LINE_AA)
        cv2.line(img, foot, (foot[0] + s * 55, foot[1] + 8), shoe, 40, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="输出路径（不带扩展名）")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    flights, touches, bounces, t_end = build_plan(args.seed)
    rng = np.random.default_rng(args.seed)
    bg = draw_background(rng)
    draw_ball(bg, 150, GROUND + 140 - R, 0.3)  # 场边静止的干扰球（在地面线前方，离人较远）
    n = int(math.ceil(t_end * FPS))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(out.with_suffix(".mp4"))],
        stdin=subprocess.PIPE,
    )
    for i in range(n):
        t = i / FPS
        img = bg.copy()
        near = None
        for tc in touches:
            d = abs(t - tc["t"])
            if d < 0.18 and tc["part"] in ("foot", "thigh"):
                k = 1 - d / 0.18
                side = -1 if tc["x"] < CX else 1
                if near is None or k > near[1]:
                    near = (side, k * k * (3 - 2 * k), tc["x"], tc["y"])
        # 运动模糊：快门内 4 个子帧平均
        draw_person(img, t, near)
        acc = np.zeros((H, W, 3), np.float32)
        for s in range(4):
            sub = img.copy()
            ts = t + s / FPS / 8
            bx, by = ball_at(flights, ts)
            draw_ball(sub, bx, by, ts * 9)
            acc += sub
        frame = (acc / 4).astype(np.uint8)
        frame = np.clip(frame.astype(np.int16) + rng.normal(0, 3, frame.shape).astype(np.int16), 0, 255).astype(np.uint8)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    truth = {
        "fps": FPS, "width": W, "height": H, "frames": n, "px_per_m": PX_PER_M, "ground_y": GROUND, "ball_radius_px": R,
        "touches": touches, "bounces": bounces,
        "combos": [sum(1 for t in touches if t["combo"] == 0), sum(1 for t in touches if t["combo"] == 1)],
    }
    out.with_suffix(".truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out.with_suffix('.mp4')} ({n} frames, {t_end:.1f}s), touches={len(touches)} combos={truth['combos']} bounces={len(bounces)}", file=sys.stderr)


if __name__ == "__main__":
    main()
