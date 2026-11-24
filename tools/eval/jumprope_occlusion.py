"""遮挡压力测试：在标注过的视频的关键点上模拟遮挡，看计数掉多少。

  python tools/eval/jumprope_occlusion.py [--dir out/jumprope] [--seeds 10]

几种遮挡（每种按 0.4–1.2 秒一块、合计约占 30% 的时间，随机放）：
  feet   脚和脚踝的关键点丢了（被前面的人、器材挡住，深色鞋踩深色地）
  legs   膝盖以下全丢
  gone   整个人这几帧没检测到（有人从镜头前走过）
  jitter 身体某一处（髋部、肩膀或头，每段视频随机挑一处）的关键点乱跳（身体被挡住一部分，姿态模型猜错）
对比"旧规则"（只看髋部、两只脚都离地、不补漏掉的跳、人不在画面里也照样断开）和现在的规则。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from combocam.jumprope import count_jumps  # noqa: E402
from combocam.pose import (HEAD, L_ANKLE, L_BIG_TOE, L_HEEL, L_HIP, L_KNEE, L_SHOULDER, L_SMALL_TOE, NOSE, R_ANKLE, R_BIG_TOE,  # noqa: E402
                         R_HEEL, R_HIP, R_KNEE, R_SHOULDER, R_SMALL_TOE)
from jumprope_run import match  # noqa: E402

FEET = [L_ANKLE, R_ANKLE, L_BIG_TOE, R_BIG_TOE, L_SMALL_TOE, R_SMALL_TOE, L_HEEL, R_HEEL]
LEGS = FEET + [L_KNEE, R_KNEE]
OLD = {"feet_rule": "both", "missed": (99, 99), "signal": "hip", "absent_max_s": 0}


def blocks(n: int, fps: float, rng, share: float = 0.3) -> np.ndarray:
    m = np.zeros(n, bool)
    while m.mean() < share:
        ln = int(rng.uniform(0.4, 1.2) * fps)
        a = int(rng.integers(0, max(1, n - ln)))
        m[a:a + ln] = True
    return m


def occlude(kpts: list, kind: str, fps: float, rng) -> tuple[list, np.ndarray]:
    m = blocks(len(kpts), fps, rng)
    part = [[L_HIP, R_HIP], [L_SHOULDER, R_SHOULDER], [HEAD, NOSE]][int(rng.integers(0, 3))]
    out = []
    for f, k in enumerate(kpts):
        if not k or not m[f]:
            out.append(k)
            continue
        if kind == "gone":
            out.append(None)
            continue
        a = np.asarray(k, np.float64).reshape(-1, 3).copy()
        if kind in ("feet", "legs"):
            a[FEET if kind == "feet" else LEGS, 2] = 0
        elif kind == "jitter":
            tall = np.nanmax(a[:, 1]) - np.nanmin(a[:, 1])
            a[part, 1] += rng.normal(0, 0.02 * tall)
        out.append([int(round(v)) for v in a.ravel()])
    return out, m


def main() -> None:
    a = sys.argv[1:]
    d_dir = Path(a[a.index("--dir") + 1]) if "--dir" in a else ROOT / "out" / "jumprope"
    seeds = int(a[a.index("--seeds") + 1]) if "--seeds" in a else 10
    labels = json.loads((ROOT / "samples" / "jumprope" / "labels.json").read_text(encoding="utf-8"))
    data = {n: json.loads((d_dir / f"{n}.pose.json").read_text(encoding="utf-8")) for n in labels}
    for kind in ("none", "feet", "legs", "gone", "jitter"):
        for rule_name, params in (("旧规则", OLD), ("现在", None)):
            th = tp = fp = tv = tpv = 0
            for seed in range(seeds if kind != "none" else 1):
                rng = np.random.default_rng(seed)
                for name, lab in labels.items():
                    d = data[name]
                    kp, hidden = (d["kpts"], np.zeros(len(d["kpts"]), bool)) if kind == "none" else occlude(d["kpts"], kind, d["fps"], rng)
                    r = count_jumps(kp, d["fps"], params, frame_h=d["height"])
                    ign = lab.get("ignore", [])
                    inside = lambda f: any(x <= f <= y for x, y in ign)  # noqa: E731
                    pred = [j["frame"] for j in r["jumps"] if not inside(j["frame"])]
                    truth = [f for f in lab["jumps"] if not inside(f)]
                    tol = max(4, round(0.2 * (r.get("period_s") or 0) * d["fps"]))
                    hit, miss, extra = match(pred, truth, tol)
                    th, tp, fp = th + len(truth), tp + hit, fp + len(extra)
                    # 看得见的跳：最高点前后 3 帧都没被遮挡
                    seen = [f for f in truth if not hidden[max(0, f - 3):f + 4].any()]
                    tv += len(seen)
                    tpv += sum(1 for f in seen if f not in miss)
            print(f"{kind:7s} {rule_name}：召回 {tp / th:6.1%}（看得见的跳 {tpv / max(1, tv):6.1%}）  精确率 {tp / max(1, tp + fp):6.1%}  （{th} 跳）")


if __name__ == "__main__":
    main()
