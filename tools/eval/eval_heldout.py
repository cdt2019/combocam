"""留出测试集评估：samples/heldout/labels.json（人工标注）对比当前计数逻辑的结果。

  python tools/eval/eval_heldout.py            # 用 out/heldout/<名>.debug.json 缓存的检测结果重算事件（不跑检测）
  python tools/eval/eval_heldout.py -v         # 逐条列出误报、漏检
  python tools/eval/eval_heldout.py --set heldout2 [--no-pose]   # 换测试集；--no-pose 不用关键点（对比旧规则）

规则：预测触球与真值相差 ≤ TOL 帧算命中（慢动作素材一次触球要持续好几帧，人工标的时刻也有 1–3 帧误差）；
ignore 时段里的预测不计分；落在 ambiguous 附近的预测不算误报；nocount（手接、抛）附近的预测算误报并单独统计。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SET = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "heldout"  # 测试集目录：samples/<SET>/、out/<SET>/
sys.path.insert(0, str(ROOT))

from combocam.analyze import postprocess  # noqa: E402
from combocam.track import Cand  # noqa: E402
from combocam.video import content_bottom  # noqa: E402

TOL = 4


def match(pred: list[float], truth: list[int]) -> tuple[list[tuple[float, int]], list[float], list[int]]:
    pairs = sorted(((abs(p - t), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth) if abs(p - t) <= TOL))
    up, ut, hits = set(), set(), []
    for _, i, j in pairs:
        if i not in up and j not in ut:
            up.add(i)
            ut.add(j)
            hits.append((pred[i], truth[j]))
    return hits, [p for i, p in enumerate(pred) if i not in up], [t for j, t in enumerate(truth) if j not in ut]


def main() -> None:
    verbose = "-v" in sys.argv
    labels = json.loads((ROOT / "samples" / SET / "labels.json").read_text(encoding="utf-8"))
    tot = dict(truth=0, tp=0, fp=0, fn=0, hand=0)
    print(f"{'视频':<17}{'真值':>4}{'命中':>5}{'误报':>5}{'漏检':>5}{'手':>4}  预测连击              真值落地/预测落地")
    for name, lab in labels.items():
        if name.startswith("_"):
            continue
        stem = Path(name).stem
        tr = json.loads((ROOT / "out" / SET / f"{stem}.track.json").read_text(encoding="utf-8"))
        dbg = json.loads((ROOT / "out" / SET / f"{stem}.debug.json").read_text(encoding="utf-8"))
        cands = [[Cand(*c) for c in fr] for fr in dbg["cands"]]
        persons = [tuple(p) if p else None for p in dbg["persons"]]
        _, _, ev = postprocess(cands, persons, tr["fps"], tr["height"], dbg.get("sel"), content_bottom=content_bottom(tr["source"]["probe"]["width"], tr["source"]["probe"]["height"], tr["width"], tr["height"]), kpts=None if "--no-pose" in sys.argv else dbg.get("kpts"))
        fps = tr["fps"]
        ign = lab.get("ignore", [])

        def inside(f, ign=ign):  # 不放宽：放宽会把时段边上的真触球也吞掉
            return any(a <= f <= b for a, b in ign)

        pred = [t["t"] * fps for t in ev.touches if not inside(t["t"] * fps)]
        truth = [t for t in lab["touches"] if not inside(t)]
        hits, fps_, fns = match(pred, truth)
        # 看不清的地方：算不算都行
        fps_ = [p for p in fps_ if not any(abs(p - a) <= TOL for a in lab.get("ambiguous", []))]
        hand = [p for p in fps_ if any(abs(p - a) <= 2 * TOL for a in lab.get("nocount", []))]
        tot["truth"] += len(truth)
        tot["tp"] += len(hits)
        tot["fp"] += len(fps_)
        tot["fn"] += len(fns)
        tot["hand"] += len(hand)
        combos = [c["count"] for c in ev.combos]
        print(f"{stem:<17}{len(truth):>4}{len(hits):>5}{len(fps_):>5}{len(fns):>5}{len(hand):>4}  {str(combos):<22}"
              f"{len(lab.get('bounces', []))}/{len(ev.bounces)}")
        if verbose and (fps_ or fns):
            print(f"    误报 {[round(p) for p in fps_]}  漏检 {fns}  预测落地 {[round(b['t'] * fps) for b in ev.bounces]}")
    t = tot
    prec = t["tp"] / max(1, t["tp"] + t["fp"])
    print(f"合计：真值 {t['truth']}，命中 {t['tp']}（召回 {t['tp'] / max(1, t['truth']):.1%}），误报 {t['fp']}（其中手接/抛 {t['hand']}），"
          f"漏检 {t['fn']}，精确率 {prec:.1%}")


if __name__ == "__main__":
    main()
