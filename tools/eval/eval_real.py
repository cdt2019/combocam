"""在人工标注的真实视频上评估触球计数。

  python tools/eval/eval_real.py                 # 重新跑检测（结果缓存到 out/<名>.debug.json）
  python tools/eval/eval_real.py --cached        # 只重跑轨迹和事件逻辑
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from combocam.analyze import analyze, postprocess  # noqa: E402
from combocam.video import content_bottom  # noqa: E402
from combocam.track import Cand  # noqa: E402

TOL = 3  # 帧


def score(pred: list[float], truth: list[int], amb: list[int]):
    used, tp, fp, amb_hit = set(), 0, 0, 0
    for p in sorted(pred):
        cands = [(abs(p - t), i) for i, t in enumerate(truth) if i not in used and abs(p - t) <= TOL]
        if cands:
            used.add(min(cands)[1])
            tp += 1
        elif any(abs(p - a) <= TOL for a in amb):
            amb_hit += 1
        else:
            fp += 1
    return tp, fp, len(truth) - tp, amb_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--refine", default="online")
    a = ap.parse_args()
    labels = json.loads((ROOT / "samples" / "labels.json").read_text(encoding="utf-8"))
    tot = [0, 0, 0, 0, 0]
    rows = []
    for name, lab in labels.items():
        if name.startswith("_"):
            continue
        stem = Path(name).stem
        dbg_path = ROOT / "out" / f"{stem}.debug.json"
        if a.cached and dbg_path.exists():
            dbg = json.loads(dbg_path.read_text(encoding="utf-8"))
            meta = json.loads((ROOT / "out" / f"{stem}.track.json").read_text(encoding="utf-8"))
            fps, height, timing = meta["fps"], meta["height"], meta["timing"]
        else:
            dbg = {}
            tr = analyze(str(ROOT / "samples" / name), model=str(ROOT / "models" / "yolox_s.onnx"), refine=a.refine, debug=dbg, log=lambda *_: None)
            dbg_path.write_text(json.dumps(dbg), encoding="utf-8")
            (ROOT / "out" / f"{stem}.track.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
            fps, height, timing = tr["fps"], tr["height"], tr["timing"]
        cands = [[Cand(*c) for c in fr] for fr in dbg["cands"]]
        persons = [tuple(p) if p else None for p in dbg["persons"]]
        pr = meta["source"]["probe"] if a.cached and dbg_path.exists() else tr["source"]["probe"]
        _, _, ev = postprocess(cands, persons, fps, height, content_bottom=content_bottom(pr["width"], pr["height"], 1080, height), kpts=dbg.get("kpts"))
        pred = [t["t"] * fps for t in ev.touches]
        tp, fp, fn, amb = score(pred, lab["touches"], lab["ambiguous"])
        btp = score([b["t"] * fps for b in ev.bounces], lab["bounces"], [])[0]
        tot = [tot[0] + len(lab["touches"]), tot[1] + tp, tot[2] + fp, tot[3] + fn, tot[4] + amb]
        rows.append((stem, len(lab["touches"]), tp, fp, fn, amb, len(lab["ambiguous"]), [c["count"] for c in ev.combos],
                     f"{btp}/{len(lab['bounces'])}", timing.get("total_s")))
    print(f"{'视频':<18}{'真值':>5}{'命中':>5}{'误报':>5}{'漏检':>5}{'模糊处计入':>10}  连击          反弹识别  分析用时s")
    for r in rows:
        print(f"{r[0]:<18}{r[1]:>5}{r[2]:>5}{r[3]:>5}{r[4]:>5}{f'{r[5]}/{r[6]}':>10}  {str(r[7]):<14}{r[8]:>6}  {r[9]}")
    print(f"合计：真值 {tot[0]}，命中 {tot[1]}（召回 {tot[1] / tot[0]:.1%}），误报 {tot[2]}，漏检 {tot[3]}，模糊处计入 {tot[4]}")


if __name__ == "__main__":
    main()
