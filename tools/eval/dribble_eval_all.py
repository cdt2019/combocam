"""运球：所有标注集一起评估（对缓存的球轨迹计数，和人工标的触地帧比，±3 帧算对上）。

  python tools/eval/dribble_eval_all.py [--dir out/dribble] [-v]

标注集：samples/dribble/labels.json（调参集）、labels_val.json（验证集）、labels_t3.json、labels_t4.json、samples/dribble3/labels.json（独立检验）。
缓存要先用 tools/eval/dribble_extract.py 跑出来（<dir>/<名>.ball.json）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dribble_run import run  # noqa: E402
from jumprope_run import match  # noqa: E402

SETS = ["dribble/labels", "dribble/labels_val", "dribble/labels_t3", "dribble/labels_t4", "dribble3/labels"]


def main() -> None:
    a = sys.argv[1:]
    cache = Path(a[a.index("--dir") + 1]) if "--dir" in a else ROOT / "out" / "dribble"
    verbose = "-v" in a
    for name in SETS:
        lab = json.loads((ROOT / "samples" / f"{name}.json").read_text(encoding="utf-8"))
        th = tp = fp = exact = 0
        bad = []
        for clip, v in lab.items():
            if not clip.startswith("pexels_"):
                continue
            f = cache / f"{clip}.ball.json"
            if not f.exists():
                print(f"  缺缓存 {f}")
                continue
            r = run(json.loads(f.read_text(encoding="utf-8")))
            ign = v.get("ignore", [])

            def inside(fr: int) -> bool:
                return any(lo <= fr <= hi for lo, hi in ign)

            pred = [h["frame"] for h in r["dribbles"] if not inside(h["frame"])]
            truth = [fr for fr in v["bounces"] if not inside(fr)]
            hit, miss, extra = match(pred, truth, 3)
            th += len(truth)
            tp += hit
            fp += len(extra)
            exact += len(pred) == len(truth)
            if miss or extra:
                bad.append(f"{clip[7:]} 漏{miss} 多{extra}")
        print(f"{name:20s} 真值 {th:3d}  召回 {tp / max(1, th):.1%}  精确率 {tp / max(1, tp + fp):.1%}  计数全对 {exact}")
        if verbose and bad:
            print("    " + " | ".join(bad))


if __name__ == "__main__":
    main()
