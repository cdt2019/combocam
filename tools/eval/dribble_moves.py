"""运球动作（单手 S / 交叉 C / 胯下 L / 背后 B）评估：对缓存计数、分动作，和 samples/dribble/moves.json 的人工标注比。

  python tools/eval/dribble_moves.py [--set tune|val|all] [-v]

只比对上了的触地（±3 帧）；标注是 "?" 的、算法没数到的不算。打印混淆矩阵（行 = 真值，列 = 算法，"-" = 算法没给动作）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dribble_run import run  # noqa: E402

CODE = {"same": "S", "cross": "C", "legs": "L", "behind": "B", None: "-"}
TOL = 3


def evaluate(sets: list[str], verbose: bool = False) -> dict:
    lab = json.loads((ROOT / "samples" / "dribble" / "moves.json").read_text(encoding="utf-8"))
    conf: dict = {}
    for st in sets:
        for name, marks in lab[st].items():
            d = json.loads((ROOT / "out" / "dribble" / f"{name}.ball.json").read_text(encoding="utf-8"))
            res = run(d, {"min_group": 1})
            got = {h["frame"]: h for h in res["dribbles"]}
            line = []
            for fs, t in marks.items():
                if t == "?":
                    continue
                f = int(fs)
                near = [g for g in got if abs(g - f) <= TOL]
                if not near:
                    line.append(f"{f}:{t}→没数到")
                    continue
                h = got[min(near, key=lambda g: abs(g - f))]
                p = CODE[h.get("move")]
                conf[(t, p)] = conf.get((t, p), 0) + 1
                if p != t:
                    line.append(f"{f}:{t}→{p}({h.get('hand_from')}{h.get('hand_to')})")
            if verbose:
                print(f"{name} [{res.get('view')}] " + ("  ".join(line) if line else "全对"))
    return conf


def main() -> None:
    a = sys.argv[1:]
    st = a[a.index("--set") + 1] if "--set" in a else "all"
    sets = ["tune", "val"] if st == "all" else [st]
    conf = evaluate(sets, "-v" in a)
    rows, cols = "SCLB", "SCLB-"
    print("真值\\算法 " + " ".join(f"{c:>4}" for c in cols) + "   对")
    tot = ok = 0
    for t in rows:
        cnt = [conf.get((t, c), 0) for c in cols]
        n = sum(cnt)
        tot, ok = tot + n, ok + conf.get((t, t), 0)
        print(f"{t:>8} " + " ".join(f"{v:>4}" for v in cnt) + f"   {conf.get((t, t), 0)}/{n}")
    print(f"合计 {ok}/{tot} = {ok / max(1, tot):.1%}")


if __name__ == "__main__":
    main()
