"""跳绳原型：对缓存的关键点跑计数，打印结果；有标注（samples/jumprope/labels.json）就算准确率。

  python tools/eval/jumprope_run.py [名 ...] [-v] [--dir 关键点缓存目录，默认 out/jumprope] [--labels 标注文件，默认 samples/jumprope/labels.json]

第二批（验证集，规则调完以后才标）：--dir out/jumprope2_mp --labels samples/jumprope2/labels.json

labels.json：{"pexels_xxx": {"jumps": [帧号, ...], "ignore": [[起, 止], ...]}}，jumps 是每跳最高点（人在空中）的帧号。
匹配容差 TOL 帧；ignore 区间里的算法结果和真值都不计。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from combocam.jumprope import count_jumps  # noqa: E402
from combocam.jumprope_track import main_person  # noqa: E402

OUT = ROOT / "out" / "jumprope"
LABELS = ROOT / "samples" / "jumprope" / "labels.json"
TOL = 4


def match(pred: list[int], truth: list[int], tol: int = TOL) -> tuple[int, list[int], list[int]]:
    """贪心一对一匹配：返回命中数、漏掉的真值、多出来的预测。"""
    used, hit, miss = set(), 0, []
    for t in truth:
        best = min((p for p in pred if p not in used and abs(p - t) <= tol), key=lambda p: abs(p - t), default=None)
        if best is None:
            miss.append(t)
        else:
            used.add(best)
            hit += 1
    return hit, miss, [p for p in pred if p not in used]


def main() -> None:
    argv = sys.argv[1:]
    out = OUT
    if "--dir" in argv:
        i = argv.index("--dir")
        out = Path(argv[i + 1])
        del argv[i:i + 2]
    labels_path = LABELS
    if "--labels" in argv:
        i = argv.index("--labels")
        labels_path = Path(argv[i + 1])
        del argv[i:i + 2]
    names = [a for a in argv if not a.startswith("-")]
    verbose = "-v" in argv
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    labels = {k: v for k, v in labels.items() if k.startswith("pexels_")}  # 去掉说明、exclude
    caches = [out / f"{n}.pose.json" for n in names] if names else sorted(out.glob("*.pose.json"))
    th = tp = fp = 0
    for c in caches:
        name = c.name.replace(".pose.json", "")
        d = json.loads(c.read_text(encoding="utf-8"))
        _, kpts, tracks = main_person(d)  # 多人入镜时挑在跳绳的那个人（见 combocam/people.py）
        r = count_jumps(kpts, d["fps"], frame_h=d["height"])
        groups = "、".join(f"{g['count']}（{g['bpm']} 次/分）" for g in r["groups"]) or "无"
        line = f"{name}: {d['frames']} 帧，身高 {r['body_px']} px，数到 {r.get('total', 0)} 跳，分组 {groups}"
        if r.get("warning"):
            line += f"；{r['warning']}"
        if len(tracks) > 1:
            line += "；多人：" + "、".join(f"{t['jumps']} 跳（高 {t['height']:.0f}，x {t['x']:.0f}，帧 {t['from']}–{t['to']}）" for t in tracks[:4])
        lab = labels.get(name)
        if lab:
            ign = lab.get("ignore", [])
            inside = lambda f: any(a <= f <= b for a, b in ign)  # noqa: E731
            pred = [j["frame"] for j in r["jumps"] if not inside(j["frame"])]
            truth = [f for f in lab["jumps"] if not inside(f)]
            # 容差按节奏缩放：慢动作视频一跳几十帧，髋部最高点和脚的最高点能差好几帧
            tol = max(TOL, round(0.2 * (r.get("period_s") or 0) * d["fps"]))
            hit, miss, extra = match(pred, truth, tol)
            th += len(truth)
            tp += hit
            fp += len(extra)
            line += f"；真值 {len(truth)}，命中 {hit}，漏 {len(miss)}，多 {len(extra)}"
            if verbose and (miss or extra):
                line += f"\n    漏：{miss}\n    多：{extra}"
        print(line)
    if th:
        print(f"\n合计：真值 {th} 跳，召回 {tp / th:.1%}，精确率 {tp / max(1, tp + fp):.1%}")


if __name__ == "__main__":
    main()
