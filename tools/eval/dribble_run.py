"""篮球运球原型：对缓存的球轨迹计数，打印结果；有标注（samples/dribble/labels.json）就算准确率。

  python tools/eval/dribble_run.py [名 ...] [-v] [--dir out/dribble] [--labels samples/dribble/labels.json] [--tracks 目录]

--tracks：顺便把渲染用的轨迹 JSON（combocam/dribble_track.build_track）写到 <目录>/<名>.track.json，
          接着 node render/render.mjs --input samples/dribble/<名>.mp4 --track <目录>/<名>.track.json --output ... 出片。

labels.json：{"pexels_xxx": {"bounces": [球触地的帧号, ...], "ignore": [[起, 止], ...]}}；匹配容差 TOL 帧。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jumprope_run import match  # noqa: E402

from combocam.dribble import ball_series, count_dribbles  # noqa: E402
from combocam.dribble_track import build_track  # noqa: E402

TOL = 3


def run(d: dict, params: dict | None = None) -> dict:
    x, y, r, r_ref = ball_series(d["cands"], d["fps"], d.get("persons"))
    return count_dribbles(x, y, r, r_ref, d["fps"], d.get("persons"), d.get("kpts"), params, d.get("content_bottom"), d.get("width"))


def main() -> None:
    argv = sys.argv[1:]
    opt = {}
    for k in ("--dir", "--labels", "--tracks"):
        if k in argv:
            i = argv.index(k)
            opt[k] = Path(argv[i + 1])
            del argv[i:i + 2]
    out = opt.get("--dir", ROOT / "out" / "dribble")
    lab_path = opt.get("--labels", ROOT / "samples" / "dribble" / "labels.json")
    verbose = "-v" in argv
    names = [a for a in argv if not a.startswith("-")]
    labels = json.loads(lab_path.read_text(encoding="utf-8")) if lab_path.exists() else {}
    labels = {k: v for k, v in labels.items() if k.startswith("pexels_")}
    caches = [out / f"{n}.ball.json" for n in names] if names else sorted(out.glob("*.ball.json"))
    tracks = opt.get("--tracks")
    if tracks:
        tracks.mkdir(parents=True, exist_ok=True)
    th = tp = fp = exact = nl = 0
    for c in caches:
        name = c.name.replace(".ball.json", "")
        d = json.loads(c.read_text(encoding="utf-8"))
        r = run(d)
        if tracks:
            src = next((p / f"{name}.mp4" for p in (ROOT / "samples" / "dribble", ROOT / "samples" / "dribble2", ROOT / "samples" / "dribble3") if (p / f"{name}.mp4").exists()), None)
            tr = build_track(d, str(src) if src else None)
            (tracks / f"{name}.track.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
        groups = "、".join(f"{g['count']}（{g['rate']} 次/分）" for g in r["groups"]) or "无"
        line = (f"{name}: {d['frames']} 帧，球 {r['ball_frames']} 帧，半径 {r['r_ref']}，数到 {r['total']} 下"
                f"（漏检补 {sum(1 for h in r['dribbles'] if h['inferred'])}），分组 {groups}")
        lab = labels.get(name)
        if lab:
            ign = lab.get("ignore", [])
            inside = lambda f: any(a <= f <= b for a, b in ign)  # noqa: E731
            pred = [h["frame"] for h in r["dribbles"] if not inside(h["frame"])]
            truth = [f for f in lab["bounces"] if not inside(f)]
            hit, miss, extra = match(pred, truth, TOL)
            th, tp, fp, nl = th + len(truth), tp + hit, fp + len(extra), nl + 1
            exact += int(len(pred) == len(truth))
            line += f"；真值 {len(truth)}，命中 {hit}，漏 {len(miss)}，多 {len(extra)}"
            if verbose and (miss or extra):
                line += f"\n    漏：{miss}\n    多：{extra}"
        elif verbose:
            line += f"\n    触地：{[h['frame'] for h in r['dribbles']]}"
        print(line)
    if th:
        print(f"\n合计 {nl} 段：真值 {th} 下，召回 {tp / th:.1%}，精确率 {tp / max(1, tp + fp):.1%}，计数完全对的 {exact} 段")


if __name__ == "__main__":
    main()
