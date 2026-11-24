"""篮球运球原型：跑颠球的分析管线（整帧检测人和球、在线复检、选球的轨迹、人体关键点），把计数要用的东西缓存下来，
之后调运球计数规则不用重跑模型。

  python tools/eval/dribble_extract.py [视频 ...] [--out out/dribble] [--force]   # 不给视频就处理 samples/dribble/*.mp4

缓存 <out>/<名>.ball.json：fps、width、height、frames、ball（选中的球轨迹 [帧, x, y, r]）、cands（每帧所有球候选）、
persons（每帧主要人物框）、kpts（每帧关键点）、scene、source、timing。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from combocam.analyze import analyze  # noqa: E402
from combocam.video import content_bottom  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    out_dir = ROOT / "out" / "dribble"
    if "--out" in argv:
        i = argv.index("--out")
        out_dir = Path(argv[i + 1])
        del argv[i:i + 2]
    vids = [Path(a) for a in argv if not a.startswith("--")] or sorted((ROOT / "samples" / "dribble").glob("*.mp4"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in vids:
        out = out_dir / f"{v.stem}.ball.json"
        if out.exists() and "--force" not in argv:
            continue
        dbg: dict = {}
        tr = analyze(str(v), model=str(ROOT / "models" / "yolox_s.onnx"), pose_model=str(ROOT / "models" / "rtmpose-m_halpe26_256x192.onnx"),
                     debug=dbg, content_only=True)
        d = {k: tr[k] for k in ("fps", "width", "height", "frames", "ball", "scene", "source", "timing")}
        d.update({"cands": dbg.get("cands"), "persons": dbg.get("persons"), "kpts": dbg.get("kpts"),
                  "content_bottom": content_bottom(tr["source"]["probe"]["width"], tr["source"]["probe"]["height"], tr["width"], tr["height"])})
        out.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
        print(f"{v.stem}: {tr['frames']} 帧，球 {len(tr['ball'])} 帧（{len(tr['ball']) / max(1, tr['frames']):.0%}），"
              f"用时 {tr['timing']['total_s']}s", flush=True)


if __name__ == "__main__":
    main()
