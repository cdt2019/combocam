"""对 samples/<集>/ 里的视频跑分析（默认 heldout，--set heldout5 换），存 out/<集>/<名>.track.json 和 .debug.json（候选框，供 reanalyze / 评估复用）。

  python tools/eval/run_heldout.py [名 ...]       # 不给名字就跑全部；已有结果的跳过（--force 重跑）
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SET = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "heldout"  # 测试集目录：samples/<SET>/、out/<SET>/
sys.path.insert(0, str(ROOT))

from combocam.analyze import analyze  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a != SET]
    force = "--force" in sys.argv
    out_dir = ROOT / "out" / SET
    out_dir.mkdir(parents=True, exist_ok=True)
    vids = sorted((ROOT / "samples" / SET).glob("*.mp4"))
    if args:
        vids = [v for v in vids if v.stem in args]
    for v in vids:
        tp, dp = out_dir / f"{v.stem}.track.json", out_dir / f"{v.stem}.debug.json"
        if tp.exists() and dp.exists() and not force:
            print(f"跳过 {v.stem}（已有结果）")
            continue
        dbg: dict = {}
        # 日志（含计数结果）只写文件，不打到屏幕：人工标注前不看算法结果
        with open(out_dir / f"{v.stem}.log", "w", encoding="utf-8") as lf:
            tr = analyze(str(v), model=str(ROOT / "models" / "yolox_s.onnx"), debug=dbg, log=lambda *a: print(*a, file=lf))
        tp.write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
        dp.write_text(json.dumps(dbg), encoding="utf-8")
        print(f"{v.stem}: {tr['frames']} 帧，用时 {tr['timing']['total_s']}s")


if __name__ == "__main__":
    main()
