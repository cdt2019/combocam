"""跳绳原型：逐帧找主要人物、跑 26 点关键点，缓存到 out/jumprope/<名>.pose.json（之后调计数规则不用重跑模型）。

  python tools/eval/jumprope_extract.py [视频 ...] [--force] [--out 目录]   # 不给视频就处理 samples/jumprope/*.mp4，已有缓存的跳过

主要人物的选法见 combocam/jumprope_track.extract。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from combocam.detector import YoloxDetector  # noqa: E402
from combocam.jumprope_track import extract  # noqa: E402
from combocam.pose import RTMPose  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    out_dir = ROOT / "out" / "jumprope"
    if "--out" in argv:
        i = argv.index("--out")
        out_dir = Path(argv[i + 1])
        del argv[i:i + 2]
    vids = [Path(a) for a in argv if not a.startswith("--")] or sorted((ROOT / "samples" / "jumprope").glob("*.mp4"))
    out_dir.mkdir(parents=True, exist_ok=True)
    det = YoloxDetector(str(ROOT / "models" / "yolox_s.onnx"))
    pose = RTMPose(str(ROOT / "models" / "rtmpose-m_halpe26_256x192.onnx"))
    for v in vids:
        out = out_dir / f"{v.stem}.pose.json"
        if out.exists() and "--force" not in argv:
            continue
        r = extract(str(v), det, pose)
        out.write_text(json.dumps(r, separators=(",", ":")), encoding="utf-8")
        found = sum(1 for k in r["kpts"] if k)
        print(f"{v.stem}: {r['frames']} 帧，有人 {found} 帧，最多 {max(r['people'] or [0])} 人，用时 {r['extract_s']}s", flush=True)


if __name__ == "__main__":
    main()
