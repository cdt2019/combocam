"""跳绳成片的关键帧小图：每段成片一行 —— 第 3 跳腾空、第 3 跳落地后、断绳后、第一个双摇（有的话）、里程碑横幅、片尾。

  python tools/debug/jumprope_sheet.py 输出.png 成片目录 名... [--h 640]

成片目录里要有 <名>.jumprope.json 和 <名>.jumprope.replay.mp4（python -m combocam jumprope 的输出）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from combocam.video import FFMPEG  # noqa: E402


def frames_for(tr: dict) -> list[int]:
    fps = tr["fps"]
    if tr.get("mode") == "pk":  # 双人 PK：双方的跳合在一起挑
        js = sorted((j for pl in tr["players"] for j in pl["jumps"]), key=lambda j: j["t"])
        js = [{**j, "total": i + 1} for i, j in enumerate(js)]
        tr = {**tr, "breaks": sorted((b for pl in tr["players"] for b in pl["breaks"]), key=lambda b: b["t"])}
    else:
        js = tr["jumps"]
    pick = []
    if len(js) >= 3:
        j = js[2]
        pick += [round((j["up_t"] + j["land_t"]) / 2 * fps), round(j["land_t"] * fps) + 3]
    if tr["breaks"]:
        pick.append(round(tr["breaks"][0]["t"] * fps) + 9)
    dbl = [j for j in js if j.get("passes", 1) >= 2]
    if dbl:  # 第一个双摇：腾空中、落地后
        pick += [round((dbl[0]["up_t"] + dbl[0]["land_t"]) / 2 * fps), round(dbl[0]["land_t"] * fps) + 4]
    m = next((j for j in js if j["total"] == 10), None)
    if m:
        pick.append(round(m["t"] * fps) + 8)
    pick.append(round(tr["frames"] * 0.7))
    pick.append(tr["frames"] + 75)  # 片尾第 2.5 秒
    return sorted(set(pick))


def main() -> None:
    args = sys.argv[1:]
    hh = int(args[args.index("--h") + 1]) if "--h" in args else 640
    rest = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--h")]
    out, d, names = rest[0], Path(rest[1]), rest[2:]
    rows = []
    for name in names:
        tr = json.loads((d / f"{name}.jumprope.json").read_text(encoding="utf-8"))
        fs = frames_for(tr)
        sel = "+".join(f"eq(n\\,{f})" for f in fs)
        row = d / f"{name}.sheet.png"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(d / f"{name}.jumprope.replay.mp4"), "-vf",
                        f"select='{sel}',scale=-2:{hh},tile={len(fs)}x1", "-frames:v", "1", str(row)], check=True)
        rows.append(row)
    if len(rows) == 1:
        Path(out).write_bytes(rows[0].read_bytes())
        return
    # 各行帧数不同：左对齐，右边补黑到最宽那行
    import cv2

    width = max(cv2.imread(str(r)).shape[1] for r in rows)
    inputs = sum([["-i", str(r)] for r in rows], [])
    pads = "".join(f"[{i}:v]pad={width}:ih:0:0[r{i}];" for i in range(len(rows)))
    subprocess.run([FFMPEG, "-v", "error", "-y", *inputs, "-filter_complex",
                    pads + "".join(f"[r{i}]" for i in range(len(rows))) + f"vstack={len(rows)}", str(out)], check=True)

if __name__ == "__main__":
    main()
