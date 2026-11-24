"""每段视频均匀抽 N 帧拼成一行缩略图，多段视频上下排成一张图，用来快速挑测试素材。

  python tools/debug/contact_sheet.py 输出.png 视频... [--n 6] [--h 180]
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from combocam.video import FFMPEG, probe  # noqa: E402


def main() -> None:
    args = sys.argv[1:]
    n = int(args[args.index("--n") + 1]) if "--n" in args else 6
    hh = int(args[args.index("--h") + 1]) if "--h" in args else 180
    files = [a for i, a in enumerate(args[1:], 1) if not a.startswith("--") and args[i - 1] not in ("--n", "--h")]
    out = args[0]
    rows = []
    for f in files:
        d = probe(f).duration
        inputs = []
        for k in range(n):
            inputs += ["-ss", f"{d * (k + 0.5) / n:.2f}", "-i", f]
        chains = ";".join(f"[{k}:v]scale=-2:{hh},setsar=1,format=rgb24[v{k}]" for k in range(n))
        graph = f"{chains};{''.join(f'[v{k}]' for k in range(n))}hstack=inputs={n}[row]"
        print(f"第 {len(rows) + 1} 行：{Path(f).stem}")
        row = Path(out).with_suffix(f".{len(rows)}.png")
        subprocess.run([FFMPEG, "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[row]", "-frames:v", "1", str(row)], check=True)
        rows.append(row)
    # 各行宽度不同：先补成同宽再竖着拼
    ins = sum((["-i", str(r)] for r in rows), [])
    graph = "".join(f"[{k}:v]pad=2400:ih:0:0:white[r{k}];" for k in range(len(rows))) + "".join(f"[r{k}]" for k in range(len(rows))) + f"vstack=inputs={len(rows)}"
    subprocess.run([FFMPEG, "-v", "error", "-y", *ins, "-filter_complex", graph, "-frames:v", "1", out], check=True)
    for r in rows:
        r.unlink()
    print(out)


if __name__ == "__main__":
    main()
