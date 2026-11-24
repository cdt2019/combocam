"""下载测试视频：按 samples/manifest.json 里的 Pexels 编号，用 Pexels 官方 API 找到当时用的那个版本，下载到对应目录。

  export PEXELS_API_KEY=...                                  # 免费申请：https://www.pexels.com/api/
  python tools/fetch_samples.py                              # 全部（约 310 段、3 GB）
  python tools/fetch_samples.py --set samples/heldout5        # 只下一个目录
  python tools/fetch_samples.py --list                       # 只列出各目录的段数和已下载的段数

- 已经下好、大小对得上的跳过，可以中断后重跑。
- 按清单里的宽、高、帧率挑版本；挑不到一模一样的，取宽高最接近的并提示（标注帧号按 30 fps 画布计，通常照样能用）。
- 下载完核对 sha256，对不上只提示不删除（Pexels 偶尔会重新转码）。
- Pexels API 默认每小时 200 次请求，超了会等一会儿再试。
- 视频的许可见 https://www.pexels.com/license/ ：可以免费用于测试，不能原样再分发。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "samples" / "manifest.json"
API = "https://api.pexels.com/videos/videos/{}"
UA = "combocam-fetch-samples/0.1 (+https://www.pexels.com/api/)"


def request(url: str, key: str | None = None, tries: int = 5) -> bytes:
    headers = {"User-Agent": UA}
    if key:
        headers["Authorization"] = key
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and k + 1 < tries:  # 超过每小时的请求数
                wait = 60 * (k + 1)
                print(f"  请求太多，等 {wait} 秒")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            if k + 1 < tries:
                time.sleep(5)
                continue
            raise
    raise RuntimeError("unreachable")


def fps_value(s: str | float | None) -> float:
    if s is None:
        return 0.0
    if isinstance(s, str) and "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def pick(files: list[dict], clip: dict) -> tuple[dict, bool]:
    """按宽高（再按帧率）挑最接近清单的版本；返回 (版本, 是否完全一致)。"""
    want_fps = fps_value(clip["fps"])

    def cost(f: dict) -> tuple:
        return (abs((f.get("width") or 0) - clip["width"]) + abs((f.get("height") or 0) - clip["height"]),
                abs(fps_value(f.get("fps")) - want_fps))

    best = min((f for f in files if f.get("link")), key=cost)
    exact = best.get("width") == clip["width"] and best.get("height") == clip["height"]
    if clip.get("file"):  # 当时记下了文件名的，优先用同名的
        same = [f for f in files if f.get("link", "").split("?")[0].endswith(clip["file"])]
        if same:
            return same[0], True
    return best, exact


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", action="append", help="只下这个目录（可重复），例如 samples/heldout5")
    ap.add_argument("--list", action="store_true", help="只列出各目录的段数和已下载的段数")
    a = ap.parse_args()
    sets = json.loads(MANIFEST.read_text(encoding="utf-8"))["sets"]
    if a.set:
        unknown = [s for s in a.set if s.rstrip("/") not in sets]
        if unknown:
            sys.exit(f"清单里没有 {unknown}，可选：{', '.join(sets)}")
        sets = {s: sets[s] for s in sets if s in {x.rstrip('/') for x in a.set}}
    if a.list:
        for d, clips in sets.items():
            have = sum((ROOT / d / f"pexels_{c['id']}.mp4").exists() for c in clips)
            print(f"{d:22s} {len(clips):3d} 段，已下载 {have}")
        return
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        sys.exit("需要 Pexels API key：到 https://www.pexels.com/api/ 免费申请，再 export PEXELS_API_KEY=...")
    done = skipped = failed = 0
    for d, clips in sets.items():
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        for c in clips:
            dst = ROOT / d / f"pexels_{c['id']}.mp4"
            if dst.exists() and dst.stat().st_size == c["bytes"]:
                skipped += 1
                continue
            try:
                info = json.loads(request(API.format(c["id"]), key))
                f, exact = pick(info.get("video_files", []), c)
                print(f"{d}/pexels_{c['id']}.mp4  ←  {f.get('width')}×{f.get('height')} {f.get('fps')}fps" + ("" if exact else "（和清单的尺寸不一样）"))
                tmp = dst.with_suffix(".part")
                tmp.write_bytes(request(f["link"]))
                tmp.replace(dst)
                if sha256(dst) != c["sha256"]:
                    print("  提示：sha256 和清单不同（Pexels 可能重新转码过），评估结果可能有细微差别")
                done += 1
            except Exception as e:  # noqa: BLE001 —— 一段失败不影响其他
                print(f"  失败 pexels_{c['id']}：{e}")
                failed += 1
    print(f"下载 {done} 段，已存在跳过 {skipped} 段，失败 {failed} 段")


if __name__ == "__main__":
    main()
