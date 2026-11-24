"""写入/更新一段视频的人工标注（samples/<集>/labels.json，默认 heldout，--set heldout5 换）。

  python tools/label/set_label.py <名> --touches 1,2,3 [--ambiguous ..] [--bounces ..] [--nocount ..] --note "..."

touches=确定的触球（脚、腿、头、胸等）；ambiguous=看不清，算不算都不扣分；
bounces=落地（会断连击）；nocount=手、手臂接触或停在身上滚动（不算触球，算了就是误报）；ignore=看不见球的时段 a-b,c-d（被挡住、出画面），
这段里算法报不报都不计分。帧号按 30fps 画布。
"""

import argparse
import json
import sys
from pathlib import Path

SET = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "heldout"
P = Path(__file__).resolve().parents[2] / "samples" / SET / "labels.json"


def ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()] if s else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    for k in ("touches", "ambiguous", "bounces", "nocount"):
        ap.add_argument(f"--{k}", default="")
    ap.add_argument("--ignore", default="", help="a-b,c-d")
    ap.add_argument("--note", default="")
    ap.add_argument("--set", default="heldout")
    a = ap.parse_args()
    data = json.loads(P.read_text(encoding="utf-8")) if P.exists() else {
        "_comment": "留出测试集：没参与调参。人工逐帧标注，标注时没看算法的计数结果。帧号按 30fps 画布（= 秒 × 30）。"
                    "touches=确定的触球；ambiguous=看不清，算不算都不扣分；bounces=落地（断连击）；nocount=手/手臂接触、球在身上滚动（不算触球）。"}
    data[f"{a.name}.mp4"] = {"note": a.note, "touches": ints(a.touches), "ambiguous": ints(a.ambiguous),
                             "bounces": ints(a.bounces), "nocount": ints(a.nocount),
                             "ignore": [[int(v) for v in r.split("-")] for r in a.ignore.split(",") if r]}
    P.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{a.name}: {len(ints(a.touches))} 次触球")


if __name__ == "__main__":
    main()
