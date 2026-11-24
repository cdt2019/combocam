"""下载模型和字体（都来自官方仓库，许可见 THIRD_PARTY_NOTICES.md）。

  python tools/fetch_assets.py            # yolox_s + RTMPose 关键点 + 字体
  python tools/fetch_assets.py --all      # 另加 yolox_tiny、yolox_m 做对比
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLOX = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{}.onnx"  # Apache-2.0
# RTMPose-m，Halpe26 关键点（带脚趾、脚跟），Apache-2.0；rtmlib BodyWithFeet 的 balanced 档用的同一个
RTMPOSE = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
           "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip")
GF = "https://raw.githubusercontent.com/google/fonts/main/ofl"
FONTS = {  # 全部 SIL OFL 1.1
    "Anton-Regular.ttf": f"{GF}/anton/Anton-Regular.ttf",
    "OFL-Anton.txt": f"{GF}/anton/OFL.txt",
    "NotoSansSC-Medium.otf": "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansSC-Medium.otf",
    "NotoSansSC-Bold.otf": "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansSC-Bold.otf",
    "NotoSansSC-Black.otf": "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansSC-Black.otf",
    "OFL-NotoSansSC.txt": f"{GF}/notosanssc/OFL.txt",
    "BebasNeue-Regular.ttf": f"{GF}/bebasneue/BebasNeue-Regular.ttf",
    "OFL-BebasNeue.txt": f"{GF}/bebasneue/OFL.txt",
    "Barlow-Bold.ttf": f"{GF}/barlow/Barlow-Bold.ttf",
    "OFL-Barlow.txt": f"{GF}/barlow/OFL.txt",
    "BarlowCondensed-BoldItalic.ttf": f"{GF}/barlowcondensed/BarlowCondensed-BoldItalic.ttf",
    "OFL-BarlowCondensed.txt": f"{GF}/barlowcondensed/OFL.txt",
    "Poppins-Bold.ttf": f"{GF}/poppins/Poppins-Bold.ttf",
    "OFL-Poppins.txt": f"{GF}/poppins/OFL.txt",
    "ZCOOLQingKeHuangYou-Regular.ttf": f"{GF}/zcoolqingkehuangyou/ZCOOLQingKeHuangYou-Regular.ttf",
    "OFL-ZCOOLQingKeHuangYou.txt": f"{GF}/zcoolqingkehuangyou/OFL.txt",
    "OFL-SmileySans.txt": "https://raw.githubusercontent.com/atelier-anchor/smiley-sans/main/LICENSE",
    "OFL-Inter.txt": f"{GF}/inter/OFL.txt",
}
SMILEY = "https://github.com/atelier-anchor/smiley-sans/releases/download/v2.0.1/smiley-sans-v2.0.1.zip"  # 得意黑
INTER_VF = f"{GF}/inter/Inter%5Bopsz%2Cwght%5D.ttf"


def get(url: str) -> bytes:
    print(f"下载 {url}")
    with urllib.request.urlopen(url) as r:
        return r.read()


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"已存在 {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(get(url))


def from_zip(url: str, member_suffix: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"已存在 {dst.relative_to(ROOT)}")
        return
    z = zipfile.ZipFile(io.BytesIO(get(url)))
    name = next(n for n in z.namelist() if n.endswith(member_suffix))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(z.read(name))


def inter_extrabold(dst: Path) -> None:
    """Inter 只发布可变字体；Skia 画不准可变字重，切出一个 wght 800、opsz 32 的静态字体。需要 fonttools。"""
    if dst.exists():
        print(f"已存在 {dst.relative_to(ROOT)}")
        return
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    font = instantiateVariableFont(TTFont(io.BytesIO(get(INTER_VF))), {"wght": 800, "opsz": 32})
    font.save(str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for m in ["yolox_s"] + (["yolox_tiny", "yolox_m"] if a.all else []):
        fetch(YOLOX.format(m), ROOT / "models" / f"{m}.onnx")
    from_zip(RTMPOSE, "end2end.onnx", ROOT / "models" / "rtmpose-m_halpe26_256x192.onnx")
    for name, url in FONTS.items():
        fetch(url, ROOT / "fonts" / name)
    from_zip(SMILEY, "SmileySans-Oblique.ttf", ROOT / "fonts" / "SmileySans-Oblique.ttf")
    try:
        inter_extrabold(ROOT / "fonts" / "Inter-ExtraBold.ttf")
    except ImportError:
        print("跳过 Inter（需要 pip install fonttools）")


if __name__ == "__main__":
    main()
