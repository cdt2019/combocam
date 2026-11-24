# Third-party notices / 第三方许可

本项目代码以 MIT 许可发布（见 [LICENSE](LICENSE)）。下面这些第三方的模型、字体、库和测试素材**不随仓库分发**，
由 `tools/fetch_assets.py`、`tools/fetch_samples.py`、pip、npm 从各自官方渠道下载，许可以各项目的声明为准。

## 模型（`tools/fetch_assets.py`）

| 模型 | 来源 | 许可 |
|---|---|---|
| YOLOX-s（以及对比用的 tiny、m） | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) release 0.1.1rc0 | Apache-2.0（权重在 COCO 上训练） |
| RTMPose-m Halpe26 | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) rtmposev1 onnx_sdk | Apache-2.0（权重在 Body7 等公开数据集上训练） |

预训练权重用到的训练数据（COCO、Halpe 等）各有自己的使用条款，商用前请自行核实。

## 字体（`tools/fetch_assets.py`）

全部是 [SIL Open Font License 1.1](https://openfontlicense.org/)，下载时一并下载各自的许可文件到 `fonts/OFL-*.txt`：
Bebas Neue、Anton、Barlow、Barlow Condensed、Inter、Poppins、Noto Sans SC（思源黑体）、Smiley Sans（得意黑）、ZCOOL QingKe HuangYou（站酷庆科黄油体）。

## 库

| 依赖 | 许可 |
|---|---|
| ONNX Runtime | MIT |
| ONNX | Apache-2.0 |
| OpenCV | Apache-2.0 |
| NumPy、SciPy | BSD-3-Clause |
| @napi-rs/canvas | MIT（底层 Skia 为 BSD-3-Clause） |
| FFmpeg | LGPL / GPL，取决于构建；本项目只调用系统里的 `ffmpeg` 命令，不分发它 |

## 测试视频（`tools/fetch_samples.py`）

`samples/manifest.json` 里列出的测试视频全部来自 [Pexels](https://www.pexels.com/)，按 [Pexels 许可](https://www.pexels.com/license/)使用：
可以免费使用和修改，不能原样再分发或出售，所以仓库里只有编号、校验值和人工标注。视频里的人物都是真实的人，
请不要把这些视频或用它们生成的成片用在会让人误以为他们为某个产品代言的场合。

README 和文档里的演示动图是用本项目处理 Pexels 视频后的成片截取的，原视频见 [docs/media/CREDITS.md](docs/media/CREDITS.md)。
