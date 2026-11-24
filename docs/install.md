# 安装

## 需要什么

| | 版本 | 说明 |
|---|---|---|
| Python | 3.10 及以上 | 分析（检测、关键点、计数） |
| Node.js | 18 及以上 | 渲染特效成片（Skia，`@napi-rs/canvas`） |
| FFmpeg | 带 libx264 | 解码、编码，要在 `PATH` 里 |
| NVIDIA 显卡 | 可选 | 没有也能跑，但很慢，见下面"只用 CPU" |

## 步骤

```bash
git clone https://github.com/cdt2019/combocam.git && cd combocam

python -m venv .venv
source .venv/bin/activate            # Windows：.venv\Scripts\activate
pip install -r requirements.txt      # 有 NVIDIA 显卡改用 requirements-gpu.txt（见下）

python tools/fetch_assets.py         # 下载模型（YOLOX-s、RTMPose-m，约 90 MB）和字体（约 37 MB）
cd render && npm ci && cd ..

python -m pytest                     # 可选：回归测试，不需要模型和视频，十几秒
```

也可以用 `pip install -e ".[cpu]"`（或 `".[gpu]"`）把 `combocam` 装成命令，之后 `combocam run ...` 和 `python -m combocam run ...` 一样。

试一下：拿一段自己拍的颠球视频（手机固定、全身入镜），或者下载测试视频（要免费的 Pexels API key，见 [evaluation.md](evaluation.md#下载测试视频)）：

```bash
PEXELS_API_KEY=... python tools/fetch_samples.py --set samples     # 4 段颠球，约 60 MB
python -m combocam run samples/pexels_38236117.mp4 -o out/demo
```

## 用 GPU

```bash
pip install -r requirements-gpu.txt
```

- `onnxruntime-gpu==1.18.0` 需要 CUDA 11.8 和 cuDNN 8。CUDA 12 的环境换成 `onnxruntime-gpu>=1.19`。
  pip 装的 CUDA / cuDNN 要和驱动支持的 CUDA 版本对上：驱动只到 12.4 时装了 12.9 的库，模型能加载，
  一推理就报 `CUDNN_STATUS_EXECUTION_FAILED_CUDART`。
- **Windows**：把 `COMBOCAM_CUDA_DLL_DIR` 指向含 `cudart64_110.dll`、`cudnn64_8.dll` 的目录
  （比如某个 PyTorch cu118 环境的 `site-packages/torch/lib`）。
- 找不到 CUDA 时自动退回 CPU，日志里会有 onnxruntime 的报错，结果不受影响，只是慢。
- `onnx` 包是可选的：装了以后启动时在内存里给模型接上"GPU 上做预处理、只拷回候选行"，检测每帧快约 30%，结果逐位相同；
  没装（或加载失败）就退回 CPU 预处理。

**只用 CPU**：YOLOX-s 每帧约 0.76 秒，30 秒的视频分析要 11 分钟以上。用来试一试可以，批量跑建议用 GPU。

## GPU 编码（可选）

渲染默认用 libx264。显卡支持 NVENC 时加 `--encoder h264_nvenc`，编码不再占 CPU：

```bash
python -m combocam run in.mp4 -o out/demo --encoder h264_nvenc
```

新版 FFmpeg 的 NVENC 对驱动版本有要求（FFmpeg 9 要 610 以上）。老显卡装不到这么新的驱动时，
可以另放一个旧版 FFmpeg 专门用来编码，用 `COMBOCAM_FFMPEG_ENC` 指过去（只换编码，解码照旧用 `PATH` 里的，画面不变）。
Docker 里用 NVENC，要在 `NVIDIA_DRIVER_CAPABILITIES` 里加上 `video`。

## 原生像素模块（可选，Linux）

渲染时 RGBA→YUV 转换和保护层压暗有一个 C 版本，每帧省约 20 ms：

```bash
bash render/native/build.sh     # 需要 gcc，生成 render/native/pixels.node
```

没编译时自动用 JS 版本，结果逐字节相同。`COMBOCAM_NO_NATIVE=1` 强制用 JS 版本。

## 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `COMBOCAM_CUDA_DLL_DIR` | 无 | Windows 上 CUDA / cuDNN DLL 所在目录 |
| `COMBOCAM_GPU_IO` | `1` | `0` 关掉"GPU 上做预处理"（对比、排查用） |
| `COMBOCAM_ZOOM` | `1` | `0` 关掉颠球的"人周围放大复检"（见 [juggling.md](juggling.md)） |
| `COMBOCAM_FRAME_CACHE_GB` | `6` | 解码帧缓存的上限（GB），`0` 关闭（见 [performance.md](performance.md)） |
| `COMBOCAM_FFMPEG_ENC` | `ffmpeg` | 只用于编码的 FFmpeg 路径 |
| `COMBOCAM_NO_NATIVE` | 无 | `1` 时不用 C 原生像素模块 |
| `PEXELS_API_KEY` | 无 | `tools/fetch_samples.py` 下载测试视频用 |

## 常见问题

- **`npm ci` 装不上 `@napi-rs/canvas`**：它按平台下载预编译包，确认 Node 是 64 位的 18 以上版本；国内网络慢可以设 npm 镜像。
- **渲染报"编码器提前退出"**：多半是 `h264_nvenc` 不可用，换回 `--encoder libx264`，或者按上面设 `COMBOCAM_FFMPEG_ENC`。
- **中文字体显示成方框**：没跑 `python tools/fetch_assets.py`，或者 `fonts/` 里缺 `NotoSansSC-*.otf`。
