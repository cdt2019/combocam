# 整体结构

```
视频 ──FFmpeg 统一成 1080×1920@30──┬─▶ 分析（Python，combocam/）───────────────▶ 轨迹 JSON ──┐
                                   │   YOLOX-s：逐帧检测人和球                             │
                                   │   RTMPose-m：主要人物的 26 个关键点（含脚趾、脚跟）     │
                                   │   颠球 / 运球：选出球的轨迹 → 触球 / 触地 → 连击        │
                                   │   跳绳：髋部起伏 + 两脚离地 → 每一跳 → 断绳 / 双摇      │
                                   └─▶ 渲染（Node + Skia，render/）◀───────────────────────┘
                                       特效是时间的纯函数，多线程乱序画 → x264 / NVENC 编码
```

分析和渲染只靠[轨迹 JSON](track-format.md)交接：换风格、换布局不用重新分析；别的前端（比如网页预览）也可以直接读这份 JSON。

## 设计取舍

- **不训练模型**：检测用 COCO 预训练的 YOLOX-s（`person`、`sports ball` 两类），关键点用 RTMPose-m Halpe26，计数全靠规则和物理量。
  这样不需要标注数据就能跑起来，代价是规则要在真实视频上反复核对（见 [evaluation.md](evaluation.md)）。
- **阈值用物理单位**：距离以"球半径""身高"为单位，速度以"每秒几个球半径"为单位，比例尺来自球的真实大小和重力加速度，
  所以近景、远景、横屏、竖屏用同一套阈值。每个项目的阈值集中在一个 `P` 字典里（`combocam/events.py`、`combocam/jumprope.py`、`combocam/dribble.py`）。
- **整段离线分析**：选球的轨迹是全片的最优路径（前后帧一起看），球的大小、重力比例尺、跳绳的节奏都按整段估。
  试过只用前一部分视频先算、先渲染：落后 3 秒的检查点只有 55% 和整段算的完全一致，所以没有做"边分析边渲染"。
- **特效是纯函数**：同一个时刻永远画出同一帧，粒子的随机数用触球序号和时刻做种子。服务端可以多线程乱序渲染，预览可以随意拖动，
  改渲染代码前后可以逐帧比对像素 MD5（见 [rendering.md](rendering.md#核对画面)）。

## 目录

```
combocam/        分析（Python 包）
render/        渲染（Node）
tools/         fetch_assets.py 下载模型和字体、fetch_samples.py 下载测试视频、make_synthetic.py 合成测试视频
  eval/        评估：跑模型、缓存中间结果、对标注算准确率
  label/       标注辅助：逐帧小图、运动轨迹图、写标注
  debug/       调试：轨迹图、关键点图、成片小图、GPU 预处理核对、绳子信号实验
tests/         计数逻辑的回归测试（合成数据，不需要模型和视频）
samples/       测试视频的清单（manifest.json）和人工标注；视频本身用 tools/fetch_samples.py 下载
docs/          文档
models/ fonts/ 用 tools/fetch_assets.py 下载
out/           输出（不入库）
```

### combocam/

| 模块 | 作用 |
|---|---|
| `cli.py` | 命令行入口（`python -m combocam ...`） |
| `video.py` | FFmpeg 读帧、统一成画布、解码帧缓存 |
| `detector.py` | YOLOX ONNX 检测（只留 person、sports ball） |
| `pose.py` | RTMPose 关键点（Halpe26） |
| `onnx_io.py` | 启动时改模型图：预处理挪到 GPU、只拷回候选行 |
| `scene.py` | 画面亮度统计（给文字可读性用） |
| `analyze.py` | 颠球 / 运球的分析：逐帧检测、人周围放大复检、漏检处局部复检、选轨迹、关键点 |
| `track.py` | 从每帧的候选框里选出"被颠的那个球"的轨迹（有向无环图上的最优路径） |
| `events.py` | 颠球：分段拟合抛物线 → 触球、落地、手接、最高点、连击、高光 |
| `body.py` | 关键点判断：地面线、触点离哪个身体部位最近、是不是手 / 前臂 |
| `verdict.py` | 颠球能不能出片 |
| `jumprope.py` | 跳绳计数：每一跳、断绳、双摇、一圈跳两下 |
| `jumprope_track.py` | 跳绳：逐帧关键点、慢动作识别、渲染用的轨迹 JSON、双人 PK |
| `people.py` | 多人入镜：每个人连成轨迹、挑在跳绳的人 |
| `dribble.py` | 运球计数和动作（单手 / 交叉 / 胯下 / 背后） |
| `dribble_track.py` | 运球：渲染用的轨迹 JSON |

### render/

| 文件 | 作用 |
|---|---|
| `render.mjs` | 成片：解码 → 多个工作线程画特效 → 按顺序送进编码器；加片尾 |
| `fx.js` | 颠球、运球的特效和 HUD（只用标准 Canvas 2D API，浏览器里也能跑） |
| `jumprope_fx.js` | 跳绳特效 |
| `jumprope_pk_fx.js` | 跳绳双人 PK 特效 |
| `styles.json` | 配色风格、里程碑间隔、PK 双方的名字和颜色 |
| `fonts.mjs` | 注册字体 |
| `video_source.mjs` | 视频帧提供者（整帧、抠球的小块、缩小的模糊底图） |
| `pixels.mjs`、`native/pixels.c` | RGBA→YUV、保护层压暗（C 版本可选） |
| `still.mjs` | 单帧预览：几种方案并排画一张图，标出实测对比度 |
| `check_legibility.mjs` | 用真实像素核对 HUD 文字的对比度 |
| `bench_fx.mjs`、`profile_parts.mjs` | 剖析每帧各部分耗时 |
