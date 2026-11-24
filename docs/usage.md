# 命令行用法

所有命令都是 `python -m combocam <子命令>`（用 `pip install -e .` 装过以后也可以直接 `combocam <子命令>`）。
输入视频任意尺寸、帧率，先统一成 1080×1920@30 的竖版画布（横屏视频居中放进竖版，上下补模糊背景）。

## 颠球

```bash
# 分析 + 渲染（输出到目录：<名>.track.json、<名>.<风格>.mp4、<名>.report.json）
python -m combocam run in.mp4 -o out/demo

# 一次出多种风格（只分析一遍）
python -m combocam run in.mp4 -o out/demo --style replay,neon,flame,comet,broadcast

# 只分析，写轨迹 JSON
python -m combocam analyze in.mp4 -o out/in.track.json

# 用已有的轨迹渲染（换风格、换布局不用重新分析）
python -m combocam render in.mp4 out/in.track.json -o out/in.flame.mp4 --style flame
```

## 跳绳

```bash
python -m combocam jumprope in.mp4 -o out/jr                     # 关键点 → 计数 → 成片 + 报告
python -m combocam jumprope in.mp4 -o out/jr --mode pk            # 双人 PK（左红右蓝）
python -m combocam jumprope in.mp4 -o out/jr --mode auto          # 两个人明显在一起跳时自动出 PK
python -m combocam jumprope in.mp4 -o out/jr --no-render --save-pose   # 只计数，另存逐帧关键点
```

- `--mode single|pk|auto`：单人（默认，多人入镜时挑在跳绳的那个）、双人 PK、自动。
- `--pk-rule auto|jumps|doubles|passes`：PK 怎么比。`auto` 两人都以双摇为主时比双摇个数，否则比总跳数；
  `passes` 比绳子过脚的次数（双摇算 2）。

## 篮球运球

```bash
python -m combocam dribble in.mp4 -o out/dr                      # 计数 + 成片
python -m combocam dribble in.mp4 -o out/dr --no-render --save-cache   # 只计数，另存球的候选和关键点
```

## 分析参数（所有子命令通用）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device auto\|cuda\|cpu` | `auto` | 推理设备 |
| `--model` | `models/yolox_s.onnx` | 检测模型 |
| `--pose-model` | `models/rtmpose-m_halpe26_256x192.onnx` | 人体关键点模型；颠球可以传 `none`，退回只看轨迹和人框的旧规则 |
| `--refine online\|offline\|off` | `online` | 球漏检时在预测位置裁原图复检：`online` 在第 1 遍里顺带做；`offline` 再解码一遍；`off` 不做（快，漏检多） |
| `--fps` | `30` | 分析和成片的帧率 |
| `--no-frame-cache` | | 不把解码帧存进内存盘给渲染用（见 [performance.md](performance.md)） |

## 渲染参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--style` | `replay` | 配色风格：`replay`（荧光黄绿）、`neon`、`flame`、`comet`、`broadcast` |
| `--layout` | `v3` | HUD 布局，见 [rendering.md](rendering.md#布局)。`v3s`、`v3l` 把数据面板抬高，避开短视频 App 的界面 |
| `--protect` | `light` | 文字保护层：`light`（上下轻渐变 + 文字阴影）、`plates`、`edges`、`veil`、`mix` |
| `--tint` | `panel` | 文字颜色：`panel`（底部面板的字用曲线颜色）、`all`（计数也换）、`none`（白字） |
| `--num-font` | `bebas` | 计数数字字体：`bebas`、`inter`、`noto`、`barlow`、`poppins`、`anton` |
| `--panel-font` | `barlowc` | 底部面板字体：`barlowc`、`noto`、`smiley`、`zcool`、`inter` |
| `--endcard card\|none` | `card` | 片尾成绩卡（3 秒） |
| `--endcard-text` | 无 | 片尾最下面加一行字，比如账号名、话题 |
| `--nickname` | 无 | 片尾顶部显示 `@昵称` |
| `--slowmo` | | 颠球高光段 0.5 倍慢放（会去掉原声） |
| `--workers` | CPU 核数的一半 | 渲染线程数 |
| `--encoder` | `libx264` | `h264_nvenc` 用 GPU 编码（见 [install.md](install.md#gpu-编码可选)） |
| `--preset` | x264 `veryfast` / NVENC `p5` | 编码速度档位 |
| `--force` | | 判为"不出片"时照样渲染（调试用） |

## 不出片

分析完先判断能不能出片。下面这些情况不渲染，命令以**退出码 3** 结束，轨迹 JSON 和报告里的 `status` 写明原因
（`title` 标题、`message` 正文、`tips` 拍摄要点，可以直接显示给用户）：

| 项目 | `status.code` | 条件 |
|---|---|---|
| 颠球 | `no_ball` | 球只在不到 15% 的帧里被找到 |
| 颠球 | `no_person` | 人只在不到 30% 的帧里被找到 |
| 颠球 | `no_juggling` | 一次触球都没有（地上带球、拨球、头上平衡） |
| 跳绳 | `no_jumps` | 少于 3 跳 |
| 跳绳 PK | `no_pk` | 有一方少于 3 跳 |
| 运球 | `no_dribble` | 少于 3 下 |

颠球颠到 1 个就出片（`combocam/verdict.py` 的 `MIN_COMBO`）。42 段测试素材里有 6 段"在地上玩球"的视频会被误数出 1–2 个、照样出片；
把 `MIN_COMBO` 提到 3 能全部挡掉（真颠球的 29 段最长连击都 ≥ 3），但也会挡掉只会颠一两个的新手。

## 输出文件

| 文件 | 内容 |
|---|---|
| `<名>.track.json`（跳绳 `.jumprope.json`，运球 `.dribble.json`） | 轨迹数据，格式见 [track-format.md](track-format.md) |
| `<名>.<风格>.mp4` | 1080×1920 的特效成片（含片尾） |
| `<名>.report.json` | 状态、计数摘要、警告、各步用时 |

`warnings` 里是给人看的提示，比如"像是慢动作视频""球径比例尺和重力比例尺差得多，高度仅供参考""脚被画面底边切掉"。

## 直接调用渲染器

`python -m combocam render` 只是包了一层 Node 渲染器，也可以直接调：

```bash
node render/render.mjs --input in.mp4 --track out/in.track.json --output out/in.mp4 [--style replay] [--layout v3s] [--encoder h264_nvenc]
```

渲染器按轨迹 JSON 的 `sport` 字段选特效：颠球、运球用 `render/fx.js`，跳绳用 `render/jumprope_fx.js`，PK 用 `render/jumprope_pk_fx.js`。

调效果时不用渲染整段，用单帧预览（见 [rendering.md](rendering.md#单帧预览)）：

```bash
node render/still.mjs --track out/in.track.json --t 3.4 --variants "v3:light:replay,v3s:light:replay:safe" --out out/still.png
```
