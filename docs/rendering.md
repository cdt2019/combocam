# 成片特效

渲染器是 Node + Skia（`@napi-rs/canvas`）。`render/render.mjs` 解码原视频，开几个工作线程并行画特效，按顺序送进编码器，最后加 3 秒片尾成绩卡。

## 画了什么

**颠球**（`render/fx.js`）：

- 球：发光拖尾（按移动距离截取，最长 6 个球径，球几乎不动时淡出）、触球冲击波和火花、贴球细圈、"N · 触球"标签；
- HUD：左上计数（每次触球弹一下）、右上"第 N 组"、里程碑横幅（10、20、30、50、100……）和震屏；
- 底部数据面板：球的高度曲线（最近 4.5 秒）+ 节奏 / 触球间隔 / 最高一颠；
- 落地、用手时提示"落地 · 重新计数""用手 · 重新计数"；
- `--slowmo`：高光段 0.5 倍慢放。

**运球**：同一套 `fx.js`，文字按轨迹 JSON 的 `labels` 换成运球的；交叉、胯下、背后那一下弹出动作名，片尾写各动作次数。

**跳绳**（`render/jumprope_fx.js`）：

- 左上总数，每跳弹一下、+1；断绳时弹出"断绳 · 这组 N 个"，总数不清零，断过以后标签旁写"最长连跳"；
- 脚下：腾空时地面亮一道光，落地时两圈扁波纹加往两边溅的火花；身旁"N · 跳"标签；里程碑横幅 + 震屏；
- 数据：髋部起伏曲线（刻度标出每一跳、红色标断绳）+ 跳速 / 用时 / 断绳；
- 双摇：标签换成"双摇"，双摇落地是三圈大一号的波纹；
- 布局按脚的位置自动选：脚在画面 76% 高度以上用底部面板；再往下（竖拍全身，脚常在 84%–96%）底部面板会盖住脚下的特效，
  改成数据竖排在右上、曲线在计数下面。

**跳绳 PK**（`render/jumprope_pk_fx.js`）：顶部两边名字和计数、中间 VS、按比分分割的红蓝条；每个人头顶一块自己颜色的计数牌，领先的戴皇冠；
领先换人时横幅"蓝方反超！"加震屏。全部信息放在顶部：两个人并排时脚和身体可能在任何高度，底部面板总会挡住其中一个。

**片尾成绩卡**（三个项目都有）：大数字（从 0 滚到最终数）、三项数据、每一下的高度或节奏柱状图（最高 / 最快的那一下白色）、
"你能颠过 N 个吗？"。`--endcard-text` 在最下面加一行字，`--nickname` 在顶部写 `@昵称`，`--endcard none` 不要片尾。

## 风格

`--style`，定义在 `render/styles.json`：

| 风格 | 说明 |
|---|---|
| `replay`（默认） | 荧光黄绿，回放感 |
| `neon` | 霓虹，颜色按连击数升级（10、20、30 个换一档） |
| `flame` | 烈焰，带火焰粒子 |
| `comet` | 彗星，长拖尾 |
| `broadcast` | 赛事转播风格，细拖尾 + 标签 |

## 布局

`--layout`。HUD 按 390×844 的基准坐标设计，按画布高度等比放大、水平居中；左右边距按画面宽度比例算。

| 布局 | 说明 |
|---|---|
| `v3`（默认） | 左上计数，数据面板贴视频底部、左右铺满 |
| `v3s` | 数据面板抬高、右边收窄，避开短视频 App 的界面（左下两个胶囊、右侧点赞栏） |
| `v3l` | 同上，但只避开一个胶囊，面板更低、少挡腿 |
| `v3c` | 高度曲线放到计数右边，底部只留一行数据 |
| `v2`、`v1` | 早期布局：计数在左上，"最高一颠"和节奏条在左下 / 左侧 |

**安全区**：`fx.js` 的 `SAFE_AREA` 是竖屏短视频 App 推荐页里被界面盖住的区域（占视频宽高的比例），按一台 909×2000 屏的截图量的：
顶部栏到 6%；右侧头像 / 点赞栏从 x 84.5%、y 46% 起到底；左下胶囊从 87%（两个胶囊时 82%）起；文案从 98% 起。不同 App、机型会有出入。
单帧预览加 `:safe` 会把这些区域画成红色，方便检查有没有被挡住。

## 文字可读性

- 分析时顺便统计画面亮度（6×10 格，每 0.5 秒一次，存在轨迹的 `scene`）；
- 渲染时在 HUD 下面垫保护层（`--protect`）：默认 `light` 是上下很淡的渐变 + 文字自带阴影，画面保持清晰；
  `plates`（按背后亮度自动调深浅的柔和底板）、`edges`（上下渐进模糊）、`veil`（整页灰度层）、`mix` 压得更重；
- 文字一律白色，白字对比度目标 ≥ 4.5；
- `node render/check_legibility.mjs out/<名>.track.json` 按真实像素核对；旧的轨迹文件用 `python -m combocam scene out/<名>.track.json` 补算亮度。

## 尺寸

跟球走的效果按球的大小缩放，但以屏宽设上下限：触球冲击波最大不超过屏宽 14%，拖尾最长 6 个球径。近景、远景都不会失控。

## 字体

计数数字默认 Bebas Neue（`--num-font`），底部面板默认思源黑体 + Barlow Condensed 粗斜体数字（`--panel-font`）。
全部是 SIL OFL 1.1 字体，用 `python tools/fetch_assets.py` 下载，见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

## 单帧预览

调效果时不用渲染整段：取一帧，按几种方案各画一张，并排拼成一张图，标出实测的白字对比度。

```bash
node render/still.mjs --track out/x.track.json --t 3.27 --out out/still.png \
  --variants "v3:light:replay,v3s:light:replay:safe,v3:plates:neon"

node render/still.mjs --track out/x.track.json --t 9.5 --end 1.5 --out out/still.png    # 画片尾第 1.5 秒
```

方案写法：`布局:保护层[:风格][:safe][:font=数字字体][:pfont=面板字体][:tint=none|panel|all][:ps=面板字号缩放][:count=N]`。
每个方案另存一张原尺寸图。`--crop 0,0.34` 只显示画面的一段高度，`--cols` 每行几格。

## 在浏览器里复用

`fx.js` 只用标准 Canvas 2D API，不依赖 Node 或 DOM；所有效果都是时间 t 的纯函数（`drawFrame(ctx, tl, t, styleId, video)`），
粒子的随机数用触球序号和时刻做种子。浏览器里可以把视频放在画布下面、`video` 传 `null`，按播放进度每帧调一次 `drawFrame` 做实时预览
（这时不做震屏、不抠球）；贴图缓存需要的离屏画布用 `OffscreenCanvas`。

## 核对画面

渲染是确定的：同一帧、任何线程数，画出来逐字节相同。改渲染代码前后可以这样比对：

```bash
node render/render.mjs --input in.mp4 --track t.json --output before.md5 --encoder md5   # 每帧像素的 MD5
node render/render.mjs --input in.mp4 --track t.json --output frames.mkv --encoder raw   # 无损帧，测画质（30 秒约 8 GB）
node render/profile_parts.mjs in.mp4 t.json                                                # 每帧各部分耗时
```
