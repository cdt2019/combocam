# 轨迹 JSON 格式

分析的结果是一个 JSON 文件，渲染器只读它和原视频。三个项目共用一套外壳，`sport` 字段区分（颠球没有这个字段，按 `juggling` 处理）。

- 坐标单位是**画布像素**：原视频先统一成 `width × height`（默认 1080×1920）的竖版画布，所有 x、y、r 都在这个画布上。
- 时间单位是**秒**（原视频时间），帧号按画布帧率 `fps`（默认 30）计，`帧号 = 秒 × fps`。
- `version` 目前是 1。

## 公共字段

| 字段 | 说明 |
|---|---|
| `sport` | `juggling`（颠球，省略时）/ `jumprope` / `dribble` |
| `width`, `height`, `fps`, `frames`, `duration` | 画布尺寸、帧率、帧数、时长 |
| `status` | `{ok, code, title, message, tips?}`，能不能出片，见 [usage.md](usage.md#不出片) |
| `summary` | 计数摘要，各项目不同（见下） |
| `warnings` | 给人看的提示（字符串数组） |
| `source` | `{path, probe: {width, height, fps, duration, has_audio}, filter}`：原视频和统一成画布用的 FFmpeg 滤镜；跳绳另有 `canvas_yuv` |
| `scene` | 画面亮度统计 `{step, cols, rows, lum}`：每 `step` 秒一组，画布切成 `cols × rows` 格，每格亮度（0–1，窗口内的最大值），渲染时决定文字底下压暗多少 |
| `meta` | 分析器版本、模型、设备 |
| `timing` | 各步用时（秒） |

## 颠球（`juggling`）

| 字段 | 说明 |
|---|---|
| `ball` | 球的轨迹，每项 `[帧号, x, y, r]`（球心和半径）。是拟合后的连续轨迹，被身体挡住的帧按抛物线补上 |
| `touches` | 每次触球 `{t, x, y, part, n, combo}`：`part` 是触球部位 `foot` / `thigh` / `head` / `chest`（没有关键点时按人框高度猜，可能是 `unknown`）；`n` 这组第几个；`combo` 第几组 |
| `apex` | 每两次触球之间的最高点 `{t, x, y, height_m, touch, combo}`：`height_m` 球离地高度（米），`touch` 是它前面那次触球的序号 |
| `combos` | 每组连击 `{from, to, count}`；球落地、用手接住就断开 |
| `bounces` | 不算触球的接触 `{t, x, y, kind}`：`kind` 是 `ground`（落地）/ `hand`（手接、手托）/ `stall`（头上平衡、颈后停球） |
| `highlight` | 高光段 `{from, to}`：最高的那一颠，从它前面那次触球前 0.4 秒到下一次触球后 0.4 秒（`--slowmo` 慢放这段） |
| `calib` | 比例尺 `{ball_radius_px, px_per_m, px_per_m_gravity, ground_y}`：按 5 号球直径 22 cm 和按重力加速度各算一个"像素/米"，差太多时 `warnings` 提示高度不可靠；`ground_y` 地面线 |
| `summary` | `{touches, best_combo, best_height_m, ball_detected_frames, track_frames}` |

## 篮球运球（`dribble`）

和颠球同一个格式（渲染器原样用 `render/fx.js` 画），区别：

| 字段 | 说明 |
|---|---|
| `ball` | 选中的原始检测 `[帧号, x, y, r]`，短漏检线性补上，**不**做抛物线平滑（会把触地的尖角磨圆） |
| `touches` | 每次触地 `{t, x, y, part: "floor", n, combo, move, hand_from, hand_to}`：`move` 是 `same`（单手）/ `cross`（体前交叉）/ `legs`（胯下）/ `behind`（背后）/ `null`（不面对镜头时不判）；`hand_from`、`hand_to` 是 `L` / `R` |
| `apex` | 两次触地之间的最高点，`height_m` 按篮球半径 0.12 m 换算 |
| `combos` | 每组连续运球 `{from, to, count}` |
| `dribbles` | 每一下 `{frame, t, x, y, n, group, inferred}`：`inferred` 表示触地那几帧没检测到、由前后两段直线的交点推出来的 |
| `labels` | 画面文字（"连续运球"、"N · 运球"、片尾挑战语……），渲染器用它替换颠球的文字；`labels.moves` 有名字的动作才在画面上标出来 |
| `summary` | `{total, best, groups, period_s, rate, ball_frames, inferred}`：`rate` 每分钟运球次数 |

## 跳绳（`jumprope`）

单人（`mode = "single"`）：

| 字段 | 说明 |
|---|---|
| `jumps` | 每一跳 `{t, frame, n, group, total, up_t, land_t, passes, x, y, y_up, inferred?}`：`n` 这组第几跳、`total` 总第几跳；`up_t` / `land_t` 离地、落地时刻；`passes` 绳子过脚几次（1 单摇、2 双摇、3 三摇）；`x` 两脚中间，`y` / `y_up` 落地、离地那一刻的脚底；`inferred` 表示人被挡住时按节拍补上的跳（没有位置） |
| `groups` | 每组连跳 `{from, to, count, bpm}` |
| `breaks` | 断绳 `{t, count, x, y}`：`count` 是断之前这组跳了几个 |
| `absent` | 人离开画面的时段 `{from, to}`：两组之间人有一半以上时间不在画面里，算离开、不算断绳 |
| `body` | 每帧 `[脚的中点 x, 地面线 y, 人框左 x, 人框右 x, 两脚离地高度 ×1000, 髋部起伏 ×1000, 头顶 y]`（高度以身高为单位），没人时是 `null` |
| `people` | 画面里每个人（连成轨迹后）各数到几跳，第一条是选中的人 |
| `pk_candidate` | 画面里有两个人在一起跳、可以出 PK 版 |
| `summary` | 见下 |

`summary`：

| 字段 | 说明 |
|---|---|
| `total`, `best`, `groups`, `breaks` | 总跳数、最长连跳、组数、断绳次数 |
| `period_s`, `bpm`, `real_s` | 一跳的周期、跳速（次/分）、从第一跳到最后一跳的时长；慢动作视频已换回真实时间 |
| `slowmo`, `speed`, `apparent_g`, `g_fits` | 是不是慢动作、慢放倍数、视在重力加速度（m/s²）和参与拟合的腾空段数，见 [jumprope.md](jumprope.md#慢动作) |
| `arm_sync` | 摇绳节奏核对：手腕在"一跳的间隔"上的自相关，数对了接近 1，看不清手时是 `null` 或接近 0 |
| `feet_visible` | 脚在不在画面里；不在时只看髋部计数，成片不画脚下的效果 |
| `body_px` | 人的高度（像素） |
| `doubles`, `triples`, `passes`, `style` | 双摇、三摇个数，绳子过脚总次数，`single` / `mixed` / `double` |
| `bounce` | `double` 表示判成了"一圈跳两下"，只数过绳的那一下 |

双人 PK（`mode = "pk"`）：

| 字段 | 说明 |
|---|---|
| `players` | 左右两人，各自是上面单人的 `jumps`、`groups`、`breaks`、`body`、`summary` 等字段，加 `side`（`left` / `right`） |
| `summary` | `{totals, doubles, rule, scores, winner, lead_changes}`：`rule` 是比法（`jumps` / `doubles` / `passes`），`winner` 是 0（左）/ 1（右）/ `null`（平局），`lead_changes` 领先换了几次 |

## 报告（`<名>.report.json`）

`combocam run` / `jumprope` / `dribble` 另写一份报告：输入路径、`status`、`summary`、`warnings`，加上分析和渲染各步用时
（`analysis`、`render`、`wall_s`）。渲染部分的 `worker_s` 是工作线程各步合计，`legibility` 是实测的文字对比度。
