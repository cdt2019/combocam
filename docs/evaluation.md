# 数据集和评估

## 测试视频

全部来自 [Pexels](https://www.pexels.com/)，按 [Pexels 许可](https://www.pexels.com/license/)可以免费用于测试，但不能原样再分发，所以**视频不在仓库里**。
仓库里有：

- `samples/manifest.json`：每段视频的 Pexels 编号、当时下载的版本（宽、高、帧率）、时长、大小、sha256；
- 各目录下的人工标注（`labels*.json`）。

### 下载测试视频

到 <https://www.pexels.com/api/> 免费申请一个 API key：

```bash
export PEXELS_API_KEY=...
python tools/fetch_samples.py --list                      # 各目录的段数和已下载的段数
python tools/fetch_samples.py --set samples/heldout5       # 下一个目录
python tools/fetch_samples.py                             # 全部（约 310 段、3 GB）
```

脚本按编号查 Pexels API，挑和清单里尺寸、帧率一样的版本下载，再核对 sha256。Pexels 偶尔会重新转码，sha256 对不上时只提示；
标注的帧号按 30 fps 画布计（`帧号 = 秒 × 30`），和原视频的帧率、分辨率无关，评估结果最多有细微差别。

### 目录

| 目录 | 项目 | 段数 | 标注 | 用途 |
|---|---|---|---|---|
| `samples/`（根目录） | 颠球 | 4 | `labels.json`，53 次触球 | 最早的开发集 |
| `samples/heldout/` … `heldout5/` | 颠球 | 12 / 11 / 15 / 19 / 19 | `labels.json` | 5 批留出集（见下） |
| `samples/jumprope/` | 跳绳 | 34 | `labels.json`，6 段 73 跳 | 第 1 批：调规则 + 验证 |
| `samples/jumprope2/` | 跳绳 | 40 | `labels.json`，28 段 484 跳 | 第 2 批：验证集 |
| `samples/jumprope3/` | 跳绳 | 40 | 无 | 找"一圈跳两下"的例子，误判检查 |
| `samples/dribble/` | 运球 | 51 | `labels.json`（调参集）、`labels_val.json`（验证集）、`labels_t3.json`、`labels_t4.json`、`moves.json`（动作） | |
| `samples/dribble2/` | 运球 | 54 | 无 | 找背后运球的例子，误判检查 |
| `samples/dribble3/` | 运球 | 10 | `labels.json`，7 段 47 下 | 独立检验 |

另外两段本地生成的视频不在清单里：`samples/bench_30s.mp4`（两段颠球素材拼成的 29 秒视频，测速度用，任何 30 秒左右的颠球视频都可以代替）、
`samples/jumprope_pk/pk_doubles.mp4`（两段双摇视频各裁一半左右拼起来，测 PK 的双摇）。

### 标注格式

所有帧号都按 30 fps 画布计。

**颠球**（`samples/labels.json`、`samples/heldout*/labels.json`），键是文件名：

```json
"pexels_38236117.mp4": {
  "note": "草地，男孩，正常速度……",
  "touches": [24, 39, 54],      // 确定的触球
  "ambiguous": [445],           // 看不清的（球出画面、贴地轻触），算不算都不扣分
  "bounces": [7, 139, 184],     // 落地反弹、从地上挑起（不计数）
  "nocount": [],                // 用手、头上平衡等不算触球的接触
  "ignore": [[100, 160]]        // 整段不评（剪辑切换、两个人传球……）
}
```

`_exclude` 里是不评的视频和理由；`labels.first_run_snapshot.json` 是第一次跑之前的原始标注（之后复查改过的地方写在 `note` 里）。

**跳绳**（`samples/jumprope*/labels.json`），键是视频名：`jumps`（每一跳的帧号，取腾空最高点）、`doubles`（双摇那几跳）、`ignore`、`note`；
`exclude` 里是没法逐跳标的视频和理由。

**运球**（`samples/dribble*/labels*.json`）：`bounces`（球碰地的帧号）、`ignore`；`_exclude` 同上。
`moves.json` 按 `tune` / `val` 分组，每一下的动作：`S` 单手、`C` 交叉、`L` 胯下、`B` 背后、`?` 看不清不算。

## 留出集的规矩

规则是手写的，看着哪段视频调，就会在哪段上变好。所以：

1. 新下一批素材，**先逐帧标注，标的时候不看算法的结果**；
2. 第一次跑的结果就是现在规则的**独立准确率**，原样记下来（颠球的 `labels.first_run_snapshot.json`）；
3. 之后看错误、改规则、复查标注都可以，但这批就不再算留出集，文档里的数字要写清楚"有偏"。

各项目的结果见 [juggling.md](juggling.md#准确率)、[jumprope.md](jumprope.md#准确率)、[dribble.md](dribble.md#准确率)，
每一步改动前后的数字见 [开发记录](development.md)。
最可信的是最后一批第一次跑的数字：颠球 86.7% / 86.7%（第 5 批）、跳绳 99.6% / 98.4%（第 2 批）、运球 97.9% / 100%（第 5 批）。
这些素材都来自图库，下一步要用手机实拍的视频（孩子、各种场地）再做一批。

## 评估命令

先跑模型、把中间结果缓存到 `out/`（要 GPU，几十分钟），之后改计数规则只重算计数，几秒钟：

```bash
# 颠球
python tools/eval/run_heldout.py --set heldout5               # 跑检测和关键点 → out/heldout5/
python tools/eval/eval_heldout.py --set heldout5 -v           # 对标注算召回、精确率，逐条列出误报、漏检
python tools/eval/eval_real.py [--cached]                     # samples/ 根目录那 4 段

# 跳绳
python tools/eval/jumprope_extract.py samples/jumprope2/*.mp4 --out out/jumprope2   # 跑关键点
python tools/eval/jumprope_run.py --dir out/jumprope2 --labels samples/jumprope2/labels.json -v
python tools/eval/jumprope_occlusion.py --seeds 15            # 遮挡压力测试

# 运球
python tools/eval/dribble_extract.py samples/dribble*/*.mp4   # 跑检测和关键点 → out/dribble/
python tools/eval/dribble_eval_all.py -v                       # 所有标注集
python tools/eval/dribble_moves.py --set val -v                # 动作的混淆矩阵

# 回归测试（合成数据，不需要模型和视频）
python -m pytest
```

## 标注工具

| 工具 | 用途 |
|---|---|
| `tools/label/label_helper.py out/heldout5/<名>` | 颠球：列出所有"球的低点"和球丢失的时段，每处画前后几帧的放大图（故意多列，宁多勿漏） |
| `tools/label/set_label.py <名> --touches ... --set heldout5` | 颠球：写标注 |
| `tools/label/jumprope_label.py <名>` | 跳绳：脚部逐帧小图（`--step 4` 给慢动作，`--body` 全身） |
| `tools/label/jumprope_kymo.py <名>` | 跳绳：像素运动轨迹图（人的中线竖带按时间排开，每跳一个拱形），和关键点算法无关 |
| `tools/label/dribble_label.py <名>` | 运球：逐帧小图读"球碰地"的帧号（`--follow` 镜头在动时跟着人裁） |
| `tools/label/dribble_rows.py`、`dribble_zoom.py` | 运球动作：每一下一行小图、放大看几帧 |

调试用的：`tools/debug/inspect_track.py`（轨迹图）、`show_pose.py`（关键点画到帧上）、`jumprope_people.py`（多人入镜的轨迹）、
`jumprope_sheet.py`（成片关键帧小图）、`contact_sheet.py`、`compare_sheet.py`（改前改后对比图）。
