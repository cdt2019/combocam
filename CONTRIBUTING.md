# 参与开发

欢迎提 issue 和 pull request。代码注释和文档用中文；issue、PR 用中文或英文都可以。

## 开发环境

```bash
pip install -r requirements.txt -r requirements-dev.txt     # 有 GPU 用 requirements-gpu.txt
python tools/fetch_assets.py
cd render && npm ci && cd ..
```

提交前：

```bash
ruff check .
python -m pytest
```

CI 会跑同样的检查，外加渲染器的语法检查。

## 改计数规则

计数是手写的规则，改一处很容易在别的视频上变差。请：

1. 在相关的标注集上跑一遍评估，把**改前、改后**的召回、精确率写进 PR（命令见 [docs/evaluation.md](docs/evaluation.md#评估命令)）；
2. 在 `tests/` 里加一条能复现这个问题的回归测试（用合成的轨迹或关键点，不依赖视频）；
3. 阈值放进各模块的 `P` 字典，用物理单位（球半径、身高、秒），写清楚依据（哪段视频、量出来是多少）；
4. 不要只看着留出集调：用了哪批的错误来改，就在文档里注明那批的数字"有偏"。

## 改渲染

渲染是确定的。改代码前后各跑一遍逐帧 MD5，确认只有想改的帧变了：

```bash
node render/render.mjs --input in.mp4 --track t.json --output before.md5 --encoder md5
# 改代码
node render/render.mjs --input in.mp4 --track t.json --output after.md5 --encoder md5
diff before.md5 after.md5
```

调效果用单帧预览 `node render/still.mjs`（见 [docs/rendering.md](docs/rendering.md#单帧预览)）。

## 提交测试视频和标注

- 只收能公开再分发、或者能按编号从公开渠道下载的视频（比如 Pexels），不要提交视频文件本身；
- 拍到的人要同意公开；
- 标注按 [docs/evaluation.md](docs/evaluation.md#标注格式) 的格式，先标后跑，第一次跑的结果原样记下来。
