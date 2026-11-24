"""画面亮度统计：给特效判断"背后有多亮"，决定压暗多少，保证文字看得清。

画布切成 COLS×ROWS 的格子，每帧算每格的相对亮度（WCAG 定义，0–1）的 75 分位数，
再按 STEP 秒一个窗口取最大值（保守：窗口里最亮的时候也要看得清）。
存成 track["scene"]，渲染器（和实时预览）按 HUD 所在的格子取值，和 HUD 布局解耦。
"""

from __future__ import annotations

import subprocess

import numpy as np

from .video import FFMPEG

COLS, ROWS, STEP = 6, 10, 0.5
DS = 4  # 统计用的降采样倍数

_v = np.arange(256) / 255.0
_LIN = np.where(_v <= 0.04045, _v / 12.92, ((_v + 0.055) / 1.055) ** 2.4)


def grid_lum(small_bgr: np.ndarray) -> np.ndarray:
    """small_bgr：降采样后的 BGR 帧。返回 ROWS×COLS 的亮度（75 分位）。"""
    lum = 0.0722 * _LIN[small_bgr[..., 0]] + 0.7152 * _LIN[small_bgr[..., 1]] + 0.2126 * _LIN[small_bgr[..., 2]]
    h, w = lum.shape
    if h % ROWS == 0 and w % COLS == 0:
        # 格子一样大时一次算完 60 个格子（分位数只和格子里有哪些值有关，和逐格算结果逐位相同，快 5 倍左右）
        cells = lum.reshape(ROWS, h // ROWS, COLS, w // COLS).transpose(0, 2, 1, 3).reshape(ROWS, COLS, -1)
        return np.percentile(cells, 75, axis=-1)
    out = np.empty((ROWS, COLS))
    for r in range(ROWS):
        for c in range(COLS):
            cell = lum[r * h // ROWS:(r + 1) * h // ROWS, c * w // COLS:(c + 1) * w // COLS]
            out[r, c] = np.percentile(cell, 75)
    return out


class SceneStats:
    def __init__(self, fps: float, every: int = 3):
        self.per_window = max(1, int(round(STEP * fps)))
        self.every = every  # 每 3 帧统计一次，窗口内取最大值，够用且省时间
        self.windows: list[np.ndarray] = []
        self._cur: np.ndarray | None = None
        self._n = 0

    def add(self, frame_bgr: np.ndarray, already_small: bool = False) -> None:
        if self._n % self.every == 0:
            g = grid_lum(frame_bgr if already_small else frame_bgr[::DS, ::DS])
            self._cur = g if self._cur is None else np.maximum(self._cur, g)
        self._n += 1
        if self._n == self.per_window:
            self.windows.append(self._cur)
            self._cur, self._n = None, 0

    def result(self) -> dict:
        wins = self.windows + ([self._cur] if self._cur is not None else [])
        return {
            "step": STEP, "cols": COLS, "rows": ROWS,
            "lum": [[round(float(v), 3) for v in w.ravel()] for w in wins],  # 每个窗口按行展开
        }


def scene_from_video(src: str, vf: str, width: int, height: int, fps: float) -> dict:
    """单独补算（给没有 scene 的旧轨迹用）：低分辨率解码，很快。"""
    w, h = width // DS, height // DS
    proc = subprocess.Popen([FFMPEG, "-v", "error", "-i", src, "-filter_complex", f"{vf},scale={w}:{h}",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
    st = SceneStats(fps)
    nb = w * h * 3
    while True:
        buf = proc.stdout.read(nb)
        if len(buf) < nb:
            break
        st.add(np.frombuffer(buf, np.uint8).reshape(h, w, 3), already_small=True)
    proc.wait()
    return st.result()
