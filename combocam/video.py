"""FFmpeg 读写。分析和渲染用同一个"画布"滤镜，保证两边坐标一致。"""

from __future__ import annotations

import glob
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


@dataclass
class Probe:
    width: int  # 已按旋转元数据校正
    height: int
    fps: float
    duration: float
    has_audio: bool
    pix_fmt: str = ""


def probe(path: str) -> Probe:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, check=True, text=True, encoding="utf-8",
    ).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(v.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w
    num, den = v.get("avg_frame_rate", "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(info["format"].get("duration") or v.get("duration") or 0)
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    return Probe(w, h, fps, duration, has_audio, v.get("pix_fmt", ""))


def canvas_filter(p: Probe, width: int = 1080, height: int = 1920, fps: float = 30.0) -> str:
    """把任意输入统一成 width×height、恒定帧率的竖版画布。

    接近 9:16 的直接缩放裁边；横屏或其他比例：等比放进画布，背景用同一帧放大模糊垫底。
    """
    fps_part = f"fps={fps:g}"
    src_ar, dst_ar = p.width / p.height, width / height
    if abs(src_ar / dst_ar - 1) < 0.08:  # 例如 2160×4096 只裁掉上下各 3%
        return f"{fps_part},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1"
    return (
        f"{fps_part},split[a][b];"
        f"[a]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=40:2,eq=brightness=-0.15[bg];"
        f"[b]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def content_bottom(src_w: int, src_h: int, width: int = 1080, height: int = 1920) -> float:
    """画布上真实画面的下边缘（y）。横屏等比放进竖版画布时，上下是模糊垫底，人的脚会被这条边截断。"""
    if abs((src_w / src_h) / (width / height) - 1) < 0.08:
        return float(height)
    fh = min(height, width * src_h / src_w)
    return (height + fh) / 2


# ---- 解码帧缓存：分析解码时顺手存一份，渲染直接读，省掉第二次解码 ----
# 存的是按画布帧率取帧后、缩放之前的原始帧（MKV，保留色彩标记），渲染在它上面跑画布滤镜剩下的部分，
# 输入和算法都没变，渲染画面逐帧相同（2026-10-10 在 V100 上用竖屏、横屏、全色域、BT.601/709、4K 素材逐帧 MD5 核对过）。
# 放在内存盘 /dev/shm（1080p 每帧 3.1 MB，30 秒约 2.7 GB），只在 Linux 上做；只缓存 8 bit 4:2:0 的视频。
CACHE_DIR = "/dev/shm"
CACHE_PIX_FMTS = ("yuv420p", "yuvj420p")
_FPS_PART = re.compile(r"^(fps=[^,;]+),")


# 原尺寸缓存只对不太大的视频划算：4K 源（2160×3840）缓存一帧 12 MB，写缓存让分析慢 3 秒，渲染读回来再缩放（FFmpeg 4.4 单线程缩放）
# 比直接解码还慢（V100 云端实测 4K 473 帧：解码 2.7 秒、缩放到画布 4.8 秒，读缓存 + 缩放 8.2 秒）。超过这么多像素就不存原尺寸缓存
CACHE_MAX_PIXELS = 2_500_000


def frame_cache_path(p: Probe, fps: float, size: tuple[int, int] | None = None) -> str | None:
    """够条件就返回一个新的缓存文件路径，否则 None。上限 COMBOCAM_FRAME_CACHE_GB（默认 6，设 0 关闭），还要给内存盘留 2 GB。
    size：缓存的是画布尺寸的帧（FrameReader 的 yuv_canvas），按这个尺寸算大小；None 是原尺寸缓存（太大的源不存，见 CACHE_MAX_PIXELS）。"""
    limit = float(os.environ.get("COMBOCAM_FRAME_CACHE_GB", "6")) * 1e9
    if limit <= 0 or os.name != "posix" or not os.path.isdir(CACHE_DIR):
        return None
    # 原尺寸缓存原样存解码帧，只收 8 位 4:2:0（颜色标记靠 MKV 保留）；画布尺寸的缓存已经统一转成 yuv420p，不挑源格式
    if size is None and (p.pix_fmt not in CACHE_PIX_FMTS or p.width * p.height > CACHE_MAX_PIXELS):
        return None
    need = cache_bytes(p, fps, size=size)
    st = os.statvfs(CACHE_DIR)
    if need > limit or need > st.f_bavail * st.f_frsize - 2e9:
        return None
    # 顺手清掉进程崩溃留下的旧缓存（1 小时以前的）
    for f in glob.glob(os.path.join(CACHE_DIR, "combocam_frames_*.mkv")):
        try:
            if time.time() - os.path.getmtime(f) > 3600:
                os.remove(f)
        except OSError:
            pass
    return os.path.join(CACHE_DIR, f"combocam_frames_{os.getpid()}_{uuid.uuid4().hex[:8]}.mkv")


def cache_bytes(p: Probe, fps: float, frames: int | None = None, size: tuple[int, int] | None = None) -> int:
    n = frames if frames is not None else int(p.duration * fps) + 2
    w, h = size or (p.width, p.height)
    return w * h * 3 // 2 * n


def cache_complete(path: str, p: Probe, frames: int, size: tuple[int, int] | None = None) -> bool:
    """写缓存那一路出错（内存盘写满等）不会影响分析，但文件不完整；按大小核对，不够就当没有。"""
    try:
        return os.path.getsize(path) >= cache_bytes(p, 0, frames, size)
    except OSError:
        return False


def remove_cache(path: str | None) -> None:
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


class FrameReader:
    """后台线程解码，主线程拿 BGR 帧；解码和推理并行。cache 给了路径就同时把解码帧存一份（见 frame_cache_path）。"""

    def __init__(self, path: str, vf: str, width: int, height: int, prefetch: int = 8, cache: str | None = None,
                 yuv_canvas: bool = False):
        """yuv_canvas：画布滤镜（缩放在 yuv 里做）之后输出画布尺寸的 yuv420p，在读帧线程里用 OpenCV 转 BGR 给分析；
        缓存存的也是这些 yuv420p 帧（渲染只需转成 rgba，不用再解码、缩放）。
        比直接从原尺寸一步缩放转 bgr24 快：FFmpeg 4.4 的缩放、转换都是单线程，管道里的数据也少一半
        （V100 云端实测，4K 源 473 帧只读帧 7.4 → 约 5 秒）。像素和不加这一步时略有不同（OpenCV 按 BT.601 转），
        颠球的分析为了计数结果不变不用它；跳绳用（render.mjs 按轨迹的 source.canvas_yuv 走同样的 yuv420p 中间格式）。"""
        self.w, self.h = width, height
        self.yuv = yuv_canvas
        self.frame_bytes = width * height * 3 // 2 if yuv_canvas else width * height * 3
        m = _FPS_PART.match(vf)
        if yuv_canvas:
            if cache:
                graph = f"{vf},format=yuv420p,split=2[mc_main][mc_save]"
                cmd = [FFMPEG, "-v", "error", "-i", path, "-filter_complex", graph, "-map", "[mc_main]", "-map", "[mc_save]",
                       "-c:v", "rawvideo", "-f", "tee", "-use_fifo", "1", "-fifo_options", "queue_size=8",
                       f"[select=0:f=rawvideo]pipe:1|[select=1:f=matroska:onfail=ignore]{cache}"]
            else:
                cmd = [FFMPEG, "-v", "error", "-i", path, "-filter_complex", f"{vf},format=yuv420p", "-f", "rawvideo",
                       "-pix_fmt", "yuv420p", "pipe:1"]
        elif cache and m:
            # fps 之后分两路：一路跑画布滤镜剩下的部分、转 bgr24 给分析（和不缓存时逐字节相同），
            # 一路原样写 MKV。tee 的 onfail=ignore：缓存那一路写失败只丢掉它，分析照常。
            # use_fifo：每一路在自己的线程里写（FFmpeg 4.4 默认都在主线程写，推理变快后它成了瓶颈，单条慢 2 秒）
            graph = f"{m.group(1)},split=2[mc_main][mc_save];[mc_main]{vf[m.end():]}[mc_out]"
            cmd = [FFMPEG, "-v", "error", "-i", path, "-filter_complex", graph, "-map", "[mc_out]", "-map", "[mc_save]",
                   "-c:v", "rawvideo", "-pix_fmt:v:0", "bgr24", "-f", "tee", "-use_fifo", "1", "-fifo_options", "queue_size=8",
                   f"[select=0:f=rawvideo]pipe:1|[select=1:f=matroska:onfail=ignore]{cache}"]
        else:
            cmd = [FFMPEG, "-v", "error", "-i", path, "-filter_complex", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=self.frame_bytes * 2)
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        stdout = self.proc.stdout
        while True:
            buf = stdout.read(self.frame_bytes)
            if len(buf) < self.frame_bytes:
                self.q.put(None)
                return
            if self.yuv:
                self.q.put(cv2.cvtColor(np.frombuffer(buf, np.uint8).reshape(self.h * 3 // 2, self.w), cv2.COLOR_YUV2BGR_I420))
            else:
                self.q.put(np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3))

    def __iter__(self):
        while True:
            f = self.q.get()
            if f is None:
                break
            yield f
        self.proc.wait()

    def close(self) -> None:
        """提前结束（出错时）：停掉 FFmpeg，清空队列让后台线程退出，再放一个结束标记，叫醒还在取帧的线程。"""
        self.proc.kill()
        while self.t.is_alive():
            try:
                self.q.get(timeout=0.1)
            except queue.Empty:
                pass
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self.proc.wait()
