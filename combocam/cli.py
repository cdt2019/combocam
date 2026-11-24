"""命令行入口。

  python -m combocam analyze  IN.mp4 -o out/track.json
  python -m combocam render   IN.mp4 out/track.json -o out/final.mp4 --style replay
  python -m combocam run      IN.mp4 -o out/            # 分析 + 渲染 + 计时报告
  python -m combocam jumprope IN.mp4 -o out/            # 跳绳：关键点 → 计数 → 特效成片
  python -m combocam dribble  IN.mp4 -o out/            # 篮球运球：球的轨迹 → 触地计数 → 特效成片
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RENDER_JS = ROOT / "render" / "render.mjs"


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _pose(a) -> str | None:
    return None if a.pose_model.lower() == "none" else a.pose_model


def cmd_analyze(a) -> dict:
    from .analyze import analyze

    out = analyze(a.input, model=a.model, device=a.device, refine=a.refine, fps=a.fps, pose_model=_pose(a))
    _dump(out, Path(a.output))
    print(f"[analyze] → {a.output}", file=sys.stderr)
    _exit_if_rejected(out["status"])
    return out


def render(input_path: str, track_path: str, output: str, style: str, endcard: str, nickname: str,
           workers: int, slowmo: bool, encoder: str, look: dict | None = None, preset: str = "", frames: str | None = None) -> dict:
    cmd = ["node", str(RENDER_JS), "--input", input_path, "--track", track_path, "--output", output,
           "--style", style, "--endcard", endcard, "--nickname", nickname, "--workers", str(workers), "--encoder", encoder]
    if preset:
        cmd += ["--preset", preset]
    if frames:
        cmd += ["--frames", frames]  # 分析时存下的解码帧，渲染不用再解码一遍
    for k, v in (look or {}).items():
        if v:
            cmd += [f"--{k}", v]
    if slowmo:
        cmd.append("--slowmo")
    t = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"渲染失败（exit {proc.returncode}）")
    stats = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
    stats["wall_s"] = round(time.perf_counter() - t, 2)
    return stats


def _look(a) -> dict:
    return {"layout": a.layout, "protect": a.protect, "num-font": a.num_font, "tint": a.tint, "panel-font": a.panel_font,
            "endcard-text": a.endcard_text}


# 视频不能出片（没找到球、没在颠球……）时的退出码；调用方（比如服务端）按它读轨迹里的 status，提示用户重拍
EXIT_REJECTED = 3


def _exit_if_rejected(status: dict | None) -> None:
    if status and not status["ok"]:
        print(f"[combocam] 不出片（{status['code']}）：{status['title']}。{status['message']}", file=sys.stderr)
        print(json.dumps({"status": status}, ensure_ascii=False))
        raise SystemExit(EXIT_REJECTED)


def cmd_render(a) -> dict:
    if not a.force:
        _exit_if_rejected(json.loads(Path(a.track).read_text(encoding="utf-8")).get("status"))
    stats = render(a.input, a.track, a.output, a.style, a.endcard, a.nickname, a.workers, a.slowmo, a.encoder, _look(a), a.preset)
    print(json.dumps(stats, ensure_ascii=False))
    return stats


def cmd_run(a) -> None:
    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.input).stem
    track_path = out_dir / f"{stem}.track.json"
    t0 = time.perf_counter()
    from .analyze import analyze
    from .video import remove_cache

    # 分析解码时顺手把帧存进内存盘，渲染直接读（见 video.frame_cache_path）；不管成败，最后都删掉
    cache: dict = {} if a.frame_cache else None
    try:
        tr = analyze(a.input, model=a.model, device=a.device, refine=a.refine, fps=a.fps, pose_model=_pose(a), frame_cache=cache)
        _dump(tr, track_path)
        t1 = time.perf_counter()
        if not tr["status"]["ok"] and not a.force:
            # 不出片：只写报告（含重拍提示），不渲染
            report = {"input": str(Path(a.input).resolve()), "status": tr["status"], "video_duration_s": tr["duration"],
                      "analysis": {**tr["timing"], "model": tr["meta"]["model"], "device": tr["meta"]["device"]},
                      "summary": tr["summary"], "warnings": tr["warnings"]}
            _dump(report, out_dir / f"{stem}.report.json")
            _exit_if_rejected(tr["status"])
        frames = (cache or {}).get("path")
        videos = {}
        for style in a.style.split(","):
            out_mp4 = out_dir / f"{stem}.{style}.mp4"
            videos[style] = render(a.input, str(track_path), str(out_mp4), style, a.endcard, a.nickname, a.workers, a.slowmo,
                                   a.encoder, _look(a), a.preset, frames)
            print(f"[run] → {out_mp4}", file=sys.stderr)
    finally:
        remove_cache((cache or {}).get("path"))
    t2 = time.perf_counter()
    report = {
        "input": str(Path(a.input).resolve()),
        "video_duration_s": tr["duration"],
        "analysis": {**tr["timing"], "model": tr["meta"]["model"], "device": tr["meta"]["device"]},
        "render": videos,
        "wall_s": {"analyze": round(t1 - t0, 2), "render": round(t2 - t1, 2), "total": round(t2 - t0, 2)},
        "frame_cache": bool(frames),
        "status": tr["status"],
        "summary": tr["summary"],
        "warnings": tr["warnings"],
    }
    _dump(report, out_dir / f"{stem}.report.json")
    print(json.dumps(report["wall_s"]), file=sys.stderr)


def cmd_jumprope(a) -> None:
    """跳绳：逐帧关键点 → 计数 → 轨迹 JSON（sport = jumprope）→ 渲染（render.mjs 自动换成跳绳特效）。
    同 run：关键点那一遍解码时顺手把帧存进内存盘，渲染直接读，最后删掉。"""
    from .video import remove_cache

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.input).stem
    t0 = time.perf_counter()
    cache: dict = {} if a.frame_cache and a.render else None
    try:
        _jumprope(a, out_dir, stem, t0, cache)
    finally:
        remove_cache((cache or {}).get("path"))


def _jumprope(a, out_dir: Path, stem: str, t0: float, cache: dict | None) -> None:
    from .jumprope_track import analyze_jumprope

    tr, pose_data = analyze_jumprope(a.input, a.model, a.pose_model, a.device, a.fps, frame_cache=cache, mode=a.mode,
                                     pk_rule=a.pk_rule)
    track_path = out_dir / f"{stem}.jumprope.json"
    _dump(tr, track_path)
    if a.save_pose:
        _dump(pose_data, out_dir / f"{stem}.pose.json")
    t1 = time.perf_counter()
    sm = tr["summary"]
    if tr["mode"] == "pk":
        win = {None: "平局", 0: "左边赢", 1: "右边赢"}[sm["winner"]]
        rule = {"jumps": "比总跳数", "doubles": "比双摇", "passes": "比过绳次数"}[sm["rule"]]
        head = (f"双人 PK（{rule}）：左 {sm['totals'][0]} 跳（双摇 {sm['doubles'][0]}）、右 {sm['totals'][1]} 跳（双摇 {sm['doubles'][1]}），"
                f"得分 {sm['scores'][0]} : {sm['scores'][1]}，{win}，领先换了 {sm['lead_changes']} 次；"
                + "；".join(f"{'左' if pl['side'] == 'left' else '右'}跳速 {pl['summary']['bpm'] or '—'} 次/分" for pl in tr["players"]))
    else:
        head = f"{sm['total']} 跳，{sm['groups']} 组，最长连跳 {sm['best']}，断绳 {sm['breaks']} 次，跳速 {sm['bpm'] or '—'} 次/分"
    print(f"[jumprope] {stem}: {head}；关键点 {tr['timing']['extract_s']}s" + "".join(f"；{w}" for w in tr["warnings"]), file=sys.stderr)
    report = {"input": str(Path(a.input).resolve()), "sport": "jumprope", "mode": tr["mode"], "video_duration_s": tr["duration"], "status": tr["status"],
              "summary": sm, "warnings": tr["warnings"], "analysis": {**tr["timing"], **tr["meta"]}}
    if not tr["status"]["ok"] and not a.force:
        _dump(report, out_dir / f"{stem}.report.json")
        _exit_if_rejected(tr["status"])
    videos = {}
    if a.render:
        for style in a.style.split(","):
            out_mp4 = out_dir / f"{stem}.jumprope.{style}.mp4"
            videos[style] = render(a.input, str(track_path), str(out_mp4), style, a.endcard, a.nickname, a.workers, False,
                                   a.encoder, _look(a), a.preset, (cache or {}).get("path"))
            print(f"[jumprope] → {out_mp4}", file=sys.stderr)
    t2 = time.perf_counter()
    report.update({"render": videos, "frame_cache": bool((cache or {}).get("path")), "wall_s": {"analyze": round(t1 - t0, 2), "render": round(t2 - t1, 2), "total": round(t2 - t0, 2)}})
    _dump(report, out_dir / f"{stem}.report.json")
    print(json.dumps(report["wall_s"]), file=sys.stderr)


def cmd_dribble(a) -> None:
    """篮球运球：颠球的分析管线（球、人、关键点）→ 触地计数（combocam/dribble.py）→ 颠球格式的轨迹（sport = dribble）→ 渲染。"""
    from .dribble_track import analyze_dribble
    from .video import remove_cache

    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.input).stem
    t0 = time.perf_counter()
    cache: dict = {} if a.frame_cache and a.render else None
    try:
        tr, d = analyze_dribble(a.input, a.model, _pose(a), a.device, a.fps, frame_cache=cache)
        track_path = out_dir / f"{stem}.dribble.json"
        _dump(tr, track_path)
        if a.save_cache:
            _dump(d, out_dir / f"{stem}.ball.json")
        t1 = time.perf_counter()
        sm = tr["summary"]
        print(f"[dribble] {stem}: 运球 {sm['total']} 下，{sm['groups']} 组，最长连续 {sm['best']} 下，节奏 {sm['rate'] or '—'} 次/分"
              f"（球 {sm['ball_frames']} 帧，漏检补 {sm['inferred']} 下）", file=sys.stderr)
        report = {"input": str(Path(a.input).resolve()), "sport": "dribble", "video_duration_s": tr["duration"], "status": tr["status"],
                  "summary": sm, "analysis": tr["timing"]}
        if not tr["status"]["ok"] and not a.force:
            _dump(report, out_dir / f"{stem}.report.json")
            _exit_if_rejected(tr["status"])
        videos = {}
        if a.render:
            for style in a.style.split(","):
                out_mp4 = out_dir / f"{stem}.dribble.{style}.mp4"
                videos[style] = render(a.input, str(track_path), str(out_mp4), style, a.endcard, a.nickname, a.workers, False,
                                       a.encoder, _look(a), a.preset, (cache or {}).get("path"))
                print(f"[dribble] → {out_mp4}", file=sys.stderr)
    finally:
        remove_cache((cache or {}).get("path"))
    t2 = time.perf_counter()
    report.update({"render": videos, "wall_s": {"analyze": round(t1 - t0, 2), "render": round(t2 - t1, 2), "total": round(t2 - t0, 2)}})
    _dump(report, out_dir / f"{stem}.report.json")
    print(json.dumps(report["wall_s"]), file=sys.stderr)


def cmd_scene(a) -> None:
    """给旧的轨迹 JSON 补上画面亮度统计（不重跑检测）。"""
    from .scene import scene_from_video

    path = Path(a.track)
    tr = json.loads(path.read_text(encoding="utf-8"))
    tr["scene"] = scene_from_video(tr["source"]["path"], tr["source"]["filter"], tr["width"], tr["height"], tr["fps"])
    _dump(tr, path)
    print(f"[scene] {path.name}: {len(tr['scene']['lum'])} 个窗口", file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="combocam")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scene", help="给旧的轨迹 JSON 补画面亮度统计")
    p.add_argument("track")
    p.set_defaults(fn=cmd_scene)

    def common_analyze(p):
        p.add_argument("--model", default=str(ROOT / "models" / "yolox_s.onnx"))
        p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
        p.add_argument("--refine", default="online", choices=["online", "offline", "off"],
                       help="局部复检：online=第 1 遍里顺带做（默认）；offline=再解码一遍；off=不做（更快，漏检更多）")
        p.add_argument("--fps", type=float, default=30.0)
        p.add_argument("--pose-model", default=str(ROOT / "models" / "rtmpose-m_halpe26_256x192.onnx"),
                       help="人体关键点模型（判断落地、手接球）；none=不用，退回只看轨迹和人框的旧规则")

    def common_render(p):
        p.add_argument("--endcard", default="card", choices=["card", "none"], help="片尾成绩卡：card=有（默认，3 秒）；none=不要")
        p.add_argument("--endcard-text", default="", help="片尾最下面加一行字（比如账号名、话题），默认不加")
        p.add_argument("--nickname", default="", help="片尾顶部显示的昵称（@昵称）")
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
        p.add_argument("--slowmo", action="store_true", help="高光段 0.5 倍慢放（会去掉原声）")
        p.add_argument("--encoder", default="libx264", help="libx264 / h264_nvenc")
        p.add_argument("--preset", default="", help="编码速度档位，默认 libx264 用 veryfast、h264_nvenc 用 p5；superfast 快约 3 成，文件大约 4 成")
        p.add_argument("--force", action="store_true", help="没识别到颠球也照样渲染（调试用）")
        p.add_argument("--layout", default="", choices=["", "v3", "v3s", "v3l", "v3c", "v2", "v1"],
                       help="HUD 布局，默认 v3（数据面板贴视频底部）；v3s/v3l=面板抬高，避开短视频 App 的界面；v3c=曲线放到计数右边；v1/v2=早期布局")
        p.add_argument("--protect", default="", choices=["", "light", "plates", "edges", "veil", "mix"],
                       help="文字保护层，默认 light（上下轻渐变 + 文字阴影）")
        p.add_argument("--tint", default="", choices=["", "panel", "all", "none"],
                       help="文字颜色，默认 panel（底部面板的字用曲线的颜色）；all=计数也换；none=白字")
        p.add_argument("--panel-font", default="", choices=["", "noto", "smiley", "zcool", "barlowc", "inter"],
                       help="底部面板字体，默认 barlowc（思源黑体 + Barlow Condensed 粗斜体数字）")
        p.add_argument("--num-font", default="", choices=["", "bebas", "inter", "noto", "barlow", "poppins", "anton"],
                       help="计数数字字体，默认 bebas（Bebas Neue）")

    p = sub.add_parser("analyze")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    common_analyze(p)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("render")
    p.add_argument("input")
    p.add_argument("track")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--style", default="replay", choices=["replay", "neon", "flame", "comet", "broadcast"])
    common_render(p)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("run")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True, help="输出目录")
    p.add_argument("--style", default="replay", help="逗号分隔，可多个：replay,neon,flame,comet,broadcast")
    p.add_argument("--no-frame-cache", dest="frame_cache", action="store_false",
                   help="不把解码帧存进内存盘给渲染用（默认存，Linux 上省掉渲染的第二次解码；上限见 COMBOCAM_FRAME_CACHE_GB）")
    common_analyze(p)
    common_render(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("jumprope", help="跳绳计数 + 特效成片")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True, help="输出目录")
    p.add_argument("--style", default="replay", help="逗号分隔，可多个：replay,neon,flame,comet,broadcast")
    p.add_argument("--mode", default="single", choices=["single", "pk", "auto"],
                   help="single 单人；pk 双人 PK（挑两个在跳的人，左红右蓝）；auto 两人明显在一起跳时出 PK")
    p.add_argument("--pk-rule", default="auto", choices=["auto", "jumps", "doubles", "passes"],
                   help="双人 PK 怎么比：auto 两人都以双摇为主时比双摇个数、否则比总跳数；jumps 总跳数；doubles 双摇个数；passes 绳子过脚次数（双摇算 2）")
    p.add_argument("--no-render", dest="render", action="store_false", help="只计数、写轨迹 JSON，不出片")
    p.add_argument("--save-pose", action="store_true", help="另存逐帧关键点（tools/eval/jumprope_run.py 调规则用）")
    p.add_argument("--no-frame-cache", dest="frame_cache", action="store_false", help="不把解码帧存进内存盘给渲染用")
    common_analyze(p)
    common_render(p)
    p.set_defaults(fn=cmd_jumprope)

    p = sub.add_parser("dribble", help="篮球运球计数 + 特效成片")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True, help="输出目录")
    p.add_argument("--style", default="replay", help="逗号分隔，可多个：replay,neon,flame,comet,broadcast")
    p.add_argument("--no-render", dest="render", action="store_false", help="只计数、写轨迹 JSON，不出片")
    p.add_argument("--save-cache", action="store_true", help="另存球的候选和关键点（tools/eval/dribble_run.py 调规则用）")
    p.add_argument("--no-frame-cache", dest="frame_cache", action="store_false", help="不把解码帧存进内存盘给渲染用")
    common_analyze(p)
    common_render(p)
    p.set_defaults(fn=cmd_dribble)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
