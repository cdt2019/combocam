"""分析完先判断这段视频能不能出片：没拍到球、没拍到人、一个都没颠起来，就不渲染，返回重拍提示。

status 给前端用：title 做标题、message 做正文、tips 做拍摄要点（比如显示一个"识别失败，请重拍"的页面）。
code 取值：ok / no_ball / no_person / no_juggling。
"""

from __future__ import annotations

# 至少颠到这么多个才出片：颠到 1 个就出片，一个都没有才提示重拍。
# 参考：42 段测试素材里最长连击 1–2 的 6 段多半是"在地上玩球"误数出来的，提到 3 能全部挡掉，但会挡掉只会颠一两个的新手。
MIN_COMBO = 1
MIN_BALL_FRAC = 0.15  # 球在多少比例的帧里被找到
MIN_PERSON_FRAC = 0.3  # 人在多少比例的帧里被找到

TIPS = ["手机立在地上，退后两三步", "全身和球都要在框里", "光线亮一点，背景简单些", "对着镜头把球颠起来"]


def verdict(n_frames: int, ball_frames: int, person_frames: int, best_combo: int) -> dict:
    n = max(1, n_frames)
    if ball_frames / n < MIN_BALL_FRAC:
        return _fail("no_ball", "没找到球", "画面里没看清足球。请在光线亮一点的地方拍，球别出画面。")
    if person_frames / n < MIN_PERSON_FRAC:
        return _fail("no_person", "没拍到人", "画面里没找到颠球的人。手机立在地上，退后两三步，让全身和球都在框里。")
    if best_combo == 0:
        return _fail("no_juggling", "没识别到颠球",
                     "这段视频里球没有被颠起来。在地上带球、拨球，或者把球顶在头上平衡都不算颠球，请对着镜头连续颠球再拍一次。")
    if best_combo < MIN_COMBO:  # MIN_COMBO > 1 时才会走到
        return _fail("too_few", f"只颠了 {best_combo} 个", f"至少连续颠 {MIN_COMBO} 个才能生成特效视频，再来一次吧。")
    return {"ok": True, "code": "ok"}


def _fail(code: str, title: str, message: str) -> dict:
    return {"ok": False, "code": code, "title": title, "message": message, "tips": TIPS}
