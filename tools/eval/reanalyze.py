"""改了计数/事件逻辑后，用缓存的检测结果重算轨迹 JSON，不重跑检测（几秒一条）。

  python tools/eval/reanalyze.py out/pexels_38236117.track.json [更多 track.json ...]

需要同名的 out/<名>.debug.json（tools/eval/eval_real.py 跑检测时会存）。画面亮度、来源、计时等字段保留原样。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from combocam.analyze import postprocess  # noqa: E402
from combocam.track import Cand  # noqa: E402
from combocam.verdict import verdict  # noqa: E402
from combocam.video import content_bottom  # noqa: E402


def reanalyze(track_path: Path) -> dict:
    dbg_path = track_path.with_name(track_path.name.replace(".track.json", ".debug.json"))
    tr = json.loads(track_path.read_text(encoding="utf-8"))
    dbg = json.loads(dbg_path.read_text(encoding="utf-8"))
    cands = [[Cand(*c) for c in fr] for fr in dbg["cands"]]
    persons = [tuple(p) if p else None for p in dbg["persons"]]
    track, _, ev = postprocess(cands, persons, tr["fps"], tr["height"], dbg.get("sel"), content_bottom=content_bottom(tr["source"]["probe"]["width"], tr["source"]["probe"]["height"], tr["width"], tr["height"]), kpts=dbg.get("kpts"))
    before = [c["count"] for c in tr["combos"]]
    tr.update({
        "ball": ev.ball, "touches": ev.touches, "apex": ev.apex, "combos": ev.combos,
        "highlight": ev.highlight, "bounces": ev.bounces, "calib": ev.calib, "warnings": ev.warnings,
    })
    tr["status"] = verdict(tr["frames"], len(track), sum(1 for p in dbg["persons"] if p),
                           max((c["count"] for c in ev.combos), default=0))
    tr["summary"].update({
        "touches": len(ev.touches),
        "best_combo": max((c["count"] for c in ev.combos), default=0),
        "best_height_m": max((a["height_m"] for a in ev.apex), default=None),
        "track_frames": len(track),
    })
    track_path.write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
    st = "" if tr["status"]["ok"] else f"，不出片：{tr['status']['code']}"
    print(f"{track_path.name}: 连击 {before} → {[c['count'] for c in ev.combos]}，落地 {len(ev.bounces)} 次{st}")
    return tr


if __name__ == "__main__":
    for p in sys.argv[1:]:
        reanalyze(Path(p))
