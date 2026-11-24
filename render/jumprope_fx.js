// 跳绳特效：外观和颠球（fx.js）同一套，包括 v3 布局、轻渐变保护层、Bebas 数字、底部数据面板和片尾成绩卡。
// 内容换成跳绳的：
// - 左上：总数，每跳弹一下、+1；断绳时弹出提示，总数不清零（跳绳按总数计，和一分钟跳绳测试一样）；
// - 脚下：腾空时地面亮一道光，落地时一圈波纹加扬起的火花；身旁"N · 跳"标签；
// - 底部面板：髋部起伏曲线（每跳一个波峰，刻度标出每一跳和断绳），数据行是跳速、用时、断绳；
// - 里程碑横幅"50 个！"，片尾成绩卡。
// 约定同 fx.js：所有效果都是时间 t 的纯函数，坐标是画布像素。轨迹格式见 combocam/jumprope_track.py。
// 双人 PK（轨迹 mode = "pk"）见 jumprope_pk_fx.js，脚下的效果从这里借。

import {
  DEFAULT_LOOK, DESIGN_W, DROP_COLOR, EDGE, HALO, LAYOUTS, PROTECT, TAU, V3,
  clamp, cjkFont, drawBanner, drawEndBackground, drawProtection, drawTagline, hash01, labV3, look, numFont, panelInk, panelType,
  rgba, rr, shakeAt, subV3, text, textWidth, v3Geom,
} from './fx.js';

// 可读性统计、保护层查表只和布局有关，直接用颠球的
export { legibility, lightScrimLUT } from './fx.js';

const BANNER = (n) => `${n} 个！`;
const SIDE_FEET = 0.76; // 脚（地面线中位数）低于这个高度比例就用 jrside 布局

export function buildTimeline(track, styles, makeCanvas, opts = {}) {
  const W = track.width, H = track.height, n = track.frames;
  const u = H / 844;
  const jr = styles.jumprope || {};
  const jumps = track.jumps.slice().sort((a, b) => a.t - b.t);
  const S = track.summary || {};
  // 双摇（passes ≥ 2，见 combocam/jumprope.rope_passes）：dtotal 到这一跳为止的双摇数，lastDoubleT 最近一次双摇的时刻
  let dc = 0, lastD = -10;
  for (const j of jumps) {
    j.passes = j.passes || 1;
    if (j.passes >= 2) { dc++; lastD = j.t; }
    j.dtotal = dc;
    j.lastDoubleT = lastD;
  }
  const doubleMode = S.style === 'double'; // 一半以上是双摇：按双摇计数
  const tl = {
    sport: 'jumprope', fps: track.fps, W, H, n, u, ox: W / 2 - (DESIGN_W / 2) * u,
    rBall: 0.03 * W, // render.mjs 按它开抠图用的小画布；跳绳不抠图
    sz: { px: W / 1080 },
    bodyPx: clamp(S.body_px || 0.45 * H, 0.15 * H, 1.2 * H),
    jumps, groups: (track.groups || []).slice().sort((a, b) => a.from - b.from),
    breaks: (track.breaks || []).slice().sort((a, b) => a.t - b.t),
    // 横幅（fx.drawBanner）和震屏按 touches 的 t、n 找里程碑：这里 n 用总数（双摇模式用双摇数）
    touches: doubleMode ? jumps.filter((j) => j.passes >= 2).map((j) => ({ t: j.t, n: j.dtotal })) : jumps.map((j) => ({ t: j.t, n: j.total })),
    styles: { ...styles, milestones: jr.milestones || styles.milestones },
    bannerText: doubleMode ? (n) => `双摇 ${n} 个！` : BANNER,
    doubleMode,
    summary: S, speed: S.speed || 1, // 慢放倍数：跳速、用时按它换回真实时间（见 combocam/jumprope_track.py）
    feet: S.feet_visible !== false, // 脚被画面底边切掉时不画脚下的效果和身旁标签（位置是猜的）
    scene: track.scene || null, duration: n / track.fps,
    opts: { ...DEFAULT_LOOK, ...Object.fromEntries(Object.entries(opts).filter(([, v]) => v != null && v !== '')), showCount: opts.showCount ?? null, panelScale: opts.panelScale || null },
    makeCanvas: makeCanvas || null, cache: new Map(), measureCtx: null,
  };
  // 每帧的锚点（见 combocam/jumprope_track.py 的 body）
  const cols = ['footX', 'groundY', 'boxX1', 'boxX2', 'air', 'bounce', 'headY'];
  for (const c of cols) tl[c] = new Float32Array(n).fill(NaN);
  (track.body || []).forEach((b, f) => {
    if (!b || f >= n) return;
    cols.forEach((c, k) => { if (b[k] != null) tl[c][f] = c === 'air' || c === 'bounce' ? b[k] / 1000 : b[k]; });
  });
  // 没指定布局时按脚的位置选：脚在 76% 高度以上（多是横屏放进竖版、下面有空）用底部面板，
  // 再往下（竖拍的全身，脚常在 84%–96%）底部面板会盖住脚下的特效，改成 jrside
  if (!opts.layout) {
    const gs = Array.from(tl.groundY).filter((v) => !Number.isNaN(v)).sort((a, b) => a - b);
    const g = gs.length ? gs[gs.length >> 1] / H : 1;
    tl.opts.layout = g <= SIDE_FEET ? 'v3' : 'jrside';
  }
  if (!LAYOUTS[tl.opts.layout] || !LAYOUTS[tl.opts.layout].v3 || !PROTECT[tl.opts.protect]) {
    throw new Error(`跳绳特效只支持 v3 系列布局：${JSON.stringify(tl.opts)}`);
  }
  if (makeCanvas) tl.measureCtx = makeCanvas(8, 8).getContext('2d');
  return tl;
}

// 某一列在 t 时刻的值（相邻两帧线性插值；缺一帧就取近的那帧）
export function at(tl, arr, t) {
  const fr = t * tl.fps, f0 = Math.floor(fr), a = fr - f0;
  if (f0 < 0 || f0 >= tl.n) return NaN;
  const f1 = Math.min(f0 + 1, tl.n - 1);
  const v0 = arr[f0], v1 = arr[f1];
  if (Number.isNaN(v0)) return a >= 0.5 ? v1 : NaN;
  if (Number.isNaN(v1)) return a < 0.5 ? v0 : NaN;
  return v0 + (v1 - v0) * a;
}

export function lastJumpIndex(tl, t) {
  let lo = 0, hi = tl.jumps.length - 1, ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (tl.jumps[m].t <= t) { ans = m; lo = m + 1; } else hi = m - 1;
  }
  return ans;
}

/** t 时刻的计数状态。count 和 total 都是总数（render.mjs 用 count 判断"开始跳了没有"）。 */
export function stateAt(tl, t) {
  let nBreaks = 0, brk = null;
  const i = lastJumpIndex(tl, t);
  const j = i >= 0 ? tl.jumps[i] : null;
  for (const b of tl.breaks) if (b.t <= t) { nBreaks++; if (j && b.t > j.t) brk = b; }
  if (!j) return { i, count: 0, total: 0, streak: 0, group: -1, lastT: -10, bestStreak: 0, breaks: 0, breakT: null, breakCount: 0, dcount: 0, lastDoubleT: -10, lastPasses: 1 };
  let bestStreak = j.n;
  for (const g of tl.groups) if (g.to < j.t - 1e-6) bestStreak = Math.max(bestStreak, g.count);
  return {
    i, count: j.total, total: j.total, streak: brk ? 0 : j.n, group: j.group, lastT: j.t, bestStreak, breaks: nBreaks,
    breakT: brk ? brk.t : null, breakCount: brk ? brk.count : 0,
    dcount: j.dtotal || 0, lastDoubleT: j.lastDoubleT ?? -10, lastPasses: j.passes || 1,
  };
}

// ---------- 脚下的效果（画布像素） ----------
export const ease3 = (x) => 1 - Math.pow(1 - clamp(x, 0, 1), 3);

// 腾空时地面亮一道光：两脚离地那段时间，脚下的地面线上一道两头渐隐的亮线
export function drawFloorGlow(ctx, tl, t, acc) {
  const i = lastJumpIndex(tl, t + 0.3);
  if (i < 0) return;
  for (let k = i; k >= Math.max(0, i - 1); k--) {
    const j = tl.jumps[k];
    const fadeIn = 0.06, fadeOut = 0.12;
    if (t < j.up_t - fadeIn || t > j.land_t + fadeOut) continue;
    const a = t < j.up_t ? (t - j.up_t + fadeIn) / fadeIn : t > j.land_t ? 1 - (t - j.land_t) / fadeOut : 1;
    // 地面高度：离地、落地两刻的脚底之间插值（手持拍摄时比逐帧估的地面线稳）
    const x = at(tl, tl.footX, t);
    const y = j.y != null && j.y_up != null ? j.y_up + (j.y - j.y_up) * clamp((t - j.up_t) / Math.max(1e-3, j.land_t - j.up_t), 0, 1) : at(tl, tl.groundY, t);
    if (Number.isNaN(x) || Number.isNaN(y)) continue;
    const half = 0.16 * tl.bodyPx;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const g = ctx.createLinearGradient(x - half, 0, x + half, 0);
    g.addColorStop(0, rgba(acc, 0));
    g.addColorStop(0.5, rgba(acc, 0.85 * a));
    g.addColorStop(1, rgba(acc, 0));
    ctx.fillStyle = g;
    const th = Math.max(3 * tl.sz.px, 0.006 * tl.bodyPx);
    ctx.fillRect(x - half, y - th / 2, 2 * half, th);
    // 光线下方一小片柔光，像地面被照亮
    ctx.globalAlpha = 0.35 * a;
    ctx.beginPath();
    ctx.ellipse(x, y + th, half * 0.7, half * 0.12, 0, 0, TAU);
    ctx.fillStyle = rgba(acc, 0.5);
    ctx.fill();
    ctx.restore();
  }
}

// 落地：两圈扁的波纹从脚下扩散，加十几条往两边溅起的火花
export function drawLanding(ctx, tl, t, acc) {
  const px = tl.sz.px, B = tl.bodyPx;
  const r0 = 0.08 * B, rMax = clamp(0.24 * B, 0.07 * tl.W, 0.22 * tl.W);
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (let i = lastJumpIndex(tl, t); i >= 0; i--) {
    const j = tl.jumps[i];
    if (t - j.t > 1.5) break;
    const tau = t - j.land_t;
    if (tau < 0 || tau > 0.6 || j.x == null) continue;
    // 双摇：三圈波纹、大一圈，火花多一些、飞得远一些
    const dbl = (j.passes || 1) >= 2;
    const rings = dbl ? [[0, 1], [0.08, 0.7], [0.16, 0.45]] : [[0, 1], [0.09, 0.55]];
    const rM = dbl ? rMax * 1.4 : rMax;
    for (const [delay, w] of rings) {
      const e = (tau - delay) / 0.45;
      if (e < 0 || e >= 1) continue;
      const rx = r0 + (rM - r0) * ease3(e);
      ctx.strokeStyle = rgba(acc, (1 - e) * w);
      ctx.lineWidth = Math.max(1.5 * px, (5 * (1 - e) + 1) * px * w * B / 800);
      ctx.beginPath();
      ctx.ellipse(j.x, j.y, rx, rx * 0.2, 0, 0, TAU);
      ctx.stroke();
    }
    const sb = B / 800;
    ctx.lineWidth = 2.2 * px * Math.max(0.6, sb);
    for (let s = 0; s < (dbl ? 22 : 14); s++) {
      const life = 0.32 + hash01(j.total, s, 3) * 0.18;
      if (tau >= life) continue;
      const side = s % 2 ? 1 : -1;
      const ang = (0.05 + hash01(j.total, s, 1) * 0.5) * Math.PI; // 0.05π–0.55π：贴地往外、略向上
      const sp = (260 + hash01(j.total, s, 2) * 320) * sb * (dbl ? 1.25 : 1);
      const vx0 = side * Math.cos(ang) * sp, vy0 = -Math.sin(ang) * sp;
      const g = 900 * sb;
      const x = j.x + side * r0 * 0.6 + vx0 * tau, y = j.y + vy0 * tau + 0.5 * g * tau * tau;
      const vx = vx0, vy = vy0 + g * tau;
      const k = 1 - tau / life;
      ctx.strokeStyle = rgba(s % 3 === 0 ? '#FFFFFF' : acc, k);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x - vx * 0.035, y - vy * 0.035);
      ctx.stroke();
    }
  }
  ctx.restore();
}

// 断绳：脚下一圈扁的红色波纹，0.6 秒
export function drawBreakFx(ctx, tl, t, color = DROP_COLOR) {
  for (const b of tl.breaks) {
    const a = t - b.t;
    if (a < 0 || a > 0.6 || b.x == null) continue;
    const e = ease3(a / 0.6);
    const rx = 0.12 * tl.bodyPx * (1 + 1.6 * e);
    ctx.save();
    ctx.globalAlpha = 0.9 * (1 - a / 0.6);
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(3 * tl.sz.px, 0.008 * tl.bodyPx);
    ctx.beginPath();
    ctx.ellipse(b.x, b.y, rx, rx * 0.22, 0, 0, TAU);
    ctx.stroke();
    ctx.restore();
  }
}

// 身旁标签"N · 跳"：贴在小腿高度、人的右侧（出画就放左侧）
function drawTag(ctx, tl, t, st, acc) {
  if (st.total < 1) return;
  const x = at(tl, tl.footX, t), y = at(tl, tl.groundY, t);
  if (Number.isNaN(x) || Number.isNaN(y)) return;
  const u = tl.u;
  const label = tl.doubleMode ? `${st.dcount} · 双摇` : `${st.total} · 跳`;
  ctx.save();
  ctx.font = `700 ${13 * u}px "Noto Sans SC", sans-serif`;
  const h = 26 * u, w = ctx.measureText(label).width + 24 * u;
  const off = 0.2 * tl.bodyPx;
  let bx = x + off;
  if (bx + w > tl.W - 12 * u) bx = x - off - w;
  const by = clamp(y - 0.14 * tl.bodyPx - h / 2, 0, tl.H - h);
  if (!tl.maskMode) {
    ctx.fillStyle = 'rgba(10,12,16,0.62)';
    rr(ctx, bx, by, w, h, h / 2);
    ctx.fill();
  }
  ctx.fillStyle = tl.opts.tint === 'all' ? acc : '#FFFFFF';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, bx + 12 * u, by + h / 2 + 1 * u);
  ctx.restore();
}

// ---------- HUD（基准坐标） ----------
function drawHud(ctx, tl, t, st, acc) {
  const u = tl.u, mask = tl.maskMode;
  const G = v3Geom(tl), hx = G.hudX;
  const ink = tl.opts.tint === 'all' ? acc : '#FFFFFF';
  ctx.save();
  ctx.transform(u, 0, 0, u, tl.ox, 0);
  // 双摇模式：大数字是双摇数，跟着双摇弹；否则是总数，每跳都弹
  const dm = tl.doubleMode;
  const tau = dm ? t - st.lastDoubleT : t - st.lastT;
  const shownN = dm ? st.dcount : st.total;
  const num = String(tl.opts.showCount ?? shownN);
  const numStyle = { text: num, font: numFont(tl, 55), size: 76, fill: ink, under: HALO, scale: u * 1.15 };
  const nw = textWidth(ctx, tl, numStyle);
  const label = { font: '700 12px "Noto Sans SC", sans-serif', size: 12, under: HALO, scale: u };
  let lx = hx + text(ctx, tl, { ...label, text: dm ? '双摇' : '跳绳', fill: ink, spacing: 2 }, hx, V3.labelY) + 10;
  // 标签后面：夹着双摇时写"双摇 N"，双摇模式里有单摇时写"单摇 N"，断过绳以后写最长的一组
  const chips = [];
  if (!dm && st.dcount > 0) chips.push([`双摇 ${st.dcount}`, acc]);
  if (dm && st.total - st.dcount > 0) chips.push([`单摇 ${st.total - st.dcount}`, 'rgba(255,255,255,0.75)']);
  if (st.breaks >= 1 && st.bestStreak > 0) chips.push([`最长连跳 ${st.bestStreak}`, acc]);
  for (const [c, fill] of chips) lx += text(ctx, tl, { ...label, text: c, fill }, lx, V3.labelY) + 10;
  const k = Math.exp(-9 * tau), s = 1 + 0.12 * k * Math.cos(19 * tau);
  ctx.save();
  ctx.translate(hx, V3.numBase);
  ctx.scale(s, s);
  ctx.translate(-hx, -V3.numBase);
  text(ctx, tl, numStyle, hx, V3.numBase);
  const flash = 0.6 * Math.exp(-10 * tau);
  if (flash > 0.01 && !mask && shownN > 0) {
    ctx.globalAlpha = flash;
    text(ctx, tl, { ...numStyle, fill: ink === acc ? '#FFFFFF' : acc, under: null }, hx, V3.numBase);
  }
  ctx.restore();
  text(ctx, tl, { text: '个', font: '700 15px "Noto Sans SC", sans-serif', size: 15, fill: ink, under: HALO, scale: u }, hx + nw + 6, V3.numBase);
  if (tau >= 0 && tau < 0.9 && shownN > 0 && !mask) {
    const op = tau < 0.08 ? tau / 0.08 : Math.max(0, 1 - Math.max(0, tau - 0.45) / 0.45);
    ctx.save();
    ctx.globalAlpha = op;
    const py = V3.numBase - 44 - 6 * (1 - Math.exp(-6 * tau));
    // 夹着的双摇：不是 +1，弹出"双摇！"
    if (!dm && st.lastPasses >= 2) text(ctx, tl, { text: st.lastPasses >= 3 ? '三摇！' : '双摇！', font: '900 15px "Noto Sans SC", sans-serif', size: 15, fill: acc, under: HALO, scale: u * 1.2 }, hx + nw + 8, py);
    else text(ctx, tl, { text: '+1', font: numFont(tl, 12), size: 16, fill: acc, under: HALO, scale: u }, hx + nw + 8, py);
    ctx.restore();
  }
  // 断绳：数字旁边弹出"断绳 · 这组 N 个"，1.4 秒后淡出（总数不清零）
  if (st.breakT != null && !mask) {
    const da = t - st.breakT;
    if (da < 1.4) {
      const op = da < 0.1 ? da / 0.1 : Math.min(1, (1.4 - da) / 0.4);
      const s2 = `断绳 · 这组 ${st.breakCount} 个`;
      const f = '700 11px "Noto Sans SC", sans-serif';
      const cw = textWidth(ctx, tl, { text: s2, font: f, size: 11, scale: u }) + 18, ch = 22;
      const cx0 = hx + nw + 32, cy0 = V3.numBase - 34 - 4 * (1 - Math.exp(-8 * da));
      ctx.save();
      ctx.globalAlpha = op;
      ctx.fillStyle = rgba(DROP_COLOR, 0.88);
      rr(ctx, cx0, cy0, cw, ch, ch / 2);
      ctx.fill();
      text(ctx, tl, { text: s2, font: f, size: 11, fill: '#FFFFFF', scale: u }, cx0 + 9, cy0 + 15);
      ctx.restore();
    }
  }
  const side = tl.opts.layout === 'jrside';
  // 右上：断过绳以后显示"第 N 组 · 连跳 M"（jrside 的右上是数据，不画）
  if (!side && st.breaks >= 1 && st.group >= 0) {
    const s3 = st.streak > 0 ? `第 ${st.group + 1} 组 · 连跳 ${st.streak}` : `第 ${st.group + 1} 组`;
    const f = '700 11px "Noto Sans SC", sans-serif';
    const pw = textWidth(ctx, tl, { text: s3, font: f, size: 11, scale: u }) + 30, px = G.right - pw, P = { y: G.pillY, h: V3.pill.h };
    if (!mask) {
      ctx.fillStyle = 'rgba(10,12,16,0.55)';
      rr(ctx, px, P.y, pw, P.h, P.h / 2);
      ctx.fill();
      ctx.fillStyle = acc;
      ctx.beginPath();
      ctx.arc(px + 12, P.y + P.h / 2, 3, 0, TAU);
      ctx.fill();
    }
    text(ctx, tl, { text: s3, font: f, size: 11, fill: ink, scale: u }, px + 21, P.y + 16);
  }
  if (side) {
    // 曲线在计数下面（左边一半宽），数据竖排在右上角
    const C = { x0: hx, x1: (0.4 * tl.W - tl.ox) / u, head: 150, top: 158, bot: 186, axis: 193, divider: null };
    drawChart(ctx, tl, t, st, acc, C, G.type);
    drawStatsColumn(ctx, tl, t, st, G.right, G.type, acc);
  } else {
    drawChart(ctx, tl, t, st, acc, G.chart, G.type);
    drawStats(ctx, tl, t, st, G.stats, G.type, acc);
  }
  ctx.restore();
}

// 髋部起伏曲线：最近 4.5 秒，纵轴按最近 6 秒的范围自动缩放；轴上标出每一跳（强调色）和断绳（红色）
function drawChart(ctx, tl, t, st, acc, C, T) {
  const mask = tl.maskMode;
  const { x0, x1 } = C;
  const hz = C.divider == null ? 11 : T.head;
  labV3(ctx, tl, T, '起跳节奏', x0, C.head, 'left', hz, panelInk(tl, acc));
  const air = at(tl, tl.air, t);
  if (st.total >= 1 && !Number.isNaN(air)) labV3(ctx, tl, T, air > 0.008 ? '腾空 ↑' : '落地', x1, C.head, 'right', hz, acc);
  if (mask) return;
  let lo = Infinity, hi = -Infinity;
  for (let f = Math.max(0, Math.ceil((t - 6) * tl.fps)); f <= Math.min(tl.n - 1, Math.floor(t * tl.fps)); f++) {
    const b = tl.bounce[f];
    if (!Number.isNaN(b)) { lo = Math.min(lo, b); hi = Math.max(hi, b); }
  }
  const range = Math.max(hi - lo, 0.02);
  const Yof = (b) => C.bot - clamp((b - lo) / range, 0, 1) * (C.bot - C.top);
  const t0 = Math.max(0, t - V3.window);
  const X = (ts) => x0 + ((ts - t0) / V3.window) * (x1 - x0);
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = acc;
  ctx.lineWidth = 1.8;
  ctx.shadowColor = acc;
  ctx.shadowBlur = 6 * tl.u;
  ctx.beginPath();
  let pen = false;
  for (let f = Math.max(0, Math.ceil(t0 * tl.fps)); f <= Math.floor(t * tl.fps) && f < tl.n; f++) {
    const b = tl.bounce[f];
    if (Number.isNaN(b)) { pen = false; continue; }
    if (!pen) { ctx.moveTo(X(f / tl.fps), Yof(b)); pen = true; } else ctx.lineTo(X(f / tl.fps), Yof(b));
  }
  const b0 = at(tl, tl.bounce, t);
  if (!Number.isNaN(b0) && pen) ctx.lineTo(X(t), Yof(b0));
  ctx.stroke();
  ctx.shadowBlur = 0;
  const cx = X(t);
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx, C.top - 4);
  ctx.lineTo(cx, C.axis + 2);
  ctx.stroke();
  if (!Number.isNaN(b0)) {
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(cx, Yof(b0), 3.2, 0, TAU);
    ctx.fill();
  }
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  for (let k = Math.ceil(t0 * 2); X(k / 2) <= x1 + 0.01; k++) {
    ctx.beginPath();
    ctx.arc(X(k / 2), C.axis, 1, 0, TAU);
    ctx.fill();
  }
  ctx.fillStyle = acc;
  for (let i = st.i; i >= 0 && tl.jumps[i].t >= t0; i--) ctx.fillRect(X(tl.jumps[i].t) - 0.75, C.axis - 4, 1.5, 8);
  ctx.fillStyle = DROP_COLOR;
  for (const b of tl.breaks) if (b.t >= t0 && b.t <= t) ctx.fillRect(X(b.t) - 1, C.axis - 6, 2, 12);
  if (C.divider != null) {
    ctx.fillStyle = 'rgba(255,255,255,0.18)';
    ctx.fillRect(x0, C.divider, x1 - x0, 0.8);
  }
  ctx.restore();
}

// 当前这组最近 6 跳的平均跳速（次/分）
export function recentBpm(tl, st) {
  if (st.i < 0 || st.streak < 2) return null;
  const ts = [];
  for (let i = st.i; i >= 0 && ts.length < 7; i--) {
    if (tl.jumps[i].group !== st.group) break;
    ts.push(tl.jumps[i].t);
  }
  return ts.length >= 2 ? Math.round((60 * (ts.length - 1)) / (ts[0] - ts[ts.length - 1])) : null;
}

// 数据：跳速 / 用时 / 断绳 → [名称, 数值, 单位]
function statCells(tl, t, st) {
  const b = recentBpm(tl, st);
  const bpm = b == null ? null : Math.round(b * tl.speed);
  const first = tl.jumps.length ? tl.jumps[0].t : 0, last = tl.jumps.length ? tl.jumps[tl.jumps.length - 1].t : 0;
  const secs = st.total >= 1 ? Math.floor((Math.min(t, last) - first) / tl.speed) : null;
  return [
    ['跳速', bpm != null ? String(bpm) : '—', bpm != null ? ' 次/分' : ''],
    ['用时', secs != null ? String(secs) : '—', secs != null ? ' 秒' : ''],
    ['断绳', st.total >= 1 ? String(st.breaks) : '—', st.total >= 1 ? ' 次' : ''],
  ];
}

// 一项数据：名称在上、数值 + 单位在下，按 align 对齐到 x
function statCell(ctx, tl, T, acc, [name, val, unit], x, align, yLabel, yValue) {
  const u = tl.u;
  subV3(ctx, tl, T, name, x, yLabel, align, T.label, panelInk(tl, acc, 0.85));
  const dash = `900 ${(1.375 * T.cap).toFixed(1)}px "Noto Sans SC", sans-serif`;
  const vs = { text: val, font: val === '—' ? dash : numFont(tl, T.cap, T.num), size: 1.375 * T.cap, fill: panelInk(tl, acc), under: HALO, scale: u };
  const us = { text: unit, font: cjkFont(T, T.font.regular, T.unit), size: T.unit, fill: panelInk(tl, acc, 0.85), under: HALO, scale: u };
  const vw = textWidth(ctx, tl, vs), uw = unit ? textWidth(ctx, tl, us) : 0;
  const left = align === 'left' ? x : align === 'center' ? x - (vw + uw) / 2 : x - vw - uw;
  text(ctx, tl, vs, left, yValue);
  if (unit) text(ctx, tl, us, left + vw, yValue);
}

// 底部数据行（v3 系列布局）
function drawStats(ctx, tl, t, st, S, T, acc) {
  const cells = statCells(tl, t, st);
  [[S.x0, 'left'], [(S.x0 + S.x1) / 2, 'center'], [S.x1, 'right']].forEach(([x, align], i) => statCell(ctx, tl, T, acc, cells[i], x, align, S.label, S.value));
}

// 右上角竖排（jrside）：每项一行名称、一行数值，右对齐
function drawStatsColumn(ctx, tl, t, st, right, T, acc) {
  const pitch = T.label + T.cap + 16;
  statCells(tl, t, st).forEach((c, i) => {
    const yl = V3.labelY + i * pitch;
    statCell(ctx, tl, T, acc, c, right, 'right', yl, yl + 6 + T.cap);
  });
}

/** 画一帧。参数同 fx.js 的 drawFrame（跳绳不抠图，video.patch 用不到）；dbg.only 只画某一部分（render/profile_parts.mjs 剖析用）。 */
export const PARTS = ['scrim', 'floor', 'landing', 'break', 'tag', 'hud', 'banner'];

export function drawFrame(ctx, tl, t, styleId, video, dbg) {
  const st = stateAt(tl, t);
  const acc = look(tl.styles, styleId, st.streak).head;
  const only = dbg && dbg.only;
  if (only) {
    ({ scrim: () => drawProtection(ctx, tl, t, st, video), floor: () => drawFloorGlow(ctx, tl, t, acc), landing: () => drawLanding(ctx, tl, t, acc),
      break: () => drawBreakFx(ctx, tl, t), tag: () => drawTag(ctx, tl, t, st, acc), hud: () => drawHud(ctx, tl, t, st, acc),
      banner: () => drawBanner(ctx, tl, t, styleId) })[only]();
    return;
  }
  const [sx, sy] = shakeAt(tl, t);
  const shaking = sx !== 0 || sy !== 0;
  if (video) {
    ctx.save();
    if (shaking) {
      ctx.translate(sx, sy);
      ctx.translate(tl.W / 2, tl.H / 2);
      ctx.scale(1.03, 1.03);
      ctx.translate(-tl.W / 2, -tl.H / 2);
    }
    video.full(ctx, shaking, true);
    ctx.restore();
  }
  drawProtection(ctx, tl, t, st, video);
  if (tl.feet) {
    ctx.save();
    ctx.translate(sx, sy);
    drawFloorGlow(ctx, tl, t, acc);
    drawLanding(ctx, tl, t, acc);
    drawBreakFx(ctx, tl, t);
    drawTag(ctx, tl, t, st, acc);
    ctx.restore();
  }
  drawHud(ctx, tl, t, st, acc);
  drawBanner(ctx, tl, t, styleId);
}

// ---------- 片尾成绩卡 ----------
export function endCardStats(tl) {
  const S = tl.summary;
  const js = tl.jumps;
  // 每一跳的瞬时跳速：和这组上一跳的间隔；每组第一跳没有
  const tempo = js.map((j, i) => (i > 0 && js[i - 1].group === j.group ? (60 * tl.speed) / Math.max(0.05, j.t - js[i - 1].t) : null));
  return {
    total: js.length, best: S.best ?? Math.max(0, ...tl.groups.map((g) => g.count)), groups: tl.groups.length, breaks: tl.breaks.length,
    secs: js.length > 1 ? Math.max(1, Math.round((js[js.length - 1].t - js[0].t) / tl.speed)) : 0,
    bpm: S.bpm == null ? null : Math.round(S.bpm),
    tempo, fastest: Math.max(0, ...tempo.filter((v) => v != null)),
    passes: js.map((j) => j.passes || 1), doubles: js.filter((j) => (j.passes || 1) >= 2).length, dm: !!tl.doubleMode,
  };
}

export function drawEndCard(ctx, tl, tc, opts) {
  const { W, u } = tl;
  const S = endCardStats(tl);
  const acc = look(tl.styles, opts.styleId, S.best).head;
  const T = panelType(tl);
  const fadeIn = ease3(tc / 0.35);
  const video = opts.lastFrame;
  if (video && fadeIn < 1) video.full(ctx, false);
  ctx.save();
  ctx.globalAlpha = fadeIn;
  drawEndBackground(ctx, tl, video, acc);
  ctx.transform(u, 0, 0, u, tl.ox, 0);
  const toD = (fx) => (fx * W - tl.ox) / u;
  const x0 = toD(EDGE.panel), x1 = toD(1 - EDGE.panel), cx = DESIGN_W / 2;
  const cjk = (w, z) => `${w} ${z}px ${T.font.cjk}, sans-serif`;
  const say = (s, x, y, align, font, size, fill, extra = {}) => text(ctx, tl, { text: s, font, size, fill, scale: u, ...extra }, x, y, align);

  if (opts.nickname) say(`@${opts.nickname}`, cx, 112, 'center', cjk(700, 15), 15, 'rgba(255,255,255,0.85)');
  // 双摇模式：大数字是双摇数；夹着双摇时下面写"其中双摇 N 个"
  const word = S.dm ? '双摇' : '跳绳';
  say(S.groups > 1 ? `${word}总数` : `连续${word}`, cx, 150, 'center', cjk(700, 14), 14, 'rgba(255,255,255,0.82)', { spacing: 5 });
  const final = S.dm ? S.doubles : S.total;
  const shown = Math.round(final * ease3((tc - 0.15) / 0.8));
  const big = { text: String(shown), font: numFont(tl, 150), size: 200, fill: acc, glow: rgba(acc, 0.45), blur: 36, scale: u };
  const bw = textWidth(ctx, tl, { ...big, text: String(final) });
  text(ctx, tl, big, cx, 330, 'center');
  say('个', cx + bw / 2 + 8, 330, 'left', cjk(700, 20), 20, '#FFFFFF');
  const sub = [];
  if (S.dm && S.total > S.doubles) sub.push(`共跳 ${S.total} 下 · 单摇 ${S.total - S.doubles} 个`);
  if (!S.dm && S.doubles) sub.push(`其中双摇 ${S.doubles} 个`);
  if (S.groups > 1) sub.push(`共 ${S.groups} 组 · 最长连跳 ${S.best} 个`);
  if (sub.length) say(sub.join(' · '), cx, 374, 'center', cjk(T.font.regular, 15), 15, 'rgba(255,255,255,0.8)');

  const cells = [
    ['平均跳速', S.bpm != null ? String(S.bpm) : '—', S.bpm != null ? ' 次/分' : '', x0, 'left'],
    ['用时', String(S.secs), ' 秒', cx, 'center'],
    ['断绳', String(S.breaks), ' 次', x1, 'right'],
  ];
  for (const [name, val, unit, x, align] of cells) {
    say(name, x, 432, align, cjk(T.font.regular, 13), 13, rgba(acc, 0.9));
    const vs = { text: val, font: val === '—' ? cjk(900, 34) : numFont(tl, 26, T.num), size: 36, fill: '#FFFFFF', scale: u };
    const us = { text: unit, font: cjk(T.font.regular, 13), size: 13, fill: 'rgba(255,255,255,0.75)', scale: u };
    const vw = textWidth(ctx, tl, vs), uw = unit ? textWidth(ctx, tl, us) : 0;
    const left = align === 'left' ? x : align === 'center' ? x - (vw + uw) / 2 : x - vw - uw;
    text(ctx, tl, vs, left, 468);
    if (unit) text(ctx, tl, us, left + vw, 468);
  }
  ctx.fillStyle = 'rgba(255,255,255,0.16)';
  ctx.fillRect(x0, 494, x1 - x0, 0.8);

  // 每一跳的节奏：柱高 = 和上一跳的间隔换算的跳速，最快那跳白色；断绳处一个红点；双摇的柱子顶上一个白点
  say(S.doubles ? '每一跳的节奏（白点是双摇）' : '每一跳的节奏', x0, 524, 'left', cjk(T.font.regular, 13), 13, rgba(acc, 0.9));
  if (S.fastest) say(`最快 ${Math.round(S.fastest)} 次/分`, x1, 524, 'right', cjk(T.font.regular, 12), 12, 'rgba(255,255,255,0.6)');
  const N = S.total;
  if (N) {
    const span = x1 - x0, base = 610;
    const gap = Math.min(3, span / N / 3.4);
    const bw2 = Math.min(10, (span - (N - 1) * gap) / N);
    const top = Math.max(1, ...S.tempo.filter((v) => v != null));
    const iMax = S.tempo.indexOf(top);
    const startX = x0 + (span - (N * bw2 + (N - 1) * gap)) / 2;
    S.tempo.forEach((v, i) => {
      const grow = ease3((tc - 0.3 - i * (0.6 / N)) / 0.25);
      if (grow <= 0) return;
      const bx = startX + i * (bw2 + gap);
      if (v == null && i > 0) { // 断绳后的第一跳
        ctx.fillStyle = DROP_COLOR;
        ctx.beginPath();
        ctx.arc(bx - gap / 2, base + 8, Math.max(1.5, Math.min(3, bw2 / 2)), 0, TAU);
        ctx.fill();
      }
      const hh = (8 + clamp((v ?? 0) / top, 0, 1) * 62) * grow;
      ctx.fillStyle = i === iMax ? '#FFFFFF' : rgba(acc, v == null ? 0.4 : 0.85);
      rr(ctx, bx, base - hh, bw2, hh, Math.min(3, bw2 / 2));
      ctx.fill();
      if (S.passes[i] >= 2) {
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(bx + bw2 / 2, base - hh - 4, Math.max(1.6, Math.min(3, bw2 / 2)), 0, TAU);
        ctx.fill();
      }
    });
  }

  const cta = ease3((tc - 0.9) / 0.3);
  ctx.globalAlpha = fadeIn * cta;
  const parts = S.dm ? [['你能双摇 ', '#FFFFFF'], [String(S.doubles), acc], [' 个吗？', '#FFFFFF']] : [['你能跳过 ', '#FFFFFF'], [String(S.total), acc], [' 个吗？', '#FFFFFF']];
  const ask = (y, align, x) => {
    const ws = parts.map(([s2]) => textWidth(ctx, tl, { text: s2, font: cjk(900, 21), size: 21, scale: u }));
    let px = align === 'center' ? x - ws.reduce((a2, b2) => a2 + b2, 0) / 2 : x;
    parts.forEach(([s2, fill], i) => { say(s2, px, y, 'left', cjk(900, 21), 21, fill); px += ws[i]; });
  };
  ask(668, 'center', cx);
  if (opts.text) drawTagline(ctx, tl, opts.text, cx, 688, acc, '#0A0C10', cjk(900, 17));
  ctx.restore();
}
