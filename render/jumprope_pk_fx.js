// 跳绳双人 PK 特效（轨迹 mode = "pk"，见 combocam/jumprope_track.build_track）：左红右蓝，外观和单人同一套。
// - 顶部：两边名字和计数（每跳弹一下、+1），中间 VS；红蓝 PK 条按双方计数分割，分界处一道亮光；
//   下面"红方领先 3 个"/"平手"，再下面双方跳速和用时；
// - 每个人头顶一块自己颜色的计数牌，领先的那个戴皇冠；脚下的腾空光、落地波纹、断绳波纹用各自的颜色；
// - 领先换人时横幅"蓝方反超！"加震屏；
// - 片尾：谁赢（或平局）、双方得分、平均跳速、最长连跳、断绳，"比赛过程"（双方累计得分随时间的两条线）。
// - 比法（summary.rule，见 combocam/jumprope_track.pk_points）：jumps 比总跳数，doubles 比双摇个数，passes 比绳子过脚次数（双摇算 2）。
//   比总跳数时谁跳了双摇就在名字旁写"双摇 N"、数字旁弹"双摇！"；比双摇时只有双摇加分，标题写"双摇 PK"。
// 全部放在顶部（jrpk 布局）：两个人并排时脚和身体可能在画面任何高度，底部面板总会挡住其中一个。
// 约定同 fx.js：所有效果都是时间 t 的纯函数，坐标是画布像素。

import { DESIGN_W, EDGE, HALO, TAU, clamp, cjkFont, drawEndBackground, drawProtection, drawTagline, numFont, panelType, rgba, rr, text, textWidth, v3Geom } from './fx.js';
import { at, buildTimeline as buildOne, drawBreakFx, drawFloorGlow, drawLanding, ease3, recentBpm, stateAt as stateOne } from './jumprope_fx.js';

export { legibility, lightScrimLUT } from './fx.js';

const DEFAULT_SIDES = { left: { name: '红方', color: '#FF4D6A' }, right: { name: '蓝方', color: '#3DA9FF' } };
const CROWN = '#FFD447';
// 领先要领先 LEAD_MARGIN 个才算：两人差不多同时跳时，谁先落地谁就"领先 1 个"，每跳都在换，不算反超。
// 皇冠、"反超"横幅都按这个；HUD 里"领先 N 个"照实时差值显示。和 combocam/jumprope_track.lead_changes 一致
const LEAD_MARGIN = 2;

export function buildTimeline(track, styles, makeCanvas, opts = {}) {
  const sides = (styles.jumprope && styles.jumprope.pk) || DEFAULT_SIDES;
  const base = { ...track, mode: 'single' };
  delete base.players;
  // 每个人按单人的时间线建（脚下效果、计数状态都复用单人的），布局不重要（不画单人的 HUD）
  const players = track.players.map((pl) => {
    const p = buildOne({ ...base, ...pl }, styles, makeCanvas, { ...opts, layout: 'jrside' });
    const sd = sides[pl.side] || DEFAULT_SIDES[pl.side];
    return Object.assign(p, { side: pl.side, name: sd.name, color: sd.color });
  });
  const A = players[0];
  const tl = {
    sport: 'jumprope', pk: true, players, summary: track.summary || {},
    fps: A.fps, W: A.W, H: A.H, n: A.n, u: A.u, ox: A.ox, rBall: A.rBall, sz: A.sz, scene: A.scene, duration: A.duration,
    styles: A.styles,
    opts: { ...A.opts, layout: 'jrpk' },
    makeCanvas: A.makeCanvas, cache: new Map(), measureCtx: A.measureCtx,
  };
  // 得分：每个人按比法把每一跳换成分数（scoreEv：得分的跳 {t, w 这一跳几分, c 到这一跳累计几分, passes}）
  tl.rule = tl.summary.rule || 'jumps';
  for (const p of players) {
    let c = 0;
    p.scoreEv = [];
    for (const j of p.jumps) {
      const w = points(j, tl.rule);
      if (w) { c += w; p.scoreEv.push({ t: j.t, w, c, passes: j.passes || 1 }); }
    }
  }
  // 领先的一方随时间怎么变（leads：每次换人的时刻）；从一方换到另一方的是"反超"
  const events = players.flatMap((p, k) => p.scoreEv.map((e) => ({ t: e.t, k, w: e.w }))).sort((a, b) => a.t - b.t);
  const cnt = [0, 0];
  let lead = null;
  tl.leads = [];
  tl.overtakes = [];
  for (const e of events) {
    cnt[e.k] += e.w;
    const d = cnt[0] - cnt[1];
    const now = d >= LEAD_MARGIN ? 0 : -d >= LEAD_MARGIN ? 1 : lead;
    if (now !== lead) {
      tl.leads.push({ t: e.t, k: now });
      if (lead != null) tl.overtakes.push({ t: e.t, k: now });
      lead = now;
    }
  }
  const firsts = players.filter((p) => p.jumps.length).map((p) => p.jumps[0].t);
  const lasts = players.filter((p) => p.jumps.length).map((p) => p.jumps[p.jumps.length - 1].t);
  tl.t0 = firsts.length ? Math.min(...firsts) : 0;
  tl.t1 = lasts.length ? Math.max(...lasts) : 0;
  tl.speed = Math.max(...players.map((p) => p.speed || 1));
  return tl;
}

/** 一跳在这种比法下得几分（同 combocam/jumprope_track.pk_points）。 */
function points(j, rule) {
  const n = j.passes || 1;
  return rule === 'passes' ? n : rule === 'doubles' ? (n >= 2 ? 1 : 0) : 1;
}

/** 双方的计数状态；count 是两人合计跳数（render.mjs 用它判断"开始跳了没有"）。
 * 每人的状态在单人的基础上加 score（得分）、scoreT / scoreW（最近一次得分的时刻、几分）、lastPasses（最近一跳过绳几次）。 */
export function stateAt(tl, t) {
  const sts = tl.players.map((p) => {
    const st = stateOne(p, t);
    let e = null;
    for (let i = p.scoreEv.length - 1; i >= 0; i--) if (p.scoreEv[i].t <= t) { e = p.scoreEv[i]; break; }
    return { ...st, score: e ? e.c : 0, scoreT: e ? e.t : -10, scoreW: e ? e.w : 0 };
  });
  const [a, b] = sts.map((s) => s.score);
  let lead = null; // 按 LEAD_MARGIN 定的领先方（戴皇冠的）
  for (const l of tl.leads) if (l.t <= t) lead = l.k;
  return { count: sts[0].total + sts[1].total, totals: [a, b], players: sts, lead };
}

// 得分平滑过渡（PK 条用）：每次得分之后 0.25 秒内从旧分滑到新分
function smoothCount(st, t) {
  return st.score ? st.score - st.scoreW + st.scoreW * ease3((t - st.scoreT) / 0.25) : 0;
}

function overtakeAt(tl, t) {
  for (let i = tl.overtakes.length - 1; i >= 0; i--) {
    const age = t - tl.overtakes[i].t;
    if (age >= 0 && age < 1.4) return { ...tl.overtakes[i], age };
    if (age >= 1.4) break;
  }
  return null;
}

function shakeAt(tl, t) {
  const b = overtakeAt(tl, t);
  if (!b || b.age >= 0.28) return [0, 0];
  const m = (0.28 - b.age) * 16 * tl.u;
  return [Math.sin(b.age * 95) * m, Math.cos(b.age * 70) * m * 0.6];
}

// ---------- 头顶计数牌（画布像素） ----------
function drawCrown(ctx, x, y, s) {
  // 三个尖的皇冠，底边中点在 (x, y)，宽 s
  ctx.beginPath();
  ctx.moveTo(x - s / 2, y);
  ctx.lineTo(x - s / 2, y - 0.45 * s);
  ctx.lineTo(x - s / 4, y - 0.2 * s);
  ctx.lineTo(x, y - 0.6 * s);
  ctx.lineTo(x + s / 4, y - 0.2 * s);
  ctx.lineTo(x + s / 2, y - 0.45 * s);
  ctx.lineTo(x + s / 2, y);
  ctx.closePath();
  ctx.fillStyle = CROWN;
  ctx.shadowColor = 'rgba(0,0,0,0.5)';
  ctx.shadowBlur = 0.15 * s;
  ctx.fill();
  ctx.shadowBlur = 0;
}

function drawPlayerTag(ctx, tl, p, t, st, leading) {
  const x1 = at(p, p.boxX1, t), x2 = at(p, p.boxX2, t);
  if (Number.isNaN(x1) || Number.isNaN(x2)) return;
  let hy = at(p, p.headY, t);
  if (Number.isNaN(hy)) hy = at(p, p.groundY, t) - p.bodyPx;
  if (Number.isNaN(hy)) return;
  const u = tl.u;
  const broke = st.breakT != null && t - st.breakT < 1.4; // 刚断绳：牌子上写出来（脚下是白色波纹，红色会被当成红方）
  // 刚跳了个双摇：牌子上写"双摇！"0.8 秒
  const dbl = st.lastPasses >= 2 && t - st.lastT < 0.8;
  const label = `${p.name} ${st.score}${broke ? ' · 断绳' : dbl ? (st.lastPasses >= 3 ? ' · 三摇！' : ' · 双摇！') : ''}`;
  ctx.save();
  ctx.font = `800 ${15 * u}px "Noto Sans SC", sans-serif`;
  const h = 30 * u, w = ctx.measureText(label).width + 26 * u;
  const cx = (x1 + x2) / 2;
  // 头顶上方，不压进顶部的计数区（约 24% 屏高以内）
  const by = clamp(hy - 0.06 * p.bodyPx - h, 0.245 * tl.H, tl.H - h);
  const tau = t - st.lastT;
  const s = 1 + (dbl ? 0.3 : 0.18) * Math.exp(-10 * Math.max(0, tau)) * (st.total > 0 ? 1 : 0);
  ctx.translate(cx, by + h / 2);
  ctx.scale(s, s);
  if (!tl.maskMode) {
    ctx.fillStyle = rgba(p.color, 0.92);
    rr(ctx, -w / 2, -h / 2, w, h, h / 2);
    ctx.fill();
    ctx.beginPath(); // 指向人的小三角
    ctx.moveTo(-6 * u, h / 2 - 1);
    ctx.lineTo(6 * u, h / 2 - 1);
    ctx.lineTo(0, h / 2 + 7 * u);
    ctx.closePath();
    ctx.fill();
    if (leading) drawCrown(ctx, 0, -h / 2 - 3 * u, 22 * u);
  }
  ctx.fillStyle = '#FFFFFF';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, 0, 1 * u);
  ctx.restore();
}

// ---------- 顶部 HUD（基准坐标） ----------
function drawPkHud(ctx, tl, t, S) {
  const u = tl.u, mask = tl.maskMode;
  const G = v3Geom(tl), L = G.hudX, R = G.right, cx = (L + R) / 2;
  const [A, B] = tl.players, [sa, sb] = S.players;
  ctx.save();
  ctx.transform(u, 0, 0, u, tl.ox, 0);
  // 名字
  const nameStyle = (p) => ({ text: p.name, font: '800 13px "Noto Sans SC", sans-serif', size: 13, fill: p.color, spacing: 2, under: HALO, scale: u });
  const nwA = text(ctx, tl, nameStyle(A), L, 52);
  const nwB = textWidth(ctx, tl, nameStyle(B));
  text(ctx, tl, nameStyle(B), R, 52, 'right');
  // 名字旁边：不是比双摇时，跳过双摇的写"双摇 N"
  if (tl.rule !== 'doubles') {
    [[A, sa, L + nwA + 8, 'left'], [B, sb, R - nwB - 8, 'right']].forEach(([p, st, x, align]) => {
      if (st.dcount > 0) text(ctx, tl, { text: `双摇 ${st.dcount}`, font: '700 11px "Noto Sans SC", sans-serif', size: 11, fill: rgba(p.color, 0.95), under: HALO, scale: u }, x, 52, align);
    });
  }
  // 中间的比法：比双摇写"双摇 PK"，比过绳次数写"比过绳次数"；比总跳数不写
  const ruleLabel = { doubles: '双摇 PK', passes: '比过绳次数' }[tl.rule];
  if (ruleLabel) text(ctx, tl, { text: ruleLabel, font: '800 12px "Noto Sans SC", sans-serif', size: 12, fill: '#FFFFFF', spacing: 2, under: HALO, scale: u }, cx, 52, 'center');
  // 得分：每次得分弹一下、闪一下自己的颜色、+1（比总跳数时双摇弹"双摇！"，比过绳次数时双摇弹"+2"）
  const numStyle = (st) => ({ text: String(st.score), font: numFont(tl, 50), size: 70, fill: '#FFFFFF', under: HALO, scale: u * 1.15 });
  [[A, sa, L, 'left'], [B, sb, R, 'right']].forEach(([p, st, x, align]) => {
    const ns = numStyle(st);
    const nw = textWidth(ctx, tl, ns);
    const tau = t - st.scoreT;
    const k = Math.exp(-9 * tau), s = 1 + 0.12 * k * Math.cos(19 * tau);
    const ax = align === 'left' ? x : x - nw; // 数字左边缘
    ctx.save();
    ctx.translate(ax, 118);
    ctx.scale(s, s);
    ctx.translate(-ax, -118);
    text(ctx, tl, ns, ax, 118);
    const flash = 0.6 * Math.exp(-10 * tau);
    if (flash > 0.01 && !mask && st.score > 0) {
      ctx.globalAlpha = flash;
      text(ctx, tl, { ...ns, fill: p.color, under: null }, ax, 118);
    }
    ctx.restore();
    if (tau >= 0 && tau < 0.9 && st.score > 0 && !mask) {
      const op = tau < 0.08 ? tau / 0.08 : Math.max(0, 1 - Math.max(0, tau - 0.45) / 0.45);
      const word = tl.rule === 'jumps' && st.lastPasses >= 2 && Math.abs(st.lastT - st.scoreT) < 1e-6;
      const ps = word ? { text: st.lastPasses >= 3 ? '三摇！' : '双摇！', font: '900 13px "Noto Sans SC", sans-serif', size: 13, fill: p.color, under: HALO, scale: u * 1.2 }
        : { text: `+${st.scoreW}`, font: numFont(tl, 12), size: 16, fill: p.color, under: HALO, scale: u };
      const pw = textWidth(ctx, tl, ps);
      ctx.save();
      ctx.globalAlpha = op;
      text(ctx, tl, ps, align === 'left' ? ax + nw + 6 : ax - 6 - pw, 118 - 40 - 6 * (1 - Math.exp(-6 * tau)));
      ctx.restore();
    }
  });
  // 中间 VS
  text(ctx, tl, { text: 'VS', font: numFont(tl, 20, 'barlowc'), size: 28, fill: '#FFFFFF', under: HALO, scale: u }, cx, 112, 'center');
  // PK 条：按平滑后的计数分割（各加 1，0 比 0 时各占一半）
  const ca = smoothCount(sa, t), cb = smoothCount(sb, t);
  const frac = (ca + 1) / (ca + cb + 2);
  const bx0 = L, bx1 = R, by = 134, bh = 10;
  const mid = bx0 + (bx1 - bx0) * frac;
  if (!mask) {
    ctx.save();
    rr(ctx, bx0, by, bx1 - bx0, bh, bh / 2);
    ctx.clip();
    ctx.fillStyle = A.color;
    ctx.fillRect(bx0, by, mid - bx0, bh);
    ctx.fillStyle = B.color;
    ctx.fillRect(mid, by, bx1 - mid, bh);
    // 条上一层浅色高光，像直播 PK 条的质感
    const g = ctx.createLinearGradient(0, by, 0, by + bh);
    g.addColorStop(0, 'rgba(255,255,255,0.35)');
    g.addColorStop(0.5, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(bx0, by, bx1 - bx0, bh);
    ctx.restore();
    // 分界处的亮光：有人刚跳完时更亮
    const hot = Math.max(Math.exp(-6 * (t - sa.scoreT)), Math.exp(-6 * (t - sb.scoreT)));
    ctx.save();
    ctx.fillStyle = '#FFFFFF';
    ctx.shadowColor = '#FFFFFF';
    ctx.shadowBlur = (6 + 10 * hot) * u;
    ctx.beginPath();
    ctx.ellipse(mid, by + bh / 2, 3 + 2 * hot, bh * 0.9, 0, 0, TAU);
    ctx.fill();
    ctx.restore();
  }
  // 领先多少
  if (sa.score + sb.score > 0) {
    const d = sa.score - sb.score;
    const lp = d > 0 ? A : B;
    const s2 = d === 0 ? '平手' : `${lp.name}领先 ${Math.abs(d)} ${tl.rule === 'passes' ? '次' : '个'}`;
    text(ctx, tl, { text: s2, font: '800 13px "Noto Sans SC", sans-serif', size: 13, fill: d === 0 ? '#FFFFFF' : lp.color, under: HALO, scale: u },
      cx, 164, 'center');
  }
  // 双方跳速、用时
  const T = panelType(tl);
  const small = (s3, x, align, fill) => text(ctx, tl, { text: s3, font: cjkFont(T, T.font.regular, 11), size: 11, fill, under: HALO, scale: u }, x, 190, align);
  const bpm = (p, st) => { const b = recentBpm(p, st); return b == null ? '—' : `${Math.round(b * (p.speed || 1))}`; };
  small(`跳速 ${bpm(A, sa)}`, L, 'left', rgba(A.color, 0.95));
  small(`跳速 ${bpm(B, sb)}`, R, 'right', rgba(B.color, 0.95));
  if (S.count > 0) small(`用时 ${Math.floor((Math.min(t, tl.t1) - tl.t0) / tl.speed)} 秒`, cx, 'center', 'rgba(255,255,255,0.85)');
  ctx.restore();
}

// "蓝方反超！"横幅：同颠球的里程碑横幅（斜着从左边飞入、停一秒、飞出），颜色换成反超那一方的
function drawOvertake(ctx, tl, t) {
  const b = overtakeAt(tl, t);
  if (!b) return;
  const p = tl.players[b.k];
  const u = tl.u;
  let off = 0;
  if (b.age < 0.2) off = -440 * Math.pow(1 - b.age / 0.2, 3);
  else if (b.age > 1.1) off = 440 * Math.pow((b.age - 1.1) / 0.3, 2);
  const label = `${p.name}反超！`;
  const paint = (g, k) => {
    g.shadowColor = p.color;
    g.shadowBlur = 30 * k;
    g.fillStyle = p.color;
    g.fillRect(-260, -36, 520, 72);
    g.shadowBlur = 0;
    g.fillStyle = 'rgba(255,255,255,0.9)';
    g.fillRect(-260, -44, 520, 3);
    g.fillRect(-260, 41, 520, 3);
    g.fillStyle = '#FFFFFF';
    g.font = '900 46px "Noto Sans SC", sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'alphabetic';
    g.fillText(label, 0, 16);
  };
  const pad = 50, key = `pkbanner|${label}`;
  let spr = tl.cache.get(key);
  if (!spr && tl.makeCanvas) {
    const c = tl.makeCanvas(Math.ceil((520 + 2 * pad) * u), Math.ceil((88 + 2 * pad) * u));
    const g = c.getContext('2d');
    g.scale(u, u);
    g.translate(260 + pad, 44 + pad);
    paint(g, u);
    spr = c;
    tl.cache.set(key, spr);
  }
  ctx.save();
  ctx.transform(u, 0, 0, u, tl.ox, 0);
  ctx.translate(195 + off, 300);
  ctx.rotate(-0.12);
  if (spr) ctx.drawImage(spr, -260 - pad, -44 - pad, 520 + 2 * pad, 88 + 2 * pad);
  else paint(ctx, u);
  ctx.restore();
}

export const PARTS = ['scrim', 'feet', 'tags', 'hud', 'banner'];

/** 画一帧。参数同 fx.js 的 drawFrame；dbg.only 只画某一部分（render/profile_parts.mjs 剖析用）。 */
export function drawFrame(ctx, tl, t, styleId, video, dbg) {
  const S = stateAt(tl, t);
  const feet = () => {
    for (const p of tl.players) {
      if (!p.feet) continue;
      drawFloorGlow(ctx, p, t, p.color);
      drawLanding(ctx, p, t, p.color);
      drawBreakFx(ctx, p, t, '#FFFFFF');
    }
  };
  const tags = () => tl.players.forEach((p, k) => drawPlayerTag(ctx, tl, p, t, S.players[k], S.lead === k));
  const only = dbg && dbg.only;
  if (only) {
    ({ scrim: () => drawProtection(ctx, tl, t, S, video), feet, tags, hud: () => drawPkHud(ctx, tl, t, S), banner: () => drawOvertake(ctx, tl, t) })[only]();
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
  drawProtection(ctx, tl, t, S, video);
  ctx.save();
  ctx.translate(sx, sy);
  feet();
  tags();
  ctx.restore();
  drawPkHud(ctx, tl, t, S);
  drawOvertake(ctx, tl, t);
}

// ---------- 片尾 ----------
export function endCardStats(tl) {
  return tl.players.map((p) => {
    const S = p.summary;
    return {
      total: p.jumps.length, best: S.best ?? Math.max(0, ...p.groups.map((g) => g.count)), breaks: p.breaks.length,
      bpm: S.bpm == null ? null : Math.round(S.bpm),
      score: p.scoreEv.length ? p.scoreEv[p.scoreEv.length - 1].c : 0,
      doubles: p.jumps.filter((j) => (j.passes || 1) >= 2).length,
    };
  });
}

export function drawEndCard(ctx, tl, tc, opts) {
  const { W, u } = tl;
  const [A] = tl.players;
  const st = endCardStats(tl);
  const win = st[0].score === st[1].score ? null : st[1].score > st[0].score ? 1 : 0;
  const acc = win == null ? '#FFFFFF' : tl.players[win].color;
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
  const xs = [cx - 92, cx + 92];

  if (opts.nickname) say(`@${opts.nickname}`, cx, 112, 'center', cjk(700, 15), 15, 'rgba(255,255,255,0.85)');
  say({ doubles: '双摇 PK', passes: '双人 PK · 比过绳次数' }[tl.rule] || '双人 PK', cx, 150, 'center', cjk(700, 14), 14, 'rgba(255,255,255,0.82)', { spacing: 5 });
  // 胜负牌：赢的一方数字上方"胜"，平局在中间
  const badge = ease3((tc - 0.9) / 0.25);
  if (badge > 0) {
    ctx.save();
    ctx.globalAlpha = fadeIn * badge;
    const label = win == null ? '平局' : '胜';
    const bw = textWidth(ctx, tl, { text: label, font: cjk(900, 16), size: 16, scale: u }) + 24, bh = 28;
    const bxc = win == null ? cx : xs[win];
    ctx.fillStyle = win == null ? 'rgba(255,255,255,0.9)' : tl.players[win].color;
    rr(ctx, bxc - bw / 2, 176, bw, bh, bh / 2);
    ctx.fill();
    say(label, bxc, 196, 'center', cjk(900, 16), 16, win == null ? '#0A0C10' : '#FFFFFF');
    if (win != null) drawCrown(ctx, bxc, 172, 26);
    ctx.restore();
  }
  // 双方得分（从 0 滚到最终数），输的一方暗一些
  tl.players.forEach((p, k) => {
    const shown = Math.round(st[k].score * ease3((tc - 0.15) / 0.8));
    ctx.save();
    if (win != null && win !== k) ctx.globalAlpha = fadeIn * (1 - 0.45 * ease3((tc - 0.9) / 0.3));
    text(ctx, tl, { text: String(shown), font: numFont(tl, 96), size: 130, fill: p.color, glow: rgba(p.color, 0.4), blur: 28, scale: u }, xs[k], 310, 'center');
    say(p.name, xs[k], 344, 'center', cjk(800, 15), 15, p.color);
    ctx.restore();
  });
  say('VS', cx, 282, 'center', numFont(tl, 20, 'barlowc'), 28, 'rgba(255,255,255,0.7)');

  // 数据对比：左值、名称、右值。有人跳了双摇时多一行：比双摇时写总跳数，否则写双摇个数
  const anyD = st.some((s) => s.doubles > 0);
  const rows = [
    ...(anyD ? [tl.rule === 'doubles' ? ['总跳数', (s) => String(s.total)] : ['双摇', (s) => String(s.doubles)]] : []),
    ['平均跳速', (s) => (s.bpm == null ? '—' : String(s.bpm))],
    ['最长连跳', (s) => String(s.best)],
    ['断绳', (s) => String(s.breaks)],
  ];
  const pitch = rows.length > 3 ? 24 : 30;
  rows.forEach(([name, val], i) => {
    const y = 392 + i * pitch;
    say(name, cx, y, 'center', cjk(T.font.regular, 12), 12, 'rgba(255,255,255,0.7)');
    st.forEach((s, k) => {
      const v = val(s);
      say(v, xs[k], y + 2, 'center', v === '—' ? cjk(900, 18) : numFont(tl, 16, T.num), 22, '#FFFFFF');
    });
  });
  ctx.fillStyle = 'rgba(255,255,255,0.16)';
  ctx.fillRect(x0, 470, x1 - x0, 0.8);

  // 比赛过程：双方累计得分随时间（阶梯线），从左往右展开；领先换人处一个白点
  say('比赛过程', x0, 500, 'left', cjk(T.font.regular, 13), 13, 'rgba(255,255,255,0.85)');
  if (tl.overtakes.length) say(`领先换了 ${tl.overtakes.length} 次`, x1, 500, 'right', cjk(T.font.regular, 12), 12, 'rgba(255,255,255,0.6)');
  const span = Math.max(0.5, tl.t1 - tl.t0), top = 516, base = 606;
  const maxN = Math.max(1, st[0].score, st[1].score);
  const X = (ts) => x0 + ((ts - tl.t0) / span) * (x1 - x0);
  const Y = (c) => base - (c / maxN) * (base - top);
  const reveal = tl.t0 + span * ease3((tc - 0.3) / 0.9);
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineWidth = 2;
  tl.players.forEach((p) => {
    ctx.strokeStyle = p.color;
    ctx.shadowColor = p.color;
    ctx.shadowBlur = 5 * u;
    ctx.beginPath();
    ctx.moveTo(X(tl.t0), Y(0));
    let c = 0;
    for (const e of p.scoreEv) {
      if (e.t > reveal) break;
      ctx.lineTo(X(e.t), Y(c));
      c = e.c;
      ctx.lineTo(X(e.t), Y(c));
    }
    ctx.lineTo(X(Math.min(reveal, tl.t1)), Y(c));
    ctx.stroke();
  });
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#FFFFFF';
  for (const o of tl.overtakes) {
    if (o.t > reveal) continue;
    const ev = tl.players[o.k].scoreEv.filter((e) => e.t <= o.t + 1e-6);
    const c = ev.length ? ev[ev.length - 1].c : 0;
    ctx.beginPath();
    ctx.arc(X(o.t), Y(c), 2.6, 0, TAU);
    ctx.fill();
  }
  ctx.restore();

  // 号召
  ctx.globalAlpha = fadeIn * ease3((tc - 1.0) / 0.3);
  say('叫上朋友来 PK', cx, 668, 'center', cjk(900, 21), 21, '#FFFFFF');
  if (opts.text) drawTagline(ctx, tl, opts.text, cx, 688, acc === '#FFFFFF' ? A.color : acc, '#FFFFFF', cjk(900, 17));
  ctx.restore();
}
