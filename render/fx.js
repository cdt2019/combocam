// 颠球特效绘制（运球也用这一套，文字按轨迹 JSON 的 labels 换）。
//
// 约定（浏览器里的实时预览和服务端成片可以共用这个文件）：
// - 只用标准 Canvas 2D API，不依赖 Node 或 DOM。
// - 所有效果都是时间 t（秒，原视频时间）的纯函数：同一个 t 永远画出同一帧，
//   预览可以随意拖动/慢放，服务端可以多线程乱序渲染。粒子的随机数由触球序号、发射时刻做种子。
// - 坐标：画布像素（与轨迹数据一致，默认 1080×1920）。
//   HUD（计数、横幅、节奏条）按 390×844 的基准坐标等比放大（以高度为准、水平居中）；
//   跟球走的效果（拖尾、光环、火花）按球的大小缩放，但以屏宽设上下限（见 buildTimeline 的 sz）。
// - 可读性：文字白色，底下垫顶部渐变和按画面亮度自适应的底板（track.scene），白字对比度目标 ≥ 4.5。

const TAU = Math.PI * 2;
const DESIGN_W = 390;
const DESIGN_H = 844;

// ---------- 小工具 ----------
function rgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
export function rgba(hex, a) {
  const c = rgb(hex);
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}
function mixPrefix(h1, h2, u) {
  const a = rgb(h1), b = rgb(h2);
  return `rgba(${Math.round(a[0] + (b[0] - a[0]) * u)},${Math.round(a[1] + (b[1] - a[1]) * u)},${Math.round(a[2] + (b[2] - a[2]) * u)},`;
}
function rr(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
// 确定性随机数：同一组整数输入永远得到同一个 [0,1) 值
function hash01(a, b, c) {
  let h = Math.imul(a | 0, 0x27d4eb2d) ^ Math.imul(b | 0, 0x165667b1) ^ Math.imul(c | 0, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ---------- 贴图缓存 ----------
// 大字号文字的发光（阴影模糊）很贵，同一段文字只渲染一次，之后每帧缩放贴图。
// 需要 tl.makeCanvas(w,h)（服务端传 createCanvas，浏览器传 OffscreenCanvas）；没有就直接画。
function textSprite(tl, key, o) {
  if (!tl.makeCanvas) return null;
  const hit = tl.cache.get(key);
  if (hit) return hit;
  const m = tl.measureCtx;
  m.font = o.font;
  const sp = o.spacing || 0;
  let width = 0;
  for (const ch of o.text) width += m.measureText(ch).width + sp;
  width -= sp;
  const mt = m.measureText(o.text);
  const asc = Math.ceil(mt.actualBoundingBoxAscent || o.size), desc = Math.ceil(mt.actualBoundingBoxDescent || o.size * 0.25);
  const pad = Math.ceil((o.blur || 0) * 1.3) + 4;
  const k = o.scale; // 贴图分辨率 = 基准单位 × k（画布像素）
  const c = tl.makeCanvas(Math.ceil((width + 2 * pad) * k), Math.ceil((asc + desc + 2 * pad) * k));
  const g = c.getContext('2d');
  g.scale(k, k);
  g.font = o.font;
  g.textAlign = 'left';
  g.textBaseline = 'alphabetic';
  g.fillStyle = o.fill;
  const paint = () => {
    let x = pad;
    if (sp) for (const ch of o.text) { g.fillText(ch, x, pad + asc); x += g.measureText(ch).width + sp; }
    else g.fillText(o.text, pad, pad + asc);
  };
  if (o.under) {
    // 暗色衬底：一层大范围柔光 + 一层紧贴笔画的，亮背景上也看得清，又不像一块底板
    g.shadowColor = o.under;
    g.shadowBlur = 12 * k;
    paint();
    g.shadowBlur = 4 * k;
    paint();
  }
  if (o.glow) { g.shadowColor = o.glow; g.shadowBlur = o.blur * k; }
  paint();
  const s = { canvas: c, w: width + 2 * pad, h: asc + desc + 2 * pad, ox: pad, oy: pad + asc, width };
  if (tl.cache.size > 400) tl.cache.delete(tl.cache.keys().next().value);
  tl.cache.set(key, s);
  return s;
}

const spriteKey = (o) => `${o.font}|${o.fill}|${o.glow}|${o.blur}|${o.under}|${o.spacing}|${o.scale}|${o.text}`;

function textWidth(ctx, tl, o) {
  const s = textSprite(tl, spriteKey(o), o);
  if (s) return s.width;
  ctx.save();
  ctx.font = o.font;
  const w = ctx.measureText(o.text).width;
  ctx.restore();
  return w;
}

// 在当前坐标系里画一段文字：(x, 基线 y)，align = left/center/right
// tl.maskMode 时只画纯白字形、不带发光和阴影（still.mjs 用来找笔画位置，量笔画周边的背景亮度）
function text(ctx, tl, o, x, y, align = 'left') {
  if (tl.maskMode) o = { ...o, fill: '#FFFFFF', glow: null, under: null, blur: 0 };
  if (align === 'right') return text(ctx, tl, o, x - textWidth(ctx, tl, o), y, 'left');
  const s = textSprite(tl, spriteKey(o), o);
  if (s) {
    const dx = align === 'center' ? x - s.width / 2 : x;
    ctx.drawImage(s.canvas, dx - s.ox, y - s.oy, s.w, s.h);
    return s.width;
  }
  ctx.save();
  ctx.font = o.font;
  ctx.fillStyle = o.fill;
  ctx.textBaseline = 'alphabetic';
  if (o.glow) { ctx.shadowColor = o.glow; ctx.shadowBlur = o.blur * o.scale; }
  const sp = o.spacing || 0;
  let width = 0;
  for (const ch of o.text) width += ctx.measureText(ch).width + sp;
  width -= sp;
  let cx = align === 'center' ? x - width / 2 : x;
  ctx.textAlign = 'left';
  if (sp) for (const ch of o.text) { ctx.fillText(ch, cx, y); cx += ctx.measureText(ch).width + sp; }
  else ctx.fillText(o.text, cx, y);
  ctx.restore();
  return width;
}

// ---------- 时间线 ----------
/**
 * @param makeCanvas 可选，(w,h) => 离屏画布；有它就启用贴图缓存和低分辨率泛光（服务端快 4 倍以上）
 */
// 成片默认外观：布局 v3（数据面板贴视频底部、左右铺满）、轻渐变保护层、Bebas Neue 数字、面板文字用曲线的颜色
//   tint：none 白字；panel 只有底部面板的字换成曲线颜色；all 计数和标签也换。panelFont 见 PANEL_FONTS
export const DEFAULT_LOOK = { layout: 'v3', protect: 'light', numFont: 'bebas', tint: 'panel', panelFont: 'barlowc' };

// 画面上的文字（v3 布局、片尾）。默认是颠球的；别的项目（篮球运球，见 combocam/dribble_track.py）在轨迹的 labels 里换掉一部分。
// {n}、{g} 是占位符：{n} 个数，{g} 组数
export const LABELS = {
  count: '连续颠球', tag: '{n} · 触球', chart: '球的高度', interval: '触球间隔', best: '最高一颠', unit: '次',
  banner: '{n} 连！', drop: '落地 · 重新计数', hand: '用手 · 重新计数',
  endTitle: '连续颠球', endTitleMulti: '最长连续颠球', endTotal: '共 {g} 组 · 总触球 {n} 次',
  endBars: '每一颠的高度', endBarsMulti: '最长那组每一颠的高度', marks: '{n} 连', askPre: '你能颠过 ', askPost: ' 个吗？',
};
const fmt = (s, v) => s.replace(/\{(\w)\}/g, (_, k) => (v[k] ?? ''));
const lab = (tl) => tl.labels || LABELS;

export function buildTimeline(track, styles, makeCanvas, opts = {}) {
  const n = track.frames;
  const bx = new Float32Array(n).fill(NaN);
  const by = new Float32Array(n).fill(NaN);
  const br = new Float32Array(n).fill(NaN);
  for (const [f, x, y, r] of track.ball) {
    if (f >= 0 && f < n) { bx[f] = x; by[f] = y; br[f] = r; }
  }
  const rs = track.ball.map((b) => b[3]).sort((a, b) => a - b);
  const rBall = track.calib && track.calib.ball_radius_px ? track.calib.ball_radius_px : rs.length ? rs[rs.length >> 1] : 40;
  const apexKey = new Map(track.apex.map((a) => [`${a.combo}:${a.touch}`, a]));
  const touches = track.touches
    .slice()
    .sort((a, b) => a.t - b.t)
    .map((tc) => {
      const ap = apexKey.get(`${tc.combo}:${tc.n}`);
      return { ...tc, h: ap ? ap.height_m : null, apexT: ap ? ap.t : null };
    });
  const u = track.height / DESIGN_H;
  const W = track.width;
  const px = W / 1080; // 以 1080 宽画布为基准的 1 像素
  const r = rBall;
  // 跟球走的效果：以屏幕为基准、按球大小微调，并设上下限。
  // 最初的效果按球只占屏宽 7% 画，直接按球放大会让近景时光环占满全屏、远景时又太小。
  const ringMax = clamp(2.5 * r, 0.06 * W, 0.14 * W);
  const sz = {
    px,
    hugR: 1.2 * r, // 贴着球的细圈
    hugW: clamp(0.05 * r, 2.5 * px, 5 * px),
    ringR0: 1.15 * r, // 触球冲击波：从球边缘扩散到 ringMax
    ringMax,
    fx: ringMax / 64, // 基准光环最大半径 64：火花、粒子按这个比例换算
    trailW: clamp(1.2 * r, 0.025 * W, 0.09 * W), // 拖尾最粗处
    trailLen: Math.min(12 * r, 0.45 * track.height), // 拖尾最长 6 个球径
    trailMinLen: 3 * r, // 移动不到 1.5 个球径时拖尾淡出，不画成一团
  };
  const tl = {
    fps: track.fps, W, H: track.height, n, bx, by, br,
    rBall, sz, u, ox: W / 2 - (DESIGN_W / 2) * u,
    touches, apex: track.apex, combos: track.combos, styles, duration: n / track.fps,
    scene: track.scene || null,
    opts: { ...DEFAULT_LOOK, ...Object.fromEntries(Object.entries(opts).filter(([, v]) => v != null && v !== '')), showCount: opts.showCount ?? null, panelScale: opts.panelScale || null },
    makeCanvas: makeCanvas || null, cache: new Map(), measureCtx: null, glowLayer: null, blurLayer: null,
    labels: { ...LABELS, ...(track.labels || {}) },
  };
  if (!LAYOUTS[tl.opts.layout] || !PROTECT[tl.opts.protect] || !['none', 'panel', 'all'].includes(tl.opts.tint)) {
    throw new Error(`未知的布局/保护层/文字颜色：${JSON.stringify(tl.opts)}`);
  }
  // 每组是怎么结束的：这组最后一触和下一组第一触之间有落地（bounce），就在落地那一刻清零、提示"落地"。
  // 球飞出画面、停太久这类断开没有落地点，等下一组开始时自然从 1 数起。
  const combos = (track.combos || []).slice().sort((a, b) => a.from - b.from);
  tl.breaks = [];
  combos.forEach((c, k) => {
    const next = combos[k + 1];
    const b = (track.bounces || []).find((bb) => bb.t > c.to && (!next || bb.t < next.from));
    if (b) tl.breaks.push({ t: b.t, x: b.x, y: b.y, count: c.count, kind: b.kind || 'ground' }); // kind：ground 落地 / hand 用手接
  });
  if (makeCanvas) {
    tl.measureCtx = makeCanvas(8, 8).getContext('2d');
    tl.glowLayer = makeCanvas(Math.ceil(tl.W / GLOW_DS), Math.ceil(tl.H / GLOW_DS));
  }
  return tl;
}
const GLOW_DS = 4; // 泛光层降采样倍数

// ---------- 可读性：画面亮度 → 压暗多少 ----------
const DEFAULT_LUM = 0.7; // 没有亮度数据时按"偏亮"处理
const TOP_SCRIM = { alpha: 0.25, until: 0.3 }; // 顶部渐变：顶边 25% 黑，到 30% 高度处为 0
const TARGET_CONTRAST = 4.5; // 白字对比度目标（小字标准；大字只需 3）

// 画布矩形区域 [x0,y0,x1,y1]（像素）在 t 时刻的亮度：取覆盖到的格子里最亮的（保守）
export function sceneLum(tl, rect, t) {
  const sc = tl.scene;
  if (!sc || !sc.lum.length) return DEFAULT_LUM;
  const c0 = clamp(Math.floor((rect[0] / tl.W) * sc.cols), 0, sc.cols - 1);
  const c1 = clamp(Math.floor(((rect[2] - 1) / tl.W) * sc.cols), 0, sc.cols - 1);
  const r0 = clamp(Math.floor((rect[1] / tl.H) * sc.rows), 0, sc.rows - 1);
  const r1 = clamp(Math.floor(((rect[3] - 1) / tl.H) * sc.rows), 0, sc.rows - 1);
  const at = (w) => {
    const g = sc.lum[clamp(w, 0, sc.lum.length - 1)];
    let m = 0;
    for (let rr_ = r0; rr_ <= r1; rr_++) for (let cc = c0; cc <= c1; cc++) m = Math.max(m, g[rr_ * sc.cols + cc]);
    return m;
  };
  const fw = t / sc.step - 0.5;
  const w0 = Math.floor(fw), a = clamp(fw - w0, 0, 1);
  return at(w0) * (1 - a) + at(w0 + 1) * a;
}

function topScrimAlpha(tl, y) {
  return TOP_SCRIM.alpha * clamp(1 - y / (TOP_SCRIM.until * tl.H), 0, 1);
}

// 压暗后的亮度：sRGB 空间里乘 (1-a)，换算到相对亮度约是 ×(1-a)^2.2
const darken = (L, a) => L * Math.pow(1 - a, 2.2);
const whiteContrast = (L) => 1.05 / (L + 0.05);
// 白字要达到对比度 c，背景需要的总压暗量
const needDarken = (L, c) => 1 - Math.pow(Math.min(1, (1.05 / c - 0.05) / Math.max(L, 1e-4)), 1 / 2.2);

// ---------- 布局（基准坐标） ----------
// v1：最早的布局。计数在左上，"最高一颠"在计数下面，节奏条在左侧中下。
// v2：计数上移；"最高一颠"和节奏条移到左下（脚下草地、短视频 App 文案区上方），避开画面中央的人和球。
// rects 是各行文字所在的区域，用来查背后亮度、核对对比度；plates 只给 plates 保护层用。
export const LAYOUTS = {
  v1: {
    labelY: 111, numTop: 118, best: { x: 20, y: 244, size: 13, bottom: false },
    rhythm: { labelX: 20, labelY: 580, barX: 20, base: 622, hMax: 34 },
    rects: { 标签: [20, 97, 120, 114], 数字: [20, 128, 120, 218], 最高一颠: [20, 230, 150, 246], 节奏: [20, 568, 170, 622] },
    plates: { hud: [4, 80, 232, 264], rhythm: [4, 552, 180, 638] },
  },
  v2: {
    labelY: 97, numTop: 104, best: { x: 20, y: 652, size: 14, bottom: true },
    rhythm: { labelX: 20, labelY: 692, barX: 54, base: 696, hMax: 28 },
    rects: { 标签: [20, 84, 120, 100], 数字: [20, 114, 120, 204], 最高一颠: [20, 637, 170, 654], 节奏: [20, 668, 200, 696] },
    plates: { hud: [4, 66, 232, 222], rhythm: [4, 620, 210, 708] },
  },
  // v3（默认）：左上小一号的计数 + 右上"第 N 组"，球旁"N · 触球"标签，
  // 底部数据面板：球的高度曲线 + 节奏 / 触球间隔 / 最高一颠。不画高度气泡（高度在面板里）。
  v3: {
    v3: true,
    // 左右边距按画面宽度比例算，可能超出基准坐标 0–390 的范围，这里的 x 放宽一些（只用来量对比度）
    rects: { 标签: [-24, 40, 110, 54], 数字: [-24, 66, 150, 126], 面板标题: [-24, 676, 414, 708], 面板数值: [-24, 776, 414, 826] },
    plates: { hud: [4, 80, 200, 190], rhythm: [4, 650, 380, 812] },
  },
  // v3 安全区版（SAFE_AREA）：面板压缩、抬到短视频 App 左下胶囊上方，右边收到按钮栏左边
  v3s: {
    v3: true,
    rects: { 标签: [-24, 40, 110, 54], 数字: [-24, 66, 150, 126], 面板标题: [-24, 552, 414, 584], 面板数值: [-24, 636, 414, 684] },
    plates: { hud: [4, 80, 200, 190], rhythm: [4, 550, 360, 690] },
  },
  // 同上，但只避开一个胶囊，面板更低、少挡腿
  v3l: {
    v3: true,
    rects: { 标签: [-24, 40, 110, 54], 数字: [-24, 66, 150, 126], 面板标题: [-24, 594, 414, 626], 面板数值: [-24, 678, 414, 726] },
    plates: { hud: [4, 80, 200, 190], rhythm: [4, 592, 360, 732] },
  },
  // v3 拆分版：高度曲线放到计数右边，底部只留一行数据（在锚点胶囊上方），尽量少挡脚下
  v3c: {
    v3: true,
    rects: { 标签: [-24, 40, 110, 54], 数字: [-24, 66, 150, 126], 面板标题: [140, 60, 414, 78], 面板数值: [-24, 642, 414, 684] },
    plates: { hud: [4, 80, 400, 190], rhythm: [4, 630, 360, 690] },
  },
  // 跳绳（render/jumprope_fx.js）脚在画面下部时：数据竖排在右上，曲线在计数下面，底部留给脚下的特效
  jrside: {
    v3: true,
    rects: { 标签: [-24, 40, 110, 54], 数字: [-24, 66, 150, 126], 面板标题: [-24, 140, 414, 190], 面板数值: [280, 40, 414, 180] },
    plates: { hud: [4, 40, 400, 200], rhythm: [4, 40, 400, 200] },
  },
  // 跳绳双人 PK（render/jumprope_pk_fx.js）：全在顶部——两边名字和计数、PK 条、领先多少、双方跳速
  jrpk: {
    v3: true,
    rects: { 标签: [-24, 40, 414, 54], 数字: [-24, 66, 414, 126], 面板标题: [-24, 150, 414, 166], 面板数值: [-24, 176, 414, 200] },
    plates: { hud: [4, 40, 400, 210], rhythm: [4, 40, 400, 210] },
  },
};

// ---------- 保护层：让 HUD 文字在任何背景上都看得清 ----------
//   plates  文字下面垫柔和暗影
//   edges   上下两段渐进模糊 + 压暗，中间的画面不动
//   veil    整页降饱和 + 轻压暗（像加了一层灰度透明层），文字所在的上下两段再补一点压暗
//   mix     轻 veil + edges
export const PROTECT = {
  light: { veil: 0, desat: 0, blur: false }, // 只有很淡的顶部渐变和底部面板渐变，画面保持清晰
  plates: { veil: 0, desat: 0, blur: false },
  edges: { veil: 0, desat: 0, blur: true },
  veil: { veil: 0.22, desat: 0.45, blur: false },
  mix: { veil: 0.1, desat: 0.25, blur: true },
};
// 渐变范围（占画面高度的比例）[满强度处, 降到 0 处]。模糊只在最边上，尽量碰不到人；压暗伸得远一点
const BANDS = {
  darkTop: [0.2, 0.36], blurTop: [0.12, 0.26],
  darkBot: [0.8, 0.62], blurBot: [0.9, 0.76],
};
const BLUR_PX = 16; // 1080 宽画布上的模糊半径
// light 保护层的两段渐变（高度比例）。top: [顶边不透明度, 降到 0 处]；bot: [起点, 满强度处, 不透明度]，满强度后保持到底边。
// 底部那段跟着面板走：面板抬高了，渐变也跟着抬
const LIGHT = {
  v3: { top: [0.42, 0.26], bot: [0.64, 1, 0.58] },
  v3s: { top: [0.42, 0.26], bot: [0.54, 0.8, 0.5] },
  v3l: { top: [0.42, 0.26], bot: [0.58, 0.85, 0.5] },
  v3c: { top: [0.45, 0.28], bot: [0.66, 0.8, 0.45] },
  jrside: { top: [0.45, 0.3], bot: [1, 1, 0] }, // 只压暗顶部
  jrpk: { top: [0.5, 0.32], bot: [1, 1, 0] },
};
function lightAlpha(tl, f) {
  const L = LIGHT[tl.opts.layout] || LIGHT.v3;
  const top = f >= L.top[1] ? 0 : L.top[0] * (1 - smooth(f / L.top[1]));
  const bot = f <= L.bot[0] ? 0 : L.bot[2] * smooth((f - L.bot[0]) / (L.bot[1] - L.bot[0]));
  return Math.max(top, bot);
}
const smooth = (x) => { const c = clamp(x, 0, 1); return c * c * (3 - 2 * c); };
function bandW(b, f) {
  const [full, zero] = b;
  return full < zero ? smooth((zero - f) / (zero - full)) : smooth((f - zero) / (full - zero));
}

function hudRectPx(tl, b) {
  return [tl.ox + b[0] * tl.u, b[1] * tl.u, tl.ox + b[2] * tl.u, b[3] * tl.u];
}

/**
 * 当前帧的保护层强度，以及每行 HUD 文字压暗后的白字对比度（数字是大字，目标 3；其余 4.5）。
 * 渲染器用它记录最差值，单帧预览用它标注。
 */
export function legibility(tl, t) {
  const lay = LAYOUTS[tl.opts.layout], mode = tl.opts.protect, P = PROTECT[mode];
  const rows = Object.entries(lay.rects).map(([name, r]) => {
    const rect = hudRectPx(tl, r);
    return { name, rect, lum: sceneLum(tl, rect, t), target: name === '数字' ? 3 : TARGET_CONTRAST, f: (rect[1] + rect[3]) / 2 / tl.H };
  });
  const out = { items: {}, lumMax: Math.max(...rows.map((r) => r.lum)), aTop: 0, aBot: 0, plates: {} };
  if (mode === 'light') {
    // 不按亮度加深，靠文字自带的暗色阴影；这里只按渐变估算整块背景（偏保守，实际以 still.mjs 的笔画周边实测为准）
    for (const r of rows) out.items[r.name] = whiteContrast(darken(r.lum, lightAlpha(tl, r.f)));
    const L = LIGHT[tl.opts.layout] || LIGHT.v3;
    out.aTop = L.top[0];
    out.aBot = L.bot[2];
    return out;
  }
  if (mode === 'plates') {
    for (const [k, b] of Object.entries(lay.plates)) {
      const rect = hudRectPx(tl, b);
      const L = sceneLum(tl, rect, t);
      const aT = topScrimAlpha(tl, (rect[1] + rect[3]) / 2);
      out.plates[k] = clamp(1 - (1 - needDarken(L, TARGET_CONTRAST)) / (1 - aT) + 0.08, 0.2, 0.68);
    }
    for (const r of rows) {
      const p = r.name === '节奏' || (lay.best.bottom && r.name === '最高一颠') ? out.plates.rhythm : out.plates.hud;
      out.items[r.name] = whiteContrast(darken(darken(r.lum, topScrimAlpha(tl, r.f * tl.H)), p));
    }
    return out;
  }
  // 每行需要的压暗量 → 扣掉整页那层 → 除以这一行在渐变里的权重 → 上下两段各取最大
  for (const r of rows) {
    const top = r.f < 0.5;
    const w = bandW(top ? BANDS.darkTop : BANDS.darkBot, r.f);
    const need = 1 - (1 - needDarken(r.lum, r.target)) / (1 - P.veil);
    const a = need / Math.max(w, 0.3);
    if (top) out.aTop = Math.max(out.aTop, a); else out.aBot = Math.max(out.aBot, a);
  }
  out.aTop = clamp(out.aTop + 0.06, 0.15, 0.7);
  out.aBot = clamp(out.aBot + 0.06, 0.15, 0.7);
  for (const r of rows) {
    const top = r.f < 0.5;
    const band = (top ? out.aTop : out.aBot) * bandW(top ? BANDS.darkTop : BANDS.darkBot, r.f);
    out.items[r.name] = whiteContrast(darken(r.lum, 1 - (1 - P.veil) * (1 - band)));
  }
  return out;
}

/**
 * light 保护层（上下两段渐变压暗）按行的查找表：在测试图上用当前画布实现画一遍渐变，读出每行"原值 → 压暗后"。
 * 服务端渲染拷视频帧时顺手查表压暗（render/pixels.mjs 的 lutCopy），结果和每帧画渐变逐字节相同，但省掉两块全宽渐变。
 * 返回 { lut: H×256 字节, active: 每行是否有压暗 }；不是 light 保护层或没有 makeCanvas 时返回 null。
 * 用法：视频帧提供者已经查表压暗时，把 video.scrimApplied 设为 true，drawFrame 就不再画渐变。
 */
export function lightScrimLUT(tl) {
  if (tl.opts.protect !== 'light' || !tl.makeCanvas || tl.W < 256) return null;
  const { W, H } = tl;
  const g = tl.makeCanvas(W, H).getContext('2d');
  const img = g.createImageData(W, H);
  for (let i = 0, p = 0; p < W * H; p++, i += 4) {
    const v = (p % W) & 255;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  drawProtection(g, tl, 0, { count: 0 }, null);
  const out = g.getImageData(0, 0, 256, H).data;
  const lut = new Uint8Array(H * 256), active = new Uint8Array(H);
  for (let y = 0; y < H; y++) {
    for (let v = 0; v < 256; v++) {
      const o = out[(y * 256 + v) * 4];
      lut[y * 256 + v] = o;
      if (o !== v) active[y] = 1;
    }
  }
  return { lut, active };
}

function drawProtection(ctx, tl, t, st, video) {
  const { W, H } = tl;
  const mode = tl.opts.protect, P = PROTECT[mode];
  if (mode === 'light') {
    if (video && video.scrimApplied) return; // 拷视频帧时已经查表压暗过（见 lightScrimLUT）
    const L = LIGHT[tl.opts.layout] || LIGHT.v3;
    for (const [y0, y1] of [[0, L.top[1]], [L.bot[0], 1]]) {
      const g = ctx.createLinearGradient(0, y0 * H, 0, y1 * H);
      for (let i = 0; i <= 16; i++) g.addColorStop(i / 16, `rgba(0,0,0,${lightAlpha(tl, y0 + ((y1 - y0) * i) / 16).toFixed(3)})`);
      ctx.fillStyle = g;
      ctx.fillRect(0, y0 * H, W, (y1 - y0) * H);
    }
    return;
  }
  const leg = legibility(tl, t);
  if (mode === 'plates') {
    const g = ctx.createLinearGradient(0, 0, 0, TOP_SCRIM.until * H);
    g.addColorStop(0, `rgba(0,0,0,${TOP_SCRIM.alpha})`);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, TOP_SCRIM.until * H);
    const lay = LAYOUTS[tl.opts.layout];
    plate(ctx, tl, lay.plates.hud, leg.plates.hud);
    if (st.count >= 1) plate(ctx, tl, lay.plates.rhythm, leg.plates.rhythm);
    return;
  }
  ctx.save();
  if (P.desat > 0) {
    ctx.globalCompositeOperation = 'saturation';
    ctx.globalAlpha = P.desat;
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, W, H);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
  }
  if (P.veil > 0) {
    ctx.fillStyle = `rgba(0,0,0,${P.veil})`;
    ctx.fillRect(0, 0, W, H);
  }
  if (P.blur && video && video.small && tl.makeCanvas) drawEdgeBlur(ctx, tl, video);
  // 上下两段渐进压暗（曲线按 BANDS 采样成渐变色标）
  for (const [b, a, y0, y1] of [[BANDS.darkTop, leg.aTop, 0, BANDS.darkTop[1]], [BANDS.darkBot, leg.aBot, BANDS.darkBot[1], 1]]) {
    const g = ctx.createLinearGradient(0, y0 * H, 0, y1 * H);
    for (let i = 0; i <= 12; i++) {
      const f = y0 + ((y1 - y0) * i) / 12;
      g.addColorStop(i / 12, `rgba(0,0,0,${(a * bandW(b, f)).toFixed(3)})`);
    }
    ctx.fillStyle = g;
    ctx.fillRect(0, y0 * H, W, (y1 - y0) * H);
  }
  ctx.restore();
}

// 上下两段渐进模糊：在 1/4 分辨率的整帧上模糊，用渐变遮罩只留上下两段，再放大叠回去
function drawEdgeBlur(ctx, tl, video) {
  const { canvas: small, ds } = video.small();
  const sw = small.width, sh = small.height;
  if (!tl.blurLayer) tl.blurLayer = tl.makeCanvas(sw, sh);
  const g = tl.blurLayer.getContext('2d');
  g.globalCompositeOperation = 'source-over';
  g.clearRect(0, 0, sw, sh);
  const br = (BLUR_PX * tl.sz.px) / ds;
  g.filter = `blur(${br.toFixed(2)}px)`;
  // 模糊会在画布边缘混入透明像素：四周各放大 2.5 倍半径，让透明的羽化边落到画布外面
  const m2 = Math.ceil(br * 2.5);
  g.drawImage(small, -m2, -m2, sw + 2 * m2, sh + 2 * m2);
  g.filter = 'none';
  g.globalCompositeOperation = 'destination-in';
  const m = g.createLinearGradient(0, 0, 0, sh);
  for (let i = 0; i <= 40; i++) {
    const f = i / 40;
    m.addColorStop(f, `rgba(0,0,0,${Math.max(bandW(BANDS.blurTop, f), bandW(BANDS.blurBot, f)).toFixed(3)})`);
  }
  g.fillStyle = m;
  g.fillRect(0, 0, sw, sh);
  g.globalCompositeOperation = 'source-over';
  const { W, H } = tl;
  const te = BANDS.blurTop[1], bs = BANDS.blurBot[1];
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high'; // 放大 4 倍，默认插值会出方块
  ctx.drawImage(tl.blurLayer, 0, 0, sw, sh * te, 0, 0, W, H * te);
  ctx.drawImage(tl.blurLayer, 0, sh * bs, sw, sh * (1 - bs), 0, H * bs, W, H * (1 - bs));
  ctx.restore();
}

// 底板：只画模糊后的"影子"，没有硬边，看起来是一团柔和的暗影而不是一张卡片
function plate(ctx, tl, b, alpha) {
  const u = tl.u;
  const feather = 18, inset = 0; // 基准单位
  const w = b[2] - b[0], h = b[3] - b[1];
  const key = `plate|${w}|${h}`;
  let spr = tl.cache.get(key);
  if (!spr && tl.makeCanvas) {
    const pad = feather * 1.4;
    const c = tl.makeCanvas(Math.ceil((w + 2 * pad) * u), Math.ceil((h + 2 * pad) * u));
    const g = c.getContext('2d');
    const far = 20000;
    g.fillStyle = '#000';
    g.shadowColor = '#000';
    g.shadowBlur = feather * u;
    g.shadowOffsetX = far;
    rr(g, (pad + inset) * u - far, (pad + inset) * u, (w - 2 * inset) * u, (h - 2 * inset) * u, 24 * u);
    g.fill();
    spr = { canvas: c, pad };
    tl.cache.set(key, spr);
  }
  ctx.save();
  ctx.globalAlpha = alpha;
  if (spr) {
    ctx.drawImage(spr.canvas, tl.ox + (b[0] - spr.pad) * u, (b[1] - spr.pad) * u, (w + 2 * spr.pad) * u, (h + 2 * spr.pad) * u);
  } else {
    ctx.fillStyle = '#000';
    ctx.shadowColor = '#000';
    ctx.shadowBlur = feather * u;
    rr(ctx, tl.ox + b[0] * u, b[1] * u, w * u, h * u, 18 * u);
    ctx.fill();
  }
  ctx.restore();
}

export function ballAt(tl, t) {
  const fr = t * tl.fps;
  const f0 = Math.floor(fr);
  const a = fr - f0;
  if (f0 < 0 || f0 >= tl.n || Number.isNaN(tl.bx[f0])) return null;
  const f1 = Math.min(f0 + 1, tl.n - 1);
  if (Number.isNaN(tl.bx[f1])) return a < 0.5 ? { x: tl.bx[f0], y: tl.by[f0], r: tl.br[f0] } : null;
  return {
    x: tl.bx[f0] + (tl.bx[f1] - tl.bx[f0]) * a,
    y: tl.by[f0] + (tl.by[f1] - tl.by[f0]) * a,
    r: tl.br[f0] + (tl.br[f1] - tl.br[f0]) * a,
  };
}

function lastTouchIndex(tl, t) {
  let lo = 0, hi = tl.touches.length - 1, ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (tl.touches[m].t <= t) { ans = m; lo = m + 1; } else hi = m - 1;
  }
  return ans;
}

export function isMilestone(styles, n) {
  const m = styles.milestones;
  return n > 0 && ((n <= m.until && n % m.every === 0) || n % m.thenEvery === 0);
}

export function stateAt(tl, t) {
  const i = lastTouchIndex(tl, t);
  let best = 0;
  for (const a of tl.apex) if (a.t <= t && a.height_m > best) best = a.height_m;
  if (i < 0) return { i, count: 0, combo: -1, lastT: -10, best, bestCombo: 0, dropT: null };
  const tc = tl.touches[i];
  let bestCombo = tc.n; // 到目前为止最长的一组（含当前这组）
  for (const c of tl.combos) if (c.to < tc.t) bestCombo = Math.max(bestCombo, c.count);
  // 这组最后一触之后球落地了：计数清零，等下一触从 1 数起
  let drop = null;
  for (const b of tl.breaks || []) if (b.t > tc.t && b.t <= t) drop = b;
  if (drop) return { i, count: 0, combo: tc.combo, lastT: tc.t, best, bestCombo, dropT: drop.t, dropKind: drop.kind, dropCount: tc.n };
  return { i, count: tc.n, combo: tc.combo, lastT: tc.t, best, bestCombo, dropT: null };
}

export function look(styles, styleId, count) {
  const st = styles.styles[styleId] || styles.styles.neon;
  let c = st.colors;
  if (!c) {
    c = styles.levels[0];
    for (const L of styles.levels) if (count >= L.from) c = L;
  }
  return { head: c.head, tail: c.tail, glow: c.glow, num: c.num, width: st.trail.width, dur: st.trail.dur, thin: st.trail.thin, particles: st.particles, tag: st.tag, name: st.name };
}

function bannerAt(tl, t) {
  for (let i = lastTouchIndex(tl, t); i >= 0; i--) {
    const tc = tl.touches[i];
    const age = t - tc.t;
    if (age > 1.4) break;
    if (isMilestone(tl.styles, tc.n)) return { text: tl.bannerText ? tl.bannerText(tc.n) : fmt(lab(tl).banner, { n: tc.n }), age, n: tc.n };
  }
  return null;
}

// 震屏偏移（画布像素）
export function shakeAt(tl, t) {
  const b = bannerAt(tl, t);
  if (!b || b.age >= 0.28) return [0, 0];
  const m = (0.28 - b.age) * 16 * tl.u;
  return [Math.sin(b.age * 95) * m, Math.cos(b.age * 70) * m * 0.6];
}

// ---------- 跟球走的效果 ----------
function drawTrail(ctx, tl, t, lk) {
  const S = tl.sz;
  // 长度按"走过的距离"截取（最长 6 个球径），而不是固定时长：
  // 最初的效果按球每 0.36 秒走 5–7 个球径设计，真实低颠只走 0.6–1 个，固定时长会画成一团。
  const lenK = clamp(lk.dur / 0.36, 1, 1.8); // 彗星、赛事风格的拖尾更长
  const maxLen = S.trailLen * lenK, win = 0.8 * lenK;
  const back = [];
  let dist = 0, prev = null;
  for (let k = 0; k <= win * 120; k++) {
    const ts = t - k / 120;
    if (ts < 0) break;
    const p = ballAt(tl, ts);
    if (!p) break; // 断了就不跨过去连
    if (prev) {
      dist += Math.hypot(p.x - prev.x, p.y - prev.y);
      if (dist > maxLen) break;
    }
    back.push({ x: p.x, y: p.y, d: dist });
    prev = p;
  }
  if (back.length < 2 || dist < 1) return;
  const total = back[back.length - 1].d;
  const fade = clamp(total / S.trailMinLen, 0, 1); // 几乎不动时淡出
  const pts = back.reverse().map((p) => ({ x: p.x, y: p.y, u: 1 - p.d / total }));
  const live = pts;
  const wMax = S.trailW * (lk.width / 1.05);
  const k = wMax / 29.4; // 基准拖尾最粗 28×1.05，按它换算发光和亮芯
  const strokeRange = (g, lo, hi) => {
    g.beginPath();
    let pen = false, any = false;
    for (const p of pts) {
      if (!p || p.u < lo || p.u > hi) { pen = false; continue; }
      if (!pen) { g.moveTo(p.x, p.y); pen = true; } else { g.lineTo(p.x, p.y); any = true; }
    }
    return any;
  };
  const bands = 9;
  const band = (b) => {
    const lo = b / bands, hi = (b + 1) / bands, um = (lo + hi) / 2;
    return {
      lo, hi, um,
      color: mixPrefix(lk.tail, lk.head, um) + (fade * (lk.thin ? 0.85 * um : Math.pow(um, 1.25))).toFixed(3) + ')',
      width: lk.thin ? Math.max(2.5 * S.px, 0.1 * S.trailW) : wMax * Math.pow(um, 0.85),
      blur: (lk.thin ? 4 : 10 * um) * k,
    };
  };
  const layer = tl.glowLayer;
  if (layer) {
    // 泛光：在 1/4 分辨率图层上画带阴影的拖尾，只处理拖尾外接框，再放大叠回来
    const margin = wMax + 12 * k;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of live) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
    x0 = Math.max(0, Math.floor((x0 - margin) / GLOW_DS)); y0 = Math.max(0, Math.floor((y0 - margin) / GLOW_DS));
    x1 = Math.min(layer.width, Math.ceil((x1 + margin) / GLOW_DS)); y1 = Math.min(layer.height, Math.ceil((y1 + margin) / GLOW_DS));
    if (x1 > x0 && y1 > y0) {
      const g = layer.getContext('2d');
      g.setTransform(1, 0, 0, 1, 0, 0);
      // 整层清空（只有 1/16 大小，很便宜）。只清外接框不够：放大贴回时双线性插值会读到框外一圈像素，
      // 那里可能留着这个线程之前画过的别的帧的泛光，结果和"之前画过哪些帧"有关（多线程渲染时同一帧画得不一样）
      g.clearRect(0, 0, layer.width, layer.height);
      g.setTransform(1 / GLOW_DS, 0, 0, 1 / GLOW_DS, 0, 0);
      g.lineCap = 'round';
      g.lineJoin = 'round';
      g.shadowColor = lk.glow;
      for (let b = 0; b < bands; b++) {
        const B = band(b);
        if (!strokeRange(g, B.lo - 0.03, B.hi + 0.03)) continue;
        g.strokeStyle = B.color;
        g.lineWidth = B.width;
        g.shadowBlur = B.blur / GLOW_DS;
        g.stroke();
      }
      ctx.drawImage(layer, x0, y0, x1 - x0, y1 - y0, x0 * GLOW_DS, y0 * GLOW_DS, (x1 - x0) * GLOW_DS, (y1 - y0) * GLOW_DS);
    }
  }
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (let b = 0; b < bands; b++) {
    const B = band(b);
    if (!strokeRange(ctx, B.lo - 0.03, B.hi + 0.03)) continue;
    ctx.strokeStyle = B.color;
    ctx.lineWidth = B.width;
    if (!layer) { ctx.shadowColor = lk.glow; ctx.shadowBlur = B.blur; }
    ctx.stroke();
  }
  if (!lk.thin && strokeRange(ctx, 0.45, 1.01)) {
    ctx.globalCompositeOperation = 'lighter';
    ctx.shadowBlur = 0;
    ctx.strokeStyle = `rgba(255,255,255,${(0.45 * fade).toFixed(3)})`;
    ctx.lineWidth = 5 * k;
    ctx.stroke();
  }
  ctx.restore();
}

// 贴着球的细圈（替代原来铺开的大光晕）：外面一圈淡暗描边，亮背景上也看得见
function drawHugRing(ctx, tl, p, lk) {
  const S = tl.sz;
  ctx.save();
  ctx.beginPath();
  ctx.arc(p.x, p.y, S.hugR, 0, TAU);
  ctx.strokeStyle = 'rgba(0,0,0,0.35)';
  ctx.lineWidth = S.hugW + 3 * S.px;
  ctx.stroke();
  ctx.strokeStyle = lk.head;
  ctx.lineWidth = S.hugW;
  ctx.shadowColor = lk.glow;
  ctx.shadowBlur = 10 * S.px;
  ctx.stroke();
  ctx.restore();
}

function drawTouchFx(ctx, tl, t, styleId) {
  const S = tl.sz;
  const sb = S.fx;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (let i = lastTouchIndex(tl, t); i >= 0; i--) {
    const tc = tl.touches[i];
    const tau = t - tc.t;
    if (tau > 0.6) break;
    const lk = look(tl.styles, styleId, Math.max(0, tc.n - 1));
    // 冲击波：从球边缘扩散到 ringMax（不超过屏宽 14%）
    const e = tau / 0.45;
    if (e < 1) {
      const k = 1 - e;
      ctx.strokeStyle = rgba(lk.glow, k);
      ctx.lineWidth = (3.5 * k + 0.5) * sb;
      ctx.beginPath();
      ctx.arc(tc.x, tc.y, S.ringR0 + (S.ringMax - S.ringR0) * (1 - Math.pow(1 - e, 3)), 0, TAU);
      ctx.stroke();
    }
    // 火花：12 条，按触球序号定种子
    ctx.lineWidth = 2 * sb;
    for (let s = 0; s < 12; s++) {
      const life = 0.38 + hash01(i, s, 3) * 0.2;
      if (tau >= life) continue;
      const ang = (TAU * s) / 12 + hash01(i, s, 1) * 0.3;
      const sp = 120 + hash01(i, s, 2) * 140;
      const vx0 = Math.cos(ang) * sp, vy0 = Math.sin(ang) * sp - 60;
      const dmp = Math.pow(0.98, 60 * tau);
      const x = tc.x + ((vx0 * (1 - dmp)) / 1.2) * sb;
      const y = tc.y + (vy0 * tau + 260 * tau * tau) * sb;
      const vx = vx0 * dmp, vy = vy0 + 520 * tau;
      const k = 1 - tau / life;
      ctx.strokeStyle = rgba(s % 3 === 0 ? '#FFFFFF' : lk.glow, k);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x - vx * 0.04 * sb, y - vy * 0.04 * sb);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawParticles(ctx, tl, t, lk) {
  if (!lk.particles) return;
  const sb = tl.sz.fx;
  const flame = lk.particles === 'flame';
  const maxLife = flame ? 0.75 : 0.8;
  const per = flame ? 2 : 1;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (let k = Math.ceil((t - maxLife) * 60); k <= Math.floor(t * 60); k++) {
    if (k < 0) continue;
    const te = k / 60;
    const p = ballAt(tl, te);
    if (!p) continue;
    const age = t - te;
    for (let j = 0; j < per; j++) {
      const r = (q) => hash01(k, j, q);
      let x, y, life, size, color, alpha;
      if (flame) {
        life = 0.45 + r(5) * 0.3;
        if (age >= life) continue;
        x = p.x + ((r(1) - 0.5) * 12 + (r(3) - 0.5) * 30 * age) * sb;
        y = p.y + ((r(2) - 0.5) * 12 + (-30 - r(4) * 50) * age) * sb;
        size = (2 + r(6) * 3) * sb;
        color = r(7) < 0.5 ? '#FFB547' : '#FF4A1F';
        alpha = 1 - age / life;
      } else {
        life = 0.8;
        if (age >= life) continue;
        x = p.x + ((r(1) - 0.5) * 18 + (r(3) - 0.5) * 14 * age) * sb;
        y = p.y + ((r(2) - 0.5) * 18 + (r(4) - 0.5) * 14 * age) * sb;
        size = (1.4 + r(6) * 2.2) * sb;
        color = r(7) < 0.6 ? '#FFE7A3' : '#FFFFFF';
        const kk = 1 - age / life;
        alpha = kk * (0.55 + 0.45 * Math.sin(age * 40));
      }
      const kk = 1 - age / life;
      ctx.fillStyle = rgba(color, alpha);
      ctx.beginPath();
      ctx.arc(x, y, size * (0.4 + 0.6 * kk), 0, TAU);
      ctx.fill();
    }
  }
  ctx.restore();
}

function drawApexLabel(ctx, tl, t) {
  if (LAYOUTS[tl.opts.layout].v3) return; // v3 的高度放在底部面板里
  const u = tl.u;
  for (const tc of tl.touches) {
    if (tc.apexT == null || tc.h == null) continue;
    const D = Math.max(0.2, 2 * (tc.apexT - tc.t));
    const d = Math.abs(t - tc.apexT) / D;
    if (d > 0.22) continue;
    const ap = tl.apex.find((a) => a.combo === tc.combo && a.touch === tc.n);
    const p = ballAt(tl, t) || { x: ap.x, y: ap.y };
    const ay = ap.y;
    const w = 78 * u, h = 30 * u;
    const off = tl.sz.hugR + 6 * tl.sz.px;
    let x1 = p.x + off, lx = x1 + 14 * u, dir = 1;
    if (lx + w > tl.W - 16 * u) { x1 = p.x - off; lx = x1 - 14 * u - w; dir = -1; }
    ctx.save();
    ctx.globalAlpha = 1 - d / 0.22;
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 1 * u;
    ctx.setLineDash([3 * u, 3 * u]);
    ctx.beginPath();
    ctx.moveTo(x1, ay);
    ctx.lineTo(dir > 0 ? lx - 4 * u : lx + w + 4 * u, ay);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    rr(ctx, lx, ay - h / 2, w, h, h / 2);
    ctx.fill();
    ctx.fillStyle = '#FFFFFF';
    ctx.font = `700 ${15 * u}px "Noto Sans SC", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`约 ${tc.h.toFixed(1)} m`, lx + w / 2, ay + 1 * u);
    ctx.restore();
  }
}

// 球旁标签（贴球细圈由 drawHugRing 统一画）。v3："17 · TOUCH"式胶囊；其余布局：赛事风格的"触球 N"
// move：这一下的动作（篮球运球的 cross / legs，轨迹 labels.moves 里有名字时写成"N · 交叉"）
function drawTag(ctx, tl, p, count, acc, move) {
  const u = tl.u;
  const v3 = LAYOUTS[tl.opts.layout].v3;
  const mv = move && lab(tl).moves ? lab(tl).moves[move] : null;
  const label = v3 ? (mv ? fmt(lab(tl).tagMove, { n: count, m: mv }) : fmt(lab(tl).tag, { n: count })) : `触球 ${count}`;
  ctx.save();
  ctx.font = `700 ${13 * u}px "Noto Sans SC", sans-serif`;
  const h = 26 * u, w = ctx.measureText(label).width + (v3 ? 24 : 19) * u;
  let bx = p.x + tl.sz.hugR + 8 * tl.sz.px;
  if (bx + w > tl.W - 12 * u) bx = p.x - tl.sz.hugR - 8 * tl.sz.px - w;
  const by = p.y - h / 2;
  if (!tl.maskMode) {
    ctx.fillStyle = v3 ? 'rgba(10,12,16,0.62)' : 'rgba(0,0,0,0.62)';
    rr(ctx, bx, by, w, h, v3 ? h / 2 : 5 * u);
    ctx.fill();
    if (!v3) { ctx.fillStyle = '#22E3FF'; ctx.fillRect(bx, by, 3 * u, h); }
  }
  ctx.fillStyle = v3 && tl.opts.tint === 'all' ? acc : '#FFFFFF';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, bx + (v3 ? 12 : 10) * u, by + h / 2 + 1 * u);
  ctx.restore();
}

// 篮球运球的花式动作（交叉、胯下）：触地点旁边弹出动作名，0.7 秒。只有轨迹 labels.moves 里有名字的动作才画；
// 只画最近一下的（快速运球时一秒三四下，叠在一起看不清）。放在触地点背着球走的那一边，不压住跟球走的标签
function drawMovePop(ctx, tl, t, lk) {
  const names = lab(tl).moves;
  const i = lastTouchIndex(tl, t);
  if (!names || i < 0) return;
  const tc = tl.touches[i];
  const name = tc.move && names[tc.move];
  const a = t - tc.t;
  if (!name || a > 0.7) return;
  const K = tl.u * 1.25; // 基准坐标 1 像素 → 画布像素
  const o = { text: fmt(lab(tl).movePop, { m: name }), font: '900 30px "Noto Sans SC", sans-serif', size: 30, fill: '#FFFFFF',
    glow: rgba(lk.glow, 0.85), blur: 16, under: 'rgba(0,0,0,0.6)', scale: K };
  const w = textWidth(ctx, tl, o) * K;
  const p = ballAt(tl, tc.t + 0.1); // 按触地后 0.1 秒球的位置定下放哪边，整个弹字期间不动
  const dir = p && p.x < tc.x - 0.3 * tl.rBall ? 1 : -1; // 球往左走就放右边，否则放左边
  let x = tc.x + dir * (tl.rBall + 10 * tl.u + w / 2);
  if (dir > 0 && p) {
    // 跟球走的标签在球右边（见 drawTag）：球刚往左走一点时标签还压在触地点右边，要放到标签外面
    ctx.save();
    ctx.font = `700 ${13 * tl.u}px "Noto Sans SC", sans-serif`;
    const tagW = ctx.measureText(fmt(lab(tl).tagMove, { n: tc.n, m: name })).width + 24 * tl.u;
    ctx.restore();
    const tagRight = p.x + tl.sz.hugR + 8 * tl.sz.px + tagW;
    if (Math.abs(p.y - tc.y) < tl.rBall + 30 * tl.u) x = Math.max(x, tagRight + 8 * tl.u + w / 2);
  }
  if (x - w / 2 < 12 * tl.u || x + w / 2 > tl.W - 12 * tl.u) x = tc.x - dir * (tl.rBall + 10 * tl.u + w / 2);
  const y = tc.y - 0.2 * tl.rBall;
  const e = 1 - Math.pow(1 - clamp(a / 0.18, 0, 1), 3);
  ctx.save();
  ctx.globalAlpha = a < 0.5 ? 1 : 1 - (a - 0.5) / 0.2;
  ctx.translate(x, y);
  ctx.scale(K * (0.7 + 0.3 * e), K * (0.7 + 0.3 * e));
  text(ctx, tl, o, 0, 10, 'center');
  ctx.restore();
}

// 落地（这一组断了）：落点处一圈扁的红色波纹，0.6 秒
const DROP_COLOR = '#FF5A5A';
function drawDropFx(ctx, tl, t) {
  for (const b of tl.breaks || []) {
    if (b.kind === 'hand') continue; // 用手接住：没有落点
    const a = t - b.t;
    if (a < 0 || a > 0.6) continue;
    const e = 1 - Math.pow(1 - a / 0.6, 3);
    const rx = tl.rBall * (1 + 1.6 * e), ry = rx * 0.32;
    ctx.save();
    ctx.globalAlpha = 0.9 * (1 - a / 0.6);
    ctx.strokeStyle = DROP_COLOR;
    ctx.lineWidth = Math.max(3 * tl.sz.px, tl.rBall * 0.08);
    ctx.beginPath();
    ctx.ellipse(b.x, b.y + tl.rBall * 0.9, rx, ry, 0, 0, TAU);
    ctx.stroke();
    ctx.restore();
  }
}

// 球经过 HUD 时让 HUD 变淡：球（含半径）离文字区域越近越淡，压在上面时只剩 25%
function hudFade(tl, t, box) {
  if (tl.maskMode) return 1;
  const p = ballAt(tl, t);
  if (!p) return 1;
  const [x0, y0, x1, y1] = [tl.ox + box[0] * tl.u, box[1] * tl.u, tl.ox + box[2] * tl.u, box[3] * tl.u];
  const dx = Math.max(x0 - p.x, 0, p.x - x1), dy = Math.max(y0 - p.y, 0, p.y - y1);
  const d = Math.hypot(dx, dy), r = tl.rBall;
  return 0.25 + 0.75 * smooth((d - r) / (1.5 * r));
}

// ---------- HUD（基准坐标） ----------
function drawHud(ctx, tl, t, st, lk) {
  const u = tl.u;
  const lay = LAYOUTS[tl.opts.layout];
  if (lay.v3) { drawHudV3(ctx, tl, t, st, lk); return; }
  ctx.save();
  ctx.transform(u, 0, 0, u, tl.ox, 0);
  // 文字一律白色（下面垫了保护层），连击等级的颜色放在发光、+1、节奏条和每次触球的闪色上
  const shade = { glow: 'rgba(0,0,0,0.55)', blur: 6 };
  // 标签
  text(ctx, tl, { text: '连续颠球', font: '700 14px "Noto Sans SC", sans-serif', size: 14, fill: '#FFFFFF', spacing: 3, scale: u, ...shade }, 20, lay.labelY);
  // 大号计数：每次触球弹一下、闪一下等级色
  const tau = t - st.lastT;
  const num = String(st.count);
  const numFont = '112px Anton, "Noto Sans SC", sans-serif';
  const top = lay.numTop, base = top + 100;
  const k = Math.exp(-9 * tau), w = Math.cos(19 * tau);
  const s = 1 + 0.26 * k * w, sq = 0.07 * Math.exp(-14 * tau), ty = -12 * k * Math.max(0, w);
  // 贴图按最大放大倍数（1.26）渲染，弹跳放大时也清晰
  const numStyle = { text: num, font: numFont, size: 112, fill: '#FFFFFF', glow: rgba(lk.glow, 0.75), blur: 26, under: 'rgba(0,0,0,0.5)', scale: u * 1.26 };
  const nw = textWidth(ctx, tl, numStyle);
  const ox = 20 + 0.35 * nw, oy = top + 0.9 * 112;
  ctx.save();
  ctx.translate(ox, oy + ty);
  ctx.scale(s - sq, s + sq);
  ctx.translate(-ox, -oy);
  text(ctx, tl, numStyle, 20, base);
  const flash = 0.7 * Math.exp(-10 * tau);
  if (flash > 0.01) {
    ctx.globalAlpha = flash;
    text(ctx, tl, { text: num, font: numFont, size: 112, fill: lk.num, scale: u * 1.26 }, 20, base);
  }
  ctx.restore();
  // +1
  if (tau >= 0 && tau < 0.75 && st.count > 0) {
    const op = tau < 0.06 ? tau / 0.06 : Math.max(0, 1 - Math.max(0, tau - 0.3) / 0.45);
    const sc = tau < 0.12 ? 0.4 + 0.95 * (tau / 0.12) : 1.35 - 0.35 * Math.min(1, (tau - 0.12) / 0.18);
    const py = -30 * (1 - Math.pow(1 - Math.min(1, tau / 0.75), 3));
    const px0 = 20 + nw + 6, py0 = top + 112 - 12;
    ctx.save();
    ctx.globalAlpha = op;
    ctx.translate(px0 + tau * 8, py0 + py);
    ctx.scale(sc, sc);
    ctx.rotate((-8 * Math.PI) / 180);
    text(ctx, tl, { text: '+1', font: '32px Anton, sans-serif', size: 32, fill: lk.num, glow: rgba(lk.glow, 0.65), blur: 20, under: 'rgba(0,0,0,0.55)', scale: u * 1.35 }, 0, 0);
    ctx.restore();
  }
  // 最高一颠（v2 放在左下，触球后才出现）
  const B = lay.best;
  if (!B.bottom || st.count >= 1) {
    text(ctx, tl, { text: st.best > 0 ? `最高一颠 约 ${st.best.toFixed(1)} m` : '最高一颠 —', font: `700 ${B.size}px "Noto Sans SC", sans-serif`, size: B.size, fill: '#FFFFFF', scale: u, ...shade }, B.x, B.y);
  }
  // 节奏条：当前这串最近 16 次
  if (st.count >= 1) {
    const R = lay.rhythm;
    const show = Math.min(st.count, 16);
    text(ctx, tl, { text: '节奏', font: '700 12px "Noto Sans SC", sans-serif', size: 12, fill: '#FFFFFF', scale: u, ...shade }, R.labelX, R.labelY);
    for (let j = 0; j < show; j++) {
      const tc = tl.touches[st.i - show + 1 + j];
      const hh = 8 + clamp(((tc.h ?? 0.6) - 0.3) / 1.5, 0, 1) * (R.hMax - 8);
      ctx.fillStyle = rgba(lk.head, 0.5 + (0.5 * (j + 1)) / show);
      rr(ctx, R.barX + j * 9, R.base - hh, 6, hh, 3);
      ctx.fill();
    }
  }
  ctx.restore();
}

// ---------- v3：HUD + 底部数据面板（基准坐标） ----------
const V3 = {
  labelY: 52, numBase: 124, // 计数上移（最初 104 / 176）：标签顶边约在 5% 屏高，在短视频 App 里紧贴顶部栏下沿
  pill: { y: 60, h: 24 }, // 和数字顶部对齐
  window: 4.5, // 曲线显示最近 4.5 秒，宽度和下面的数据行对齐
};
// 竖屏短视频 App 推荐页里被界面盖住的区域（占视频宽、高的比例），按一台 909×2000 屏的截图量的：
// 9:16 视频按宽度铺满、贴着状态栏放，左右不裁切。顶部栏到 6%；右侧头像/点赞栏从 x 84.5%、y 46% 起到底；
// 左下胶囊 87.3–92.1%，叠两个胶囊时从 82% 起；@昵称 93.9%；文案 98% 起，宽到 x 70%。不同 App、机型会有出入。
export const SAFE_AREA = { top: 0.06, rightX: 0.845, rightY: 0.46, pills: 0.82, pill: 0.873, name: 0.939, textX: 0.7 };
// 左右边距按画面宽度算（不受基准坐标 390 宽的限制）：计数离左边 6.5%，面板左右各留 6%
const EDGE = { hud: 0.065, panel: 0.06 };
// 面板字号（基准单位）。PANEL_SCALE 统一放大：1 = 最初的字号，默认 1.35
export const PANEL_SCALE = 1.35;
const CHART_H = 50; // 曲线区高度（基准单位），原来 34
function panelType(tl) {
  const k = tl.opts.panelScale || PANEL_SCALE;
  const F = PANEL_FONTS[tl.opts.panelFont] || PANEL_FONTS.noto;
  return { k, head: 11 * k, label: 9 * k, cap: 16 * k, unit: 10 * k, font: F, num: F.num || tl.opts.numFont };
}
// 面板底边（数据行基线，占画面高度的比例）
const PANEL_BOTTOM = {
  v3: 0.97, // 贴视频底部（不管 App 界面遮挡）
  v3s: SAFE_AREA.pills - 0.015, // 两个胶囊上方
  v3l: SAFE_AREA.pill - 0.018, // 只避开一个胶囊；有两个胶囊时左边的"节奏"会被盖住
  v3c: SAFE_AREA.pills - 0.015,
};
function v3Geom(tl) {
  const layout = tl.opts.layout;
  const toD = (fx) => (fx * tl.W - tl.ox) / tl.u; // 画面宽度比例 → 基准 x
  const toDy = (fy) => (fy * tl.H) / tl.u; // 画面高度比例 → 基准 y
  const x0 = toD(EDGE.panel), right = toD(1 - EDGE.panel);
  // 贴底的原样版铺满左右；其余右边离点赞栏 2.5% 屏宽
  const x1 = layout === 'v3' ? right : toD(SAFE_AREA.rightX - 0.025);
  const g = { hudX: toD(EDGE.hud), right, pillY: V3.pill.y, type: panelType(tl) };
  // 从数据行基线往上排：数值 → 名称 → 分隔线 → 时间轴 → 曲线 → 标题，间距跟着字号走
  const T = g.type, vb = toDy(PANEL_BOTTOM[layout] || PANEL_BOTTOM.v3s);
  const label = vb - T.cap - 6, divider = label - 0.95 * T.label - 6, axis = divider - 7, bot = axis - 8, top = bot - CHART_H;
  g.stats = { x0, x1, label, value: vb };
  if (layout === 'v3c') {
    // 曲线在计数右边（与数字同高），"第 N 组"上移到标签那一行；底部只留数据行
    g.pillY = V3.labelY - 16;
    g.chart = { x0: toD(0.42), x1: right, head: V3.numBase - 49, top: V3.numBase - 42, bot: V3.numBase - 8, axis: V3.numBase, divider: null };
  } else {
    g.chart = { x0, x1, head: top - 8, top, bot, axis, divider };
  }
  return g;
}

// 计数数字可选字体（均为 SIL OFL）。不同字体按"数字实际高度"换算字号，看起来一样大
export const NUM_FONTS = {
  noto: { family: '"Noto Sans SC"', weight: 900, name: '思源黑体 Black' },
  inter: { family: 'Inter', weight: 800, name: 'Inter ExtraBold' },
  barlow: { family: 'Barlow', weight: 700, name: 'Barlow Bold' },
  bebas: { family: '"Bebas Neue"', weight: 400, name: 'Bebas Neue' },
  poppins: { family: 'Poppins', weight: 700, name: 'Poppins Bold' },
  anton: { family: 'Anton', weight: 400, name: 'Anton' },
  smiley: { family: '"Smiley Sans"', weight: 400, name: '得意黑' },
  zcool: { family: '"ZCOOL QingKe HuangYou"', weight: 400, name: '站酷庆科黄油体' },
  barlowc: { family: '"Barlow Condensed"', weight: 700, style: 'italic', name: 'Barlow Condensed 粗斜体' },
};
// 底部面板的字体：cjk 是中文（标题、名称、单位），num 是数值用的数字字体（null = 跟计数一样）
export const PANEL_FONTS = {
  noto: { name: '思源黑体 + 计数同款数字', cjk: '"Noto Sans SC"', bold: 700, regular: 500, num: null },
  smiley: { name: '得意黑', cjk: '"Smiley Sans", "Noto Sans SC"', bold: 400, regular: 400, num: 'smiley' },
  zcool: { name: '站酷庆科黄油体', cjk: '"ZCOOL QingKe HuangYou", "Noto Sans SC"', bold: 400, regular: 400, num: 'zcool' },
  barlowc: { name: '思源黑体 + Barlow Condensed 粗斜体', cjk: '"Noto Sans SC"', bold: 700, regular: 500, num: 'barlowc' },
  inter: { name: '思源黑体 + Inter', cjk: '"Noto Sans SC"', bold: 700, regular: 500, num: 'inter' },
};
// 数字高度（基准单位）= cap 的字体 CSS
function numFont(tl, cap, key = tl.opts.numFont) {
  const f = NUM_FONTS[key] || NUM_FONTS.noto;
  const css = (px) => `${f.style ? `${f.style} ` : ''}${f.weight} ${px}px ${f.family}`;
  let ratio = tl.cache.get(`capratio|${key}`);
  if (ratio == null) {
    ratio = 0.72;
    if (tl.measureCtx) {
      tl.measureCtx.font = css(100);
      const m = tl.measureCtx.measureText('0');
      if (m.actualBoundingBoxAscent > 10) ratio = m.actualBoundingBoxAscent / 100;
    }
    tl.cache.set(`capratio|${key}`, ratio);
  }
  return css((cap / ratio).toFixed(1));
}
const HALO = 'rgba(0,0,0,0.75)'; // 白字自带的暗色阴影，亮背景上靠它看清

// 球离"最低点"的高度（像素）。曲线的纵轴按最近几秒的最高值自动缩放，普通颠球也有足够起伏
function ballRise(tl, t) {
  const p = ballAt(tl, t);
  if (!p) return null;
  if (tl.groundRef == null) {
    let maxY = -Infinity, minY = Infinity;
    for (let i = 0; i < tl.n; i++) if (!Number.isNaN(tl.by[i])) { maxY = Math.max(maxY, tl.by[i]); minY = Math.min(minY, tl.by[i]); }
    tl.groundRef = maxY;
    tl.riseRange = Math.max(1, maxY - minY);
  }
  return tl.groundRef - p.y;
}
function chartScale(tl, t) {
  let m = 0;
  for (let k = Math.ceil((t - 6) * 10); k <= Math.floor(t * 10); k++) {
    const r = k >= 0 ? ballRise(tl, k / 10) : null;
    if (r != null) m = Math.max(m, r);
  }
  return Math.max(m * 1.1, 0.25 * (tl.riseRange || 1), 1);
}

function comboStats(tl, t, st) {
  if (st.i < 0) return { rhythm: null, interval: null };
  const ts = [];
  for (let i = st.i; i >= 0 && ts.length < 6; i--) {
    if (tl.touches[i].combo !== st.combo) break;
    ts.push(tl.touches[i].t);
  }
  if (ts.length < 2) return { rhythm: null, interval: null };
  const mean = (ts[0] - ts[ts.length - 1]) / (ts.length - 1);
  return { rhythm: Math.round(60 / mean), interval: ts[0] - ts[1] };
}

function drawHudV3(ctx, tl, t, st, lk) {
  const u = tl.u, mask = tl.maskMode, acc = lk.head;
  const G = v3Geom(tl), hx = G.hudX;
  const ink = tl.opts.tint === 'all' ? acc : '#FFFFFF'; // 计数区文字颜色
  ctx.save();
  ctx.transform(u, 0, 0, u, tl.ox, 0);
  // ---- 左上：标签、计数、单位、+1 ----
  const tau = t - st.lastT;
  const num = String(tl.opts.showCount ?? st.count); // showCount 只给单帧预览比较字体用
  const numStyle = { text: num, font: numFont(tl, 55), size: 76, fill: ink, under: HALO, scale: u * 1.15 };
  const nw = textWidth(ctx, tl, numStyle);
  const fade = hudFade(tl, t, [hx - 6, V3.labelY - 16, hx + Math.max(120, nw + 50), V3.numBase + 6]);
  ctx.save();
  ctx.globalAlpha = fade;
  const lw = text(ctx, tl, { text: lab(tl).count, font: '700 12px "Noto Sans SC", sans-serif', size: 12, fill: ink, spacing: 2, under: HALO, scale: u }, hx, V3.labelY);
  // 第二组起在标签后面写上最好成绩，计数清零时也知道之前颠到过多少
  if (st.combo >= 1 && st.bestCombo > 0) {
    text(ctx, tl, { text: `最佳 ${st.bestCombo}`, font: '700 12px "Noto Sans SC", sans-serif', size: 12, fill: acc, under: HALO, scale: u }, hx + lw + 10, V3.labelY);
  }
  const k = Math.exp(-9 * tau), s = 1 + 0.12 * k * Math.cos(19 * tau);
  ctx.save();
  ctx.translate(hx, V3.numBase);
  ctx.scale(s, s);
  ctx.translate(-hx, -V3.numBase);
  text(ctx, tl, numStyle, hx, V3.numBase);
  const flash = 0.6 * Math.exp(-10 * tau);
  if (flash > 0.01 && !mask && st.count > 0) {
    ctx.globalAlpha = fade * flash;
    text(ctx, tl, { ...numStyle, fill: ink === acc ? '#FFFFFF' : acc, under: null }, hx, V3.numBase); // 触球时闪一下另一种颜色
  }
  ctx.restore();
  text(ctx, tl, { text: lab(tl).unit, font: '700 15px "Noto Sans SC", sans-serif', size: 15, fill: ink, under: HALO, scale: u }, hx + nw + 6, V3.numBase);
  if (tau >= 0 && tau < 0.9 && st.count > 0 && !mask) {
    const op = tau < 0.08 ? tau / 0.08 : Math.max(0, 1 - Math.max(0, tau - 0.45) / 0.45);
    ctx.globalAlpha = fade * op;
    text(ctx, tl, { text: '+1', font: numFont(tl, 12), size: 16, fill: acc, under: HALO, scale: u }, hx + nw + 8, V3.numBase - 44 - 6 * (1 - Math.exp(-6 * tau)));
  }
  // 落地或用手接住：数字旁边弹出"落地 · 重新计数" / "用手 · 重新计数"，1.4 秒后淡出
  if (st.dropT != null && !mask) {
    const da = t - st.dropT;
    if (da < 1.4) {
      const op = da < 0.1 ? da / 0.1 : Math.min(1, (1.4 - da) / 0.4);
      const label = st.dropKind === 'hand' ? lab(tl).hand : lab(tl).drop;
      const f = '700 11px "Noto Sans SC", sans-serif';
      const cw = textWidth(ctx, tl, { text: label, font: f, size: 11, scale: u }) + 18, ch = 22;
      const cx0 = hx + nw + 32, cy0 = V3.numBase - 34 - 4 * (1 - Math.exp(-8 * da));
      ctx.globalAlpha = fade * op;
      ctx.fillStyle = rgba(DROP_COLOR, 0.88);
      rr(ctx, cx0, cy0, cw, ch, ch / 2);
      ctx.fill();
      text(ctx, tl, { text: label, font: f, size: 11, fill: '#FFFFFF', scale: u }, cx0 + 9, cy0 + 15);
    }
  }
  ctx.restore();
  // ---- 右上：第 N 组 ----
  if (st.combo >= 0) {
    const label = `第 ${st.combo + 1} 组`;
    const lw2 = textWidth(ctx, tl, { text: label, font: '700 11px "Noto Sans SC", sans-serif', size: 11, scale: u });
    const pw = lw2 + 30, px = G.right - pw, P = { y: G.pillY, h: V3.pill.h };
    ctx.save();
    ctx.globalAlpha = hudFade(tl, t, [px, P.y, px + pw, P.y + P.h]);
    if (!mask) {
      ctx.fillStyle = 'rgba(10,12,16,0.55)';
      rr(ctx, px, P.y, pw, P.h, P.h / 2);
      ctx.fill();
      ctx.fillStyle = acc;
      ctx.beginPath();
      ctx.arc(px + 12, P.y + P.h / 2, 3, 0, TAU);
      ctx.fill();
    }
    text(ctx, tl, { text: label, font: '700 11px "Noto Sans SC", sans-serif', size: 11, fill: ink, scale: u }, px + 21, P.y + 16);
    ctx.restore();
  }
  drawPanelV3(ctx, tl, t, st, acc, G);
  ctx.restore();
}

function drawPanelV3(ctx, tl, t, st, acc, G) {
  drawChartV3(ctx, tl, t, st, acc, G.chart, G.type);
  drawStatsV3(ctx, tl, t, st, G.stats, G.type, acc);
}

// 面板文字颜色：tint 不是 none 时用曲线的颜色（a 为不透明度）
const panelInk = (tl, acc, a = 1) => (tl.opts.tint === 'none' ? `rgba(255,255,255,${a})` : rgba(acc, a));
const cjkFont = (T, weight, z) => `${weight} ${z.toFixed(1)}px ${T.font.cjk}, sans-serif`;
const labV3 = (ctx, tl, T, s, x, y, align, z, fill) => text(ctx, tl, { text: s, font: cjkFont(T, T.font.bold, z), size: z, fill, spacing: 1, under: HALO, scale: tl.u }, x, y, align);
const subV3 = (ctx, tl, T, s, x, y, align, z, fill) => text(ctx, tl, { text: s, font: cjkFont(T, T.font.regular, z), size: z, fill, under: HALO, scale: tl.u }, x, y, align);

// 球的高度曲线（纵轴按最近 6 秒的最高值自动缩放）
function drawChartV3(ctx, tl, t, st, acc, C, T) {
  const u = tl.u, mask = tl.maskMode;
  const { x0, x1 } = C;
  const scale = chartScale(tl, t);
  const hAt = (ts) => { const r = ballRise(tl, ts); return r == null ? null : clamp(r / scale, 0, 1); };
  const h0 = hAt(t), h1 = hAt(t - 1 / 30);
  const rising = h0 != null && h1 != null && h0 > h1 + 0.002;
  // 曲线在顶部（v3c）时标题保持原字号，那里空间小
  const hz = C.divider == null ? 11 : T.head;
  labV3(ctx, tl, T, lab(tl).chart, x0, C.head, 'left', hz, panelInk(tl, acc));
  if (h0 != null) labV3(ctx, tl, T, rising ? '上升 ↑' : '下落 ↓', x1, C.head, 'right', hz, acc);
  if (mask) return;
  const t0 = Math.max(0, t - V3.window); // 曲线左端的时刻
  const X = (ts) => x0 + ((ts - t0) / V3.window) * (x1 - x0);
  const cx = X(t);
  const Y = (h) => C.bot - h * (C.bot - C.top);
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = acc;
  ctx.lineWidth = 1.8;
  ctx.shadowColor = acc;
  ctx.shadowBlur = 6 * u;
  ctx.beginPath();
  let pen = false;
  for (let k = Math.ceil(t0 * 30); k <= Math.floor(t * 30); k++) {
    const h = k >= 0 ? hAt(k / 30) : null;
    if (h == null) { pen = false; continue; }
    if (!pen) { ctx.moveTo(X(k / 30), Y(h)); pen = true; } else ctx.lineTo(X(k / 30), Y(h));
  }
  if (h0 != null) ctx.lineTo(cx, Y(h0));
  ctx.stroke();
  ctx.shadowBlur = 0;
  // 游标和当前点
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx, C.top - 4);
  ctx.lineTo(cx, C.axis + 2);
  ctx.stroke();
  if (h0 != null) {
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(cx, Y(h0), 3.2, 0, TAU);
    ctx.fill();
  }
  // 时间刻度（每 0.5 秒一个点）和触球标记
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  for (let k = Math.ceil(t0 * 2); X(k / 2) <= x1 + 0.01; k++) {
    ctx.beginPath();
    ctx.arc(X(k / 2), C.axis, 1, 0, TAU);
    ctx.fill();
  }
  ctx.fillStyle = acc;
  for (let i = st.i; i >= 0 && tl.touches[i].t >= t0; i--) ctx.fillRect(X(tl.touches[i].t) - 0.75, C.axis - 4, 1.5, 8);
  // 分隔线
  if (C.divider != null) {
    ctx.fillStyle = 'rgba(255,255,255,0.18)';
    ctx.fillRect(x0, C.divider, x1 - x0, 0.8);
  }
  ctx.restore();
}

// 数据行：节奏 / 触球间隔 / 最高一颠
function drawStatsV3(ctx, tl, t, st, S, T, acc) {
  const u = tl.u;
  const { x0, x1 } = S;
  const cs = comboStats(tl, t, st);
  const cols = [
    ['节奏', cs.rhythm != null ? String(cs.rhythm) : '—', cs.rhythm != null ? ' 次/分' : '', x0, 'left'],
    [lab(tl).interval, cs.interval != null ? cs.interval.toFixed(2) : '—', cs.interval != null ? ' 秒' : '', (x0 + x1) / 2, 'center'],
    [lab(tl).best, st.best > 0 ? st.best.toFixed(1) : '—', st.best > 0 ? ' m' : '', x1, 'right'],
  ];
  for (const [name, val, unit, x, align] of cols) {
    subV3(ctx, tl, T, name, x, S.label, align, T.label, panelInk(tl, acc, 0.85));
    // 数值和计数用同一种数字字体；没有数据时的"—"用思源黑体（部分数字字体不含这个符号）
    const dash = `900 ${(1.375 * T.cap).toFixed(1)}px "Noto Sans SC", sans-serif`;
    const vs = { text: val, font: val === '—' ? dash : numFont(tl, T.cap, T.num), size: 1.375 * T.cap, fill: panelInk(tl, acc), under: HALO, scale: u };
    const us = { text: unit, font: cjkFont(T, T.font.regular, T.unit), size: T.unit, fill: panelInk(tl, acc, 0.85), under: HALO, scale: u };
    const vw = textWidth(ctx, tl, vs), uw = unit ? textWidth(ctx, tl, us) : 0;
    const left = align === 'left' ? x : align === 'center' ? x - (vw + uw) / 2 : x - vw - uw;
    text(ctx, tl, vs, left, S.value);
    if (unit) text(ctx, tl, us, left + vw, S.value);
  }
}

function drawBanner(ctx, tl, t, styleId) {
  const b = bannerAt(tl, t);
  if (!b) return;
  const lk = look(tl.styles, styleId, b.n);
  const u = tl.u;
  let off = 0;
  if (b.age < 0.2) off = -440 * Math.pow(1 - b.age / 0.2, 3);
  else if (b.age > 1.1) off = 440 * Math.pow((b.age - 1.1) / 0.3, 2);
  ctx.save();
  ctx.transform(u, 0, 0, u, tl.ox, 0);
  ctx.translate(195 + off, 250);
  ctx.rotate(-0.12);
  const paint = (g, k) => {
    g.shadowColor = lk.glow;
    g.shadowBlur = 30 * k;
    g.fillStyle = lk.glow;
    g.fillRect(-260, -36, 520, 72);
    g.shadowBlur = 0;
    g.fillStyle = 'rgba(255,255,255,0.9)';
    g.fillRect(-260, -44, 520, 3);
    g.fillRect(-260, 41, 520, 3);
    g.fillStyle = '#07090D';
    g.font = '900 46px "Noto Sans SC", sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'alphabetic';
    g.fillText(b.text, 0, 16);
  };
  const pad = 50;
  const key = `banner|${b.text}|${lk.glow}`;
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
  if (spr) ctx.drawImage(spr, -260 - pad, -44 - pad, 520 + 2 * pad, 88 + 2 * pad);
  else paint(ctx, u);
  ctx.restore();
}

/**
 * 画一帧。
 * @param ctx      画布 2D 上下文（尺寸 = 轨迹数据的 width×height）
 * @param tl       buildTimeline() 的结果
 * @param t        原视频时间（秒）
 * @param styleId  neon / flame / comet / broadcast
 * @param video    当前视频帧的提供者，传 null 表示视频在画布下面（实时预览），这时不做震屏、不抠球：
 *                 { full(ctx, transformed) 画整帧：transformed=false 时可直接写像素（不受变换影响），
 *                   patch(ctx, x, y, r)   在当前变换下画以 (x,y) 为中心、半径 r 的那一小块原画面,
 *                   small()               可选：1/4 分辨率整帧 {canvas, ds}，有它才做渐进模糊 }
 */
export const PARTS = ['scrim', 'trail', 'ring', 'particles', 'touchfx', 'apex', 'hud', 'banner'];

export function drawFrame(ctx, tl, t, styleId, video, dbg) {
  const only = dbg && dbg.only;
  const on = (k) => !only || only === k;
  const st = stateAt(tl, t);
  const lk = look(tl.styles, styleId, st.count);
  const [sx, sy] = shakeAt(tl, t);
  const p = ballAt(tl, t);
  if (only) {
    if (on('scrim')) drawProtection(ctx, tl, t, st, video);
    if (on('trail')) drawTrail(ctx, tl, t, lk);
    if (on('ring') && p) drawHugRing(ctx, tl, p, lk);
    if (on('particles')) drawParticles(ctx, tl, t, lk);
    if (on('touchfx')) { drawTouchFx(ctx, tl, t, styleId); drawDropFx(ctx, tl, t); }
    if (on('apex')) drawApexLabel(ctx, tl, t);
    if (on('hud')) drawHud(ctx, tl, t, st, lk);
    if (on('banner')) drawBanner(ctx, tl, t, styleId);
    return;
  }
  const shaking = sx !== 0 || sy !== 0;
  // 震屏时整帧平移并略放大（避免露出黑边）
  const shakeXf = () => {
    ctx.translate(sx, sy);
    ctx.translate(tl.W / 2, tl.H / 2);
    ctx.scale(1.03, 1.03);
    ctx.translate(-tl.W / 2, -tl.H / 2);
  };
  if (video) {
    ctx.save();
    if (shaking) shakeXf();
    video.full(ctx, shaking, true); // true：允许顺带查表压暗（见 lightScrimLUT），片尾贴最后一帧时不压
    ctx.restore();
  }
  // 可读性保护层（压在视频上、特效下面；真球是从原画面抠回来的，所以不会被压暗或模糊，反而更突出）
  drawProtection(ctx, tl, t, st, video);
  ctx.save();
  ctx.translate(sx, sy);
  // 叠放顺序：拖尾 → 粒子/触球爆点 → 球 → 贴球细圈 → 标注
  drawTrail(ctx, tl, t, lk);
  drawParticles(ctx, tl, t, lk);
  drawTouchFx(ctx, tl, t, styleId);
  drawDropFx(ctx, tl, t);
  if (p && video) {
    // 把真实的球从原画面里抠回来，盖在特效上面
    ctx.save();
    ctx.translate(-sx, -sy);
    if (shaking) shakeXf();
    const r = tl.rBall * 1.04;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, TAU);
    ctx.clip();
    video.patch(ctx, p.x, p.y, r);
    ctx.restore();
  }
  if (p) drawHugRing(ctx, tl, p, lk);
  drawApexLabel(ctx, tl, t);
  if ((lk.tag || LAYOUTS[tl.opts.layout].v3) && p && st.count >= 1) drawTag(ctx, tl, p, st.count, lk.head, tl.touches[st.i]?.move);
  drawMovePop(ctx, tl, t, lk);
  ctx.restore();
  drawHud(ctx, tl, t, st, lk);
  drawBanner(ctx, tl, t, styleId);
}

// ---------- 片尾成绩卡 ----------
export function endCardStats(tl) {
  let best = null;
  for (const c of tl.combos) if (!best || c.count > best.count) best = c;
  const touches = best ? tl.touches.filter((tc) => tc.t >= best.from - 1e-6 && tc.t <= best.to + 1e-6) : [];
  const bestH = Math.max(0, ...tl.apex.map((a) => a.height_m));
  const span = touches.length > 1 ? touches[touches.length - 1].t - touches[0].t : 0;
  return {
    count: best ? best.count : 0,
    secs: best ? Math.max(1, Math.round(best.to - best.from)) : 0,
    bestH,
    rhythm: span > 0 ? Math.round(((touches.length - 1) / span) * 60) : null, // 最长那组的平均节奏（次/分）
    touches,
    groups: tl.combos.length,
    total: tl.touches.length,
  };
}

// 片尾背景：最后一帧放大模糊、压暗，和成片自然衔接；顶部透一点曲线颜色的光。没有画面时用深色
const END_BLUR = 10; // 1/4 分辨率画布上的模糊半径（全尺寸约 40 像素）
function drawEndBackground(ctx, tl, video, acc) {
  const { W, H } = tl;
  if (video && video.small && tl.makeCanvas) {
    const { canvas: small } = video.small();
    const sw = small.width, sh = small.height;
    if (!tl.endLayer) tl.endLayer = tl.makeCanvas(sw, sh);
    const g = tl.endLayer.getContext('2d');
    g.filter = `blur(${END_BLUR}px)`;
    const m = Math.ceil(END_BLUR * 2.5); // 同 drawEdgeBlur：让模糊的透明羽化边落到画布外
    g.drawImage(small, -m, -m, sw + 2 * m, sh + 2 * m);
    g.filter = 'none';
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tl.endLayer, 0, 0, W, H);
    ctx.restore();
    ctx.fillStyle = 'rgba(6,8,12,0.74)';
  } else {
    ctx.fillStyle = '#07090D';
  }
  ctx.fillRect(0, 0, W, H);
  const rg = ctx.createRadialGradient(W / 2, H * 0.2, 0, W / 2, H * 0.2, W * 0.8);
  rg.addColorStop(0, rgba(acc, 0.16));
  rg.addColorStop(1, rgba(acc, 0));
  ctx.fillStyle = rg;
  ctx.fillRect(0, 0, W, H);
}

/**
 * 片尾成绩卡，和成片里的 HUD 同一套风格：曲线颜色做强调色，Bebas Neue 大数字，数据用底部面板的数字字体。
 * @param tc     片尾开始后的秒数
 * @param opts   { text: 片尾最下面一行自定义文字（可空）, nickname, styleId, lastFrame: 同 drawFrame 的 video（最后一帧） }
 */
export function drawEndCard(ctx, tl, tc, opts) {
  const { W, u } = tl;
  const stats = endCardStats(tl);
  const acc = look(tl.styles, opts.styleId, stats.count).head;
  const T = panelType(tl);
  const ease = (x) => 1 - Math.pow(1 - clamp(x, 0, 1), 3);
  const fadeIn = ease(tc / 0.35);
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

  // 标题 + 大数字（从 0 滚到最终数），右边小字单位（颠球"次"、运球"下"）
  if (opts.nickname) say(`@${opts.nickname}`, cx, 112, 'center', cjk(700, 15), 15, 'rgba(255,255,255,0.85)');
  say(stats.groups > 1 ? lab(tl).endTitleMulti : lab(tl).endTitle, cx, 150, 'center', cjk(700, 14), 14, 'rgba(255,255,255,0.82)', { spacing: 5 });
  const shown = Math.round(stats.count * ease((tc - 0.15) / 0.8));
  const big = { text: String(shown), font: numFont(tl, 150), size: 200, fill: acc, glow: rgba(acc, 0.45), blur: 36, scale: u };
  const bw = textWidth(ctx, tl, { ...big, text: String(stats.count) }); // 按最终数字定位，滚动时不左右跳
  text(ctx, tl, big, cx, 330, 'center');
  say(lab(tl).unit, cx + bw / 2 + 8, 330, 'left', cjk(700, 20), 20, '#FFFFFF');
  // 颠了不止一组时说明一下：最长那组是大数字，下面是总数
  if (stats.groups > 1) say(fmt(lab(tl).endTotal, { g: stats.groups, n: stats.total }), cx, 374, 'center', cjk(T.font.regular, 15), 15, 'rgba(255,255,255,0.8)');

  // 三项数据（同底部面板：名称用曲线颜色，数值用面板的数字字体）
  const cells = [
    [lab(tl).best, stats.bestH > 0 ? stats.bestH.toFixed(1) : '—', stats.bestH > 0 ? ' m' : '', x0, 'left'],
    ['用时', String(stats.secs), ' 秒', cx, 'center'],
    ['平均节奏', stats.rhythm != null ? String(stats.rhythm) : '—', stats.rhythm != null ? ' 次/分' : '', x1, 'right'],
  ];
  const vCap = 26, uz = 13;
  for (const [name, val, unit, x, align] of cells) {
    say(name, x, 432, align, cjk(T.font.regular, 13), 13, rgba(acc, 0.9));
    const vs = { text: val, font: val === '—' ? cjk(900, 34) : numFont(tl, vCap, T.num), size: 36, fill: '#FFFFFF', scale: u };
    const us = { text: unit, font: cjk(T.font.regular, uz), size: uz, fill: 'rgba(255,255,255,0.75)', scale: u };
    const vw = textWidth(ctx, tl, vs), uw = unit ? textWidth(ctx, tl, us) : 0;
    const left = align === 'left' ? x : align === 'center' ? x - (vw + uw) / 2 : x - vw - uw;
    text(ctx, tl, vs, left, 468);
    if (unit) text(ctx, tl, us, left + vw, 468);
  }
  ctx.fillStyle = 'rgba(255,255,255,0.16)';
  ctx.fillRect(x0, 494, x1 - x0, 0.8);

  // 每一颠的高度：曲线颜色的柱子，最高那一颠用白色
  say(stats.groups > 1 ? lab(tl).endBarsMulti : lab(tl).endBars, x0, 524, 'left', cjk(T.font.regular, 13), 13, rgba(acc, 0.9));
  const marks = [10, 20, 30, 50, 100].filter((m) => m <= stats.count).slice(-2).map((m) => fmt(lab(tl).marks, { n: m }));
  const moves = lab(tl).moves ? Object.entries(lab(tl).moves).map(([k, name]) => [name, tl.touches.filter((t2) => t2.move === k).length]).filter(([, c]) => c > 0) : [];
  if (moves.length) say(moves.map(([name, c]) => `${name} ${c}`).join(' · '), x1, 524, 'right', cjk(700, 13), 13, '#FFFFFF');
  else if (marks.length) say(marks.join(' · '), x1, 524, 'right', cjk(T.font.regular, 12), 12, 'rgba(255,255,255,0.6)');
  const N = stats.touches.length;
  if (N) {
    const span = x1 - x0, base = 610;
    const gap = Math.min(3, span / N / 3.4);
    const bw2 = Math.min(10, (span - (N - 1) * gap) / N);
    const hs = stats.touches.map((t2) => t2.h ?? 0.6);
    const iMax = hs.indexOf(Math.max(...hs));
    const startX = x0 + (span - (N * bw2 + (N - 1) * gap)) / 2;
    stats.touches.forEach((t2, i) => {
      const grow = ease((tc - 0.3 - i * (0.6 / N)) / 0.25);
      if (grow <= 0) return;
      const hh = (8 + clamp(hs[i] / Math.max(hs[iMax], 0.3), 0, 1) * 62) * grow; // 按这一组的最高一颠归一
      ctx.fillStyle = i === iMax ? '#FFFFFF' : rgba(acc, 0.85);
      rr(ctx, startX + i * (bw2 + gap), base - hh, bw2, hh, Math.min(3, bw2 / 2));
      ctx.fill();
    });
  }

  // 挑战语 + 可选的一行自定义文字
  const cta = ease((tc - 0.9) / 0.3);
  ctx.globalAlpha = fadeIn * cta;
  const ask = (y, align, x) => {
    // "你能颠过 N 个吗？"，数字用曲线颜色
    const parts = [[lab(tl).askPre, '#FFFFFF'], [String(stats.count), acc], [lab(tl).askPost, '#FFFFFF']];
    const ws = parts.map(([s2]) => textWidth(ctx, tl, { text: s2, font: cjk(900, 21), size: 21, scale: u }));
    let px = align === 'center' ? x - ws.reduce((a2, b2) => a2 + b2, 0) / 2 : x;
    parts.forEach(([s2, fill], i) => { say(s2, px, y, 'left', cjk(900, 21), 21, fill); px += ws[i]; });
  };
  ask(668, 'center', cx);
  if (opts.text) drawTagline(ctx, tl, opts.text, cx, 688, acc, '#0A0C10', cjk(900, 17));
  ctx.restore();
}

// 片尾最下面一行自定义文字（--endcard-text）：强调色圆角胶囊，居中，顶边在 y（基准坐标，调用时已按 u 缩放）
function drawTagline(ctx, tl, s, cx, y, fill, ink, font) {
  const pw = textWidth(ctx, tl, { text: s, font, size: 17, scale: tl.u }) + 44;
  ctx.fillStyle = fill;
  rr(ctx, cx - pw / 2, y, pw, 50, 25);
  ctx.fill();
  text(ctx, tl, { text: s, font, size: 17, fill: ink, scale: tl.u }, cx, y + 31, 'center');
}

// 给其他项目的特效（render/jumprope_fx.js）复用的内部工具；颠球自己的画法不受影响
export { TAU, DESIGN_W, DESIGN_H, HALO, EDGE, V3, DROP_COLOR, clamp, smooth, rr, hash01, text, textWidth, numFont, panelType, v3Geom, panelInk, cjkFont, labV3, subV3, drawProtection, drawEndBackground, drawBanner, drawTagline };
