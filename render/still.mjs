// 单帧预览：调特效时不用渲染整段视频。取一帧，按几种方案各画一张，并排拼成一张图，标出实测对比度。
//
//   node render/still.mjs --track out/x.track.json --t 3.27 --out out/still.png
//        [--style neon] [--variants v1:plates,v2:edges,v2:veil,v2:mix] [--panel 360]
//
// 每个方案还会存一张原尺寸图 out/still.<布局>-<保护层>.png。
// --end 1.5：画片尾第 1.5 秒（--t 那一帧当作最后一帧），可加 --endcard-text "一行字"、--nickname 昵称。
// 对比度：只画"视频 + 保护层"（不画文字），量各行文字所在区域背景亮度的 75 分位，换算白字对比度。

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createCanvas, GlobalFonts, ImageData } from '@napi-rs/canvas';
import * as fx from './fx.js';
import { registerFonts } from './fonts.mjs';
import { rawVideoSource } from './video_source.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
registerFonts(GlobalFonts);

const args = { style: 'neon', variants: 'v1:plates,v2:edges,v2:veil,v2:mix', panel: 360, t: 3.27 };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 2) args[argv[i].replace(/^--/, '')] = argv[i + 1];
const NAMES = { plates: '暗影底板', edges: '上下渐进模糊', veil: '整页灰度层', mix: '轻灰度 + 渐进模糊', light: '轻渐变' };
const LAYOUT_NAMES = { v3: '面板贴视频底部', v3s: '底部面板（避开 App 界面）', v3l: '底部面板压低（只避开一个胶囊）', v3c: '曲线放上面、底部一行数据' };

const track = JSON.parse(fs.readFileSync(args.track, 'utf-8'));
const styles = JSON.parse(fs.readFileSync(path.join(HERE, 'styles.json'), 'utf-8'));
const { width: W, height: H, fps } = track;
const t = parseFloat(args.t);
const f = Math.round(t * fps);
const raw = spawnSync('ffmpeg', ['-v', 'error', '-i', track.source.path, '-filter_complex', `${track.source.filter},select=eq(n\\,${f})`,
  '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'], { maxBuffer: W * H * 8 }).stdout;
if (raw.length < W * H * 4) throw new Error(`取不到第 ${f} 帧`);

const lin = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
const LUT = Array.from({ length: 256 }, (_, i) => lin(i));
const lumAt = (d, i) => 0.2126 * LUT[d[i]] + 0.7152 * LUT[d[i + 1]] + 0.0722 * LUT[d[i + 2]];
const rectPx = (tl, b) => [Math.round(tl.ox + b[0] * tl.u), Math.round(b[1] * tl.u), Math.round(tl.ox + b[2] * tl.u), Math.round(b[3] * tl.u)];

// 整块区域背景亮度（75 分位）→ 白字对比度。适用于"文字下面垫一块"的方案
function measureArea(canvas, tl) {
  const d = canvas.data();
  const out = {};
  for (const [name, b] of Object.entries(fx.LAYOUTS[tl.opts.layout].rects)) {
    const [x0, y0, x1, y1] = rectPx(tl, b);
    const L = [];
    for (let y = y0; y < y1; y += 2) for (let x = x0; x < x1; x += 2) L.push(lumAt(d, (y * W + x) * 4));
    L.sort((a, c) => a - c);
    out[name] = 1.05 / (L[Math.floor(L.length * 0.75)] + 0.05);
  }
  return out;
}

// 笔画周边：文字字形外 2–5 像素那一圈的背景亮度（75 分位）→ 白字对比度。
// 适用于靠文字自带暗色阴影的方案；人眼判断字清不清楚，看的正是笔画紧挨着的背景。
function dilate(src, r) {
  const a = new Uint8Array(W * H), b = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    let run = -1e9;
    for (let x = 0; x < W; x++) { if (src[y * W + x]) run = x; if (x - run <= r) a[y * W + x] = 1; }
    run = 1e9;
    for (let x = W - 1; x >= 0; x--) { if (src[y * W + x]) run = x; if (run - x <= r) a[y * W + x] = 1; }
  }
  for (let x = 0; x < W; x++) {
    let run = -1e9;
    for (let y = 0; y < H; y++) { if (a[y * W + x]) run = y; if (y - run <= r) b[y * W + x] = 1; }
    run = 1e9;
    for (let y = H - 1; y >= 0; y--) { if (a[y * W + x]) run = y; if (run - y <= r) b[y * W + x] = 1; }
  }
  return b;
}
function measureRing(finalCanvas, maskCanvas, tl) {
  const d = finalCanvas.data(), m = maskCanvas.data();
  const g = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) g[i] = m[i * 4] > 20 ? 1 : 0;
  const s = W / 1080;
  const inner = dilate(g, Math.round(1 * s)), outer = dilate(g, Math.round(5 * s));
  const out = {};
  for (const [name, b] of Object.entries(fx.LAYOUTS[tl.opts.layout].rects)) {
    const [x0, y0, x1, y1] = rectPx(tl, b);
    const L = [];
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const i = y * W + x;
      if (outer[i] && !inner[i]) L.push(lumAt(d, i * 4));
    }
    if (!L.length) continue;
    L.sort((a, c) => a - c);
    out[name] = 1.05 / (L[Math.floor(L.length * 0.75)] + 0.05);
  }
  return out;
}

// 短视频 App 推荐页会盖住的区域（fx.SAFE_AREA，单位是画面宽高的比例）
function drawSafeArea(ctx) {
  const D = fx.SAFE_AREA;
  const zones = [[0, 0, 1, D.top, '顶部栏'], [D.rightX, D.rightY, 1, 1, '头像 · 点赞 · 评论'], [0, D.pills, D.textX, 1, '胶囊 · @昵称 · 文案']];
  ctx.save();
  for (const [x0, y0, x1, y1, label] of zones) {
    const [a, b, c, d] = [x0 * W, y0 * H, x1 * W, y1 * H];
    ctx.fillStyle = 'rgba(255,60,60,0.28)';
    ctx.fillRect(a, b, c - a, d - b);
    ctx.strokeStyle = 'rgba(255,90,90,0.9)';
    ctx.setLineDash([10, 8]);
    ctx.lineWidth = 3;
    ctx.strokeRect(a + 1.5, b + 1.5, c - a - 3, d - b - 3);
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '700 28px "Noto Sans SC"';
    if (c - a < 300) {
      ctx.save();
      ctx.translate((a + c) / 2 + 10, b + 30);
      ctx.rotate(Math.PI / 2);
      ctx.fillText(label, 0, 0);
      ctx.restore();
    } else ctx.fillText(label, a + 24, b + 44);
  }
  ctx.restore();
}

// 方案写法：布局:保护层[:风格][:safe][:font=字体][:count=N]，例如 v3s:light:replay:font=bebas
//   safe 叠红色的 App 界面遮挡区；ps=1 面板字号缩放（默认 1.35）；tint=none|panel|all 文字颜色；pfont=noto|smiley|zcool|barlowc|inter 底部面板字体
const variants = args.variants.split(',').map((v) => {
  const [layout, protect, ...rest] = v.split(':');
  const o = { layout, protect, style: args.style, safe: false, numFont: undefined };
  for (const r of rest) {
    if (r === 'safe') o.safe = true;
    else if (r.startsWith('font=')) o.numFont = r.slice(5);
    else if (r.startsWith('count=')) o.showCount = parseInt(r.slice(6), 10);
    else if (r.startsWith('ps=')) o.panelScale = parseFloat(r.slice(3));
    else if (r.startsWith('tint=')) o.tint = r.slice(5);
    else if (r.startsWith('pfont=')) o.panelFont = r.slice(6);
    else if (r) o.style = r;
  }
  return o;
});
// --crop 0,0.34：每格只显示画面的这一段高度（比较字体时用）；--cols：每行几格
const [cy0, cy1] = (args.crop || '0,1').split(',').map(Number);
const nCells = variants.length;
const cols = parseInt(args.cols || String(nCells), 10);
const pw = parseInt(args.panel, 10), cap = 64, gap = 10;
const ph = Math.round((pw * H * (cy1 - cy0)) / W);
const rowsN = Math.ceil(nCells / cols);
const sheet = createCanvas(cols * pw + (cols - 1) * gap, rowsN * (ph + cap) + (rowsN - 1) * gap);
const sc = sheet.getContext('2d');
sc.fillStyle = '#FFFFFF';
sc.fillRect(0, 0, sheet.width, sheet.height);
const mk = (w, h) => createCanvas(w, h);
const canvas = createCanvas(W, H), ctx = canvas.getContext('2d');

const maskCanvas = createCanvas(W, H), mctx = maskCanvas.getContext('2d');
const cellXY = (k) => [(k % cols) * (pw + gap), Math.floor(k / cols) * (ph + cap + gap)];
variants.forEach((v, k0) => {
  const k = k0;
  const tl = fx.buildTimeline(track, styles, mk, v);
  const video = rawVideoSource(createCanvas, ImageData, W, H, Math.ceil(tl.rBall * 2.2) + 4);
  video.set(new Uint8ClampedArray(raw.buffer.slice(raw.byteOffset, raw.byteOffset + W * H * 4)));
  // 完整一帧
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  if (args.end) {
    fx.drawEndCard(ctx, tl, parseFloat(args.end), { text: args['endcard-text'] || '', nickname: args.nickname || '', styleId: v.style, lastFrame: video });
    const [x, y] = cellXY(k);
    sc.drawImage(canvas, 0, cy0 * H, W, (cy1 - cy0) * H, x, y, pw, ph);
    sc.fillStyle = '#111111';
    sc.font = '700 15px "Noto Sans SC"';
    sc.fillText(`${String.fromCharCode(65 + k0)}  片尾第 ${args.end} 秒`, x + 6, y + ph + 22);
    fs.writeFileSync(path.join(path.dirname(args.out), 'still.end.png'), canvas.toBuffer('image/png'));
    return;
  }
  fx.drawFrame(ctx, tl, t, v.style, video);
  let c, how;
  if (v.protect === 'light') {
    // 字形遮罩：只画 HUD 文字的纯白字形
    mctx.setTransform(1, 0, 0, 1, 0, 0);
    mctx.fillStyle = '#000';
    mctx.fillRect(0, 0, W, H);
    tl.maskMode = true;
    fx.drawFrame(mctx, tl, t, v.style, null, { only: 'hud' });
    tl.maskMode = false;
    c = measureRing(canvas, maskCanvas, tl);
    how = '笔画周边';
  } else {
    // 只画视频 + 保护层（不画文字），量整块背景
    const tmp = createCanvas(W, H), tctx = tmp.getContext('2d');
    video.full(tctx, false);
    fx.drawFrame(tctx, tl, t, v.style, video, { only: 'scrim' });
    c = measureArea(tmp, tl);
    how = '整块背景';
  }
  if (v.safe) drawSafeArea(ctx);
  const tag = `${v.layout}-${v.protect}-${v.style}${v.numFont ? `-${v.numFont}` : ''}${v.panelScale ? `-ps${v.panelScale}` : ''}${v.tint ? `-${v.tint}` : ''}${v.panelFont ? `-pf${v.panelFont}` : ''}${v.safe ? '-safe' : ''}`;
  const file = path.join(path.dirname(args.out), `still.${tag}.png`);
  fs.writeFileSync(file, canvas.toBuffer('image/png'));
  const [x, y] = cellXY(k);
  sc.drawImage(canvas, 0, cy0 * H, W, (cy1 - cy0) * H, x, y, pw, ph);
  sc.fillStyle = '#111111';
  sc.font = '700 15px "Noto Sans SC"';
  const styleName = styles.styles[v.style].name;
  const fontName = v.numFont ? ` · ${fx.NUM_FONTS[v.numFont].name}` : '';
  const what = LAYOUT_NAMES[v.layout] ? `${LAYOUT_NAMES[v.layout]}` : `${NAMES[v.protect]} · ${styleName}${fontName}`;
  const TINTS = { none: '白字', panel: '面板字用曲线颜色', all: '全部字用曲线颜色' };
  const ps = (v.panelScale ? ` · 字号 ×${v.panelScale}` : '') + (v.tint ? ` · ${TINTS[v.tint]}` : '') + (v.panelFont ? ` · ${fx.PANEL_FONTS[v.panelFont].name}` : '');
  sc.fillText(`${String.fromCharCode(65 + k0)}  ${what}${ps}${v.safe ? ' · App 界面遮挡示意' : ''}`, x + 6, y + ph + 22);
  sc.font = '500 12px "Noto Sans SC"';
  sc.fillStyle = '#444444';
  const min = Math.min(...Object.values(c));
  sc.fillText(`白字对比度（${how}）：最低 ${min.toFixed(1)}`, x + 6, y + ph + 42);
  sc.fillText(Object.entries(c).map(([n, val]) => `${n} ${val.toFixed(1)}`).join(' · '), x + 6, y + ph + 58);
  console.log(`${String.fromCharCode(65 + k0)} ${v.layout}:${v.protect}:${v.style}  ${how} ${Object.entries(c).map(([n, val]) => `${n} ${val.toFixed(2)}`).join('，')}  → ${file}`);
});
fs.writeFileSync(args.out, sheet.toBuffer('image/png'));
console.log(args.out);
