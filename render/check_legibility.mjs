// 用真实像素核对 HUD 可读性：每 0.5 秒取一帧，只画"视频 + 顶部渐变 + 底板"（不画文字），
// 量文字所在区域的亮度（75 分位，偏保守），算白字对比度。
//   node render/check_legibility.mjs out/<名>.track.json [--off]   --off：不加保护层（对照）
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createCanvas, ImageData } from '@napi-rs/canvas';
import * as fx from './fx.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const track = JSON.parse(fs.readFileSync(process.argv[2], 'utf-8'));
const off = process.argv.includes('--off');
const styles = JSON.parse(fs.readFileSync(path.join(HERE, 'styles.json'), 'utf-8'));
const tl = fx.buildTimeline(track, styles, (w, h) => createCanvas(w, h), { layout: 'v1', protect: 'plates' });
const { W, H, u, ox } = tl;
const raw = spawnSync('ffmpeg', ['-v', 'error', '-i', track.source.path, '-filter_complex', `${track.source.filter},fps=2`,
  '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'], { maxBuffer: 2 ** 31 }).stdout;
const fb = W * H * 4;
const lin = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
const LUT = Array.from({ length: 256 }, (_, i) => lin(i));
// 文字所在的区域（基准坐标）：标签行、数字、最高一颠行
const areas = { 标签: [20, 97, 120, 114], 数字: [20, 128, 120, 220], 最高一颠: [20, 230, 150, 246] };
const canvas = createCanvas(W, H), ctx = canvas.getContext('2d');
const worst = {};
for (let k = 0; k * fb < raw.length; k++) {
  const t = k * 0.5;
  ctx.putImageData(new ImageData(new Uint8ClampedArray(raw.buffer, raw.byteOffset + k * fb, fb), W, H), 0, 0);
  if (!off) fx.drawFrame(ctx, tl, t, 'neon', null, { only: 'scrim' });
  const d = canvas.data();
  for (const [name, b] of Object.entries(areas)) {
    const x0 = Math.round(ox + b[0] * u), y0 = Math.round(b[1] * u), x1 = Math.round(ox + b[2] * u), y1 = Math.round(b[3] * u);
    const L = [];
    for (let y = y0; y < y1; y += 2) for (let x = x0; x < x1; x += 2) {
      const i = (y * W + x) * 4;
      L.push(0.2126 * LUT[d[i]] + 0.7152 * LUT[d[i + 1]] + 0.0722 * LUT[d[i + 2]]);
    }
    L.sort((a, b) => a - b);
    const c = 1.05 / (L[Math.floor(L.length * 0.75)] + 0.05);
    worst[name] = Math.min(worst[name] ?? Infinity, c);
  }
}
console.log(`${path.basename(process.argv[2])} ${off ? '无保护层' : '有保护层'}：` +
  Object.entries(worst).map(([k, v]) => `${k} ${v.toFixed(2)}`).join('，'));
