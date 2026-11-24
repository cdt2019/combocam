// 单线程剖析每帧各步骤耗时：node render/bench_fx.mjs out/<名>.track.json [style]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createCanvas, GlobalFonts, ImageData } from '@napi-rs/canvas';
import * as fx from './fx.js';
import { registerFonts } from './fonts.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
registerFonts(GlobalFonts);
const track = JSON.parse(fs.readFileSync(process.argv[2], 'utf-8'));
const styles = JSON.parse(fs.readFileSync(path.join(HERE, 'styles.json'), 'utf-8'));
const style = process.argv[3] || 'neon';
const tl = fx.buildTimeline(track, styles, (w, h) => createCanvas(w, h));
const { W, H } = tl;
const canvas = createCanvas(W, H), ctx = canvas.getContext('2d');
const times = [];
for (let k = 0; k < 60; k++) times.push(tl.touches[0].t - 0.5 + k / 30);
const T = {};
const tick = (k, fn) => { const a = performance.now(); const r = fn(); T[k] = (T[k] || 0) + performance.now() - a; return r; };
for (const t of times) {
  for (const part of fx.PARTS) tick(`part:${part}`, () => fx.drawFrame(ctx, tl, t, style, null, { only: part }));
  tick('drawFrame(all, no video)', () => fx.drawFrame(ctx, tl, t, style, null));
  tick('canvas.data()+copy', () => { const d = canvas.data(); const c = new Uint8Array(d.length); c.set(d); });
}
for (const k of Object.keys(T)) T[k] = +(T[k] / times.length).toFixed(1);
console.log(JSON.stringify(T, null, 1));
