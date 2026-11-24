// 剖析：真实视频帧上，drawFrame 各部分和像素搬运的单线程耗时（毫秒/帧）。在要部署的机器上跑才有参考意义。
//   node render/profile_parts.mjs IN.mp4 out/<名>.track.json [帧数，默认 300] [起始帧，默认 0]
// 跳绳的轨迹（sport = jumprope）剖析 jumprope_fx.js 的各部分。
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createCanvas, GlobalFonts, ImageData } from '@napi-rs/canvas';
import { registerFonts } from './fonts.mjs';
import { rawVideoSource } from './video_source.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
registerFonts(GlobalFonts);
const [input, trackPath, nStr] = process.argv.slice(2);
const N = parseInt(nStr || '300', 10);
const F0 = parseInt(process.argv[5] || '0', 10);
const track = JSON.parse(fs.readFileSync(trackPath, 'utf-8'));
const fx = await import(track.sport !== 'jumprope' ? './fx.js' : track.mode === 'pk' ? './jumprope_pk_fx.js' : './jumprope_fx.js');
const styles = JSON.parse(fs.readFileSync(path.join(HERE, 'styles.json'), 'utf-8'));
const tl = fx.buildTimeline(track, styles, (w, h) => createCanvas(w, h), {});
const { W, H } = tl;
const canvas = createCanvas(W, H), ctx = canvas.getContext('2d');
const video = rawVideoSource(createCanvas, ImageData, W, H, Math.ceil(tl.rBall * 2.2) + 4);
const T = {};
const tick = (k, fn) => { const a = performance.now(); const r = fn(); T[k] = (T[k] || 0) + performance.now() - a; return r; };
const reset = () => { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over'; };

const fb = W * H * 4;
const vf = `${track.source.filter}${track.source.canvas_yuv ? ',format=yuv420p' : ''},trim=start_frame=${F0},setpts=PTS-STARTPTS`;
const dec = spawn('ffmpeg', ['-v', 'error', '-i', input, '-filter_complex', vf, '-frames:v', String(N), '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1']);
const chunks = [];
dec.stdout.on('data', (c) => chunks.push(c));
await new Promise((r) => dec.on('close', r));
const all = Buffer.concat(chunks);
const frames = [];
for (let o = 0; o + fb <= all.length; o += fb) frames.push(new Uint8ClampedArray(all.buffer, all.byteOffset + o, fb));
console.log(`${frames.length} 帧`);
frames.forEach((raw, k) => {
  const t = (F0 + k) / tl.fps;
  tick('new ImageData', () => video.set(raw));
  reset();
  tick('putImageData（贴视频帧）', () => video.full(ctx, false));
  tick('canvas.data().set（直接写像素）', () => canvas.data().set(raw));
  for (const part of fx.PARTS) { reset(); tick(`part:${part}`, () => fx.drawFrame(ctx, tl, t, 'replay', video, { only: part })); }
  reset();
  tick('drawFrame 全部（含视频）', () => fx.drawFrame(ctx, tl, t, 'replay', video));
  tick('stateAt', () => fx.stateAt(tl, t));
  tick('拷出：new Uint8Array + set', () => { const d = canvas.data(); const c = new Uint8Array(d.length); c.set(d); });
  const keep = new Uint8Array(fb);
  tick('拷出：复用缓冲区 set', () => keep.set(canvas.data()));
});
for (const k of Object.keys(T)) T[k] = +(T[k] / frames.length).toFixed(2);
console.log(JSON.stringify(T, null, 1));
