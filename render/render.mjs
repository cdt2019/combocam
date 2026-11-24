// 成片渲染：原视频 + 轨迹数据 → 带特效的竖版成片（含片尾成绩卡）。
//
//   node render/render.mjs --input in.mp4 --track track.json --output out.mp4 [--style replay]
//        [--layout v3|v3s|v3l|v3c|v2|v1] [--tint panel|all|none] [--panel-font noto|smiley|zcool|barlowc|inter] [--protect light|plates|edges|veil|mix] [--num-font bebas|inter|noto|...]
//        [--endcard card|none] [--endcard-text "一行字"] [--nickname 昵称] [--workers 4] [--slowmo] [--encoder libx264|h264_nvenc] [--preset veryfast] [--force]
//
// 流水线：FFmpeg 解码（主线程顺序读）→ N 个工作线程并行画特效（fx.js，纯函数，可乱序）→ 按序送进一个编码器。
// 最后一行 stdout 输出 JSON 计时。

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { blackI420, i420Bytes, lutCopy, nativePixels, rgbaToI420 } from './pixels.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));

if (isMainThread) {
  await main();
} else {
  await workerMain();
}

function parseArgs(argv) {
  const a = { style: 'replay', layout: '', protect: '', 'num-font': '', tint: '', 'panel-font': '', endcard: 'card', 'endcard-text': '', nickname: '', workers: Math.max(1, Math.floor(os.cpus().length / 2)), slowmo: false, encoder: 'libx264', endcardSec: 3.0 };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i].replace(/^--/, '');
    if (k === 'slowmo') { a.slowmo = true; continue; }
    if (k === 'force') { a.force = true; continue; }
    const v = argv[++i];
    if (k === 'workers') a.workers = Math.max(1, parseInt(v, 10));
    else if (k === 'endcard-sec') a.endcardSec = parseFloat(v);
    else a[k] = v;
  }
  for (const k of ['input', 'track', 'output']) if (!a[k]) throw new Error(`缺少 --${k}`);
  return a;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const t0 = performance.now();
  const track = JSON.parse(fs.readFileSync(args.track, 'utf-8'));
  // 分析判定不出片（没找到球、没在颠球……）：不渲染，退出码 3，stdout 给出重拍提示。--force 照样渲染（调试用）
  if (track.status && !track.status.ok && !args.force) {
    console.error(`[render] 不出片（${track.status.code}）：${track.status.title}。${track.status.message}`);
    console.log(JSON.stringify({ status: track.status }));
    process.exit(3);
  }
  const styles = JSON.parse(fs.readFileSync(path.join(HERE, 'styles.json'), 'utf-8'));
  const { width: W, height: H, fps, frames: n } = track;
  const frameBytes = W * H * 4;

  // ---- 输出时间线：每个输出帧用哪一帧原视频、在哪个时刻画特效 ----
  // 慢放：高光段每帧原视频出两帧，特效时间各差半帧，所以特效是真 0.5 倍速、视频帧重复。
  const hl = args.slowmo && track.highlight ? track.highlight : null;
  const uses = Array.from({ length: n }, () => []); // 原视频帧 → [输出帧序号, 特效时刻]
  let nOut = 0;
  for (let f = 0; f < n; f++) {
    const t = f / fps;
    uses[f].push([nOut++, t]);
    if (hl && t >= hl.from && t <= hl.to) uses[f].push([nOut++, t + 0.5 / fps]);
  }
  const nVideo = nOut;
  const nEnd = args.endcard === 'none' ? 0 : Math.round(args.endcardSec * fps);
  const total = nVideo + nEnd;

  // ---- 编码器 ----
  const hasAudio = !hl && track.source && track.source.probe && track.source.probe.has_audio;
  // null：只渲染不编码，用来拆分计时；md5：每帧像素的 MD5 写进 --output（文本），用来核对优化前后画面一模一样
  // 送给编码器的像素格式：默认 yuv420p，工作线程里自己转（见 pixels.mjs，比交给 FFmpeg 单线程转快）；md5 核对画面时用 rgba
  const pix = args.pix || (['md5', 'raw'].includes(args.encoder) ? 'rgba' : 'yuv420p');
  if (!['rgba', 'yuv420p'].includes(pix)) throw new Error(`不支持的像素格式 ${pix}`);
  const outBytes = pix === 'rgba' ? frameBytes : i420Bytes(W, H);
  const enc = {
    // 按 VMAF 调的（GTX 1050 Ti，30 秒颠球样片，对未压缩的成片帧）：libx264 veryfast crf20 31.4 MB / VMAF 95.7；
    // 这组 28.8 MB / 96.5，最差 1% 的帧 93.6（x264 93.1），编码快 1.7 倍。原来的 p4 cq21 是 55.8 MB / 99.1，画质高出用不着的一截
    h264_nvenc: ['-c:v', 'h264_nvenc', '-preset', args.preset || 'p5', '-cq', '27', '-b:v', '0', '-rc-lookahead', '20', '-bf', '3',
      '-pix_fmt', 'yuv420p', '-profile:v', 'high'],
    libx264: ['-c:v', 'libx264', '-preset', args.preset || 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-profile:v', 'high'],
    null: [],
    md5: [],
    raw: [],
  }[args.encoder];
  if (!enc) throw new Error(`不支持的编码器 ${args.encoder}`);
  const rawIn = ['-v', 'error', '-f', 'rawvideo', '-pix_fmt', pix, '-s', `${W}x${H}`, '-r', String(fps), '-i', 'pipe:0'];
  const encArgs = args.encoder === 'null' ? [...rawIn, '-f', 'null', '-']
    : args.encoder === 'md5' ? ['-y', ...rawIn, '-f', 'framemd5', args.output]
    : args.encoder === 'raw' ? ['-y', ...rawIn, '-f', 'rawvideo', args.output] // 原始帧（测画质用，很大）
    : ['-y', ...rawIn,
      ...(hasAudio ? ['-i', args.input, '-map', '0:v', '-map', '1:a:0?', '-c:a', 'aac', '-b:a', '128k'] : ['-an']),
      ...enc, '-movflags', '+faststart', args.output];
  // COMBOCAM_FFMPEG_ENC：只换编码用的 ffmpeg（解码照旧用 PATH 里的，画面逐位不变）。老显卡的驱动装不到新版 ffmpeg 的 h264_nvenc
  // 要求的版本时（比如 Pascal 停在 581，ffmpeg 9 要 610 以上），指向一个旧版 ffmpeg 专门用来编码
  const encoder = spawn(process.env.COMBOCAM_FFMPEG_ENC || 'ffmpeg', encArgs, { stdio: ['pipe', 'ignore', 'inherit'] });
  const fail = (msg) => { console.error(`[render] ${msg}`); process.exit(2); };
  encoder.stdin.on('error', () => fail(`编码器 ${args.encoder} 提前退出（h264_nvenc 需要较新的 NVIDIA 驱动，可改用 --encoder libx264）`));
  const encDone = new Promise((res) => encoder.on('close', (c) => (c === 0 ? res() : fail(`编码器退出码 ${c}`))));

  // ---- 工作线程 ----
  const workers = Array.from({ length: args.workers }, () => new Worker(fileURLToPath(import.meta.url), {
    workerData: { track, styles, style: args.style, endcardText: args['endcard-text'], nickname: args.nickname, pix,
      look: { layout: args.layout, protect: args.protect, numFont: args['num-font'], tint: args.tint, panelFont: args['panel-font'] } },
  }));
  let renderMs = 0;
  const profMs = [0, 0, 0, 0];
  const legMin = { hud: Infinity, rhythm: Infinity, lumMax: 0, plateMax: 0 };
  const pending = new Map();
  let nextWrite = 0;
  let inFlight = 0;
  let wakeProducer = null;
  const wake = () => { if (wakeProducer) { const f = wakeProducer; wakeProducer = null; f(); } };
  let writing = Promise.resolve();
  // 写完的帧内存回收给解码用（见 workerMain 里"输出像素装进哪块内存"）
  const pool = [];
  const recycle = (buf) => { if (buf.byteLength === frameBytes && pool.length < 64) pool.push(buf); };
  const flush = () => {
    writing = writing.then(async () => {
      while (pending.has(nextWrite)) {
        const buf = pending.get(nextWrite);
        pending.delete(nextWrite);
        nextWrite++;
        if (!encoder.stdin.write(Buffer.from(buf, 0, outBytes), () => recycle(buf))) await new Promise((r) => encoder.stdin.once('drain', r));
        wake();
      }
    });
  };
  for (const w of workers) {
    w.on('message', (m) => {
      renderMs += m.ms;
      if (m.prof) m.prof.forEach((v, i) => { profMs[i] += v; });
      if (m.leg) {
        legMin.hud = Math.min(legMin.hud, m.leg[0]);
        if (m.leg[1] != null) legMin.rhythm = Math.min(legMin.rhythm, m.leg[1]);
        legMin.lumMax = Math.max(legMin.lumMax, m.leg[2]);
        legMin.plateMax = Math.max(legMin.plateMax, m.leg[3]);
      }
      pending.set(m.id, m.buf);
      if (m.last) inFlight--;
      wake();
      flush();
    });
    w.on('error', (e) => { console.error(e); process.exit(1); });
  }
  const maxInFlight = args.workers * 2;
  let rr = 0;
  const dispatch = async (task, transfer) => {
    while (inFlight >= maxInFlight || pending.size > maxInFlight * 2) await new Promise((r) => (wakeProducer = r));
    inFlight++;
    workers[rr++ % workers.length].postMessage(task, transfer);
  };

  // ---- 解码，按帧分发 ----
  const tDec0 = performance.now();
  let decodeWaitMs = 0;
  // --frames：分析时顺手存下的解码帧（已按画布帧率取帧，见 combocam/video.py 的 FrameReader），有就不再解码原视频，
  // 只在它上面跑画布滤镜剩下的部分（缩放、裁剪……），输入和算法都一样，结果逐帧相同
  // source.canvas_yuv（跳绳）：画布滤镜后先转 yuv420p，和分析时一样；它的缓存已经是画布尺寸的 yuv420p，只需转 rgba
  const fpsPart = /^fps=[^,;]+,/;
  const canvasYuv = !!track.source.canvas_yuv;
  const cached = args.frames && (canvasYuv || fpsPart.test(track.source.filter)) && fs.existsSync(args.frames);
  const decArgs = cached
    ? (canvasYuv ? ['-v', 'error', '-i', args.frames] : ['-v', 'error', '-i', args.frames, '-filter_complex', track.source.filter.replace(fpsPart, '')])
    : ['-v', 'error', '-i', args.input, '-filter_complex', canvasYuv ? `${track.source.filter},format=yuv420p` : track.source.filter];
  const dec = spawn('ffmpeg', [...decArgs, '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'],
    { stdio: ['ignore', 'pipe', 'inherit'] });
  let lastFrame = null;
  let f = 0;
  const reader = frameReader(dec.stdout, frameBytes);
  for (;;) {
    const a = performance.now();
    const buf = await reader.next(pool.pop());
    decodeWaitMs += performance.now() - a;
    if (!buf || f >= n) break;
    const jobs = uses[f].map(([id, t]) => ({ id, kind: 'video', t }));
    if (f === n - 1 && nEnd) lastFrame = Buffer.from(buf); // 片尾淡入和模糊背景要用
    await dispatch({ jobs, buf: buf.buffer }, [buf.buffer]);
    f++;
  }
  dec.kill();
  // 原视频实际帧数可能比分析时少一两帧：补齐编号，避免编码器卡住
  for (; f < n; f++) for (const [id] of uses[f]) pending.set(id, pix === 'rgba' ? new ArrayBuffer(frameBytes) : blackI420(W, H).buffer);
  flush();
  for (let k = 0; k < nEnd; k++) {
    const tc = k / fps;
    const buf = lastFrame ? Buffer.from(lastFrame).buffer : null;
    await dispatch({ jobs: [{ id: nVideo + k, kind: 'end', t: tc }], buf }, buf ? [buf] : []);
  }
  while (nextWrite < total) await new Promise((r) => setTimeout(r, 20));
  await writing;
  encoder.stdin.end();
  await encDone;
  await Promise.all(workers.map((w) => w.terminate()));
  const wall = (performance.now() - t0) / 1000;
  const stats = {
    output: args.output, sport: track.sport || 'juggling', style: args.style, look: { layout: args.layout || 'default', protect: args.protect || 'default', numFont: args['num-font'] || 'default', tint: args.tint || 'default', panelFont: args['panel-font'] || 'default' }, frames: total, video_frames: nVideo, endcard_frames: nEnd,
    workers: args.workers, encoder: args.encoder, pix, native_pixels: nativePixels, frame_cache: !!cached, slowmo: !!hl, audio: !!hasAudio,
    render_cpu_s: +(renderMs / 1000).toFixed(2), decode_wait_s: +(decodeWaitMs / 1000).toFixed(2),
    // 工作线程里各步合计（秒）：拿到视频帧、画特效、拷出像素、可读性统计
    worker_s: Object.fromEntries(['set', 'draw', 'copy', 'legibility'].map((k, i) => [k, +(profMs[i] / 1000).toFixed(2)])),
    pipeline_s: +((performance.now() - tDec0) / 1000).toFixed(2), wall_s: +wall.toFixed(2), out_fps: +(total / wall).toFixed(1),
    legibility: {
      hud_min_contrast: +legMin.hud.toFixed(2),
      rhythm_min_contrast: Number.isFinite(legMin.rhythm) ? +legMin.rhythm.toFixed(2) : null,
      hud_bg_lum_max: +legMin.lumMax.toFixed(2), shade_max: +legMin.plateMax.toFixed(2),
      scene_data: !!track.scene,
    },
  };
  console.error(`[render] ${path.basename(args.output)}: ${total} 帧（视频 ${nVideo} + 片尾 ${nEnd}），${args.workers} 线程，用时 ${wall.toFixed(1)}s（${stats.out_fps} 帧/秒）；`
    + `计数白字最低对比度（按画面亮度估算）${stats.legibility.hud_min_contrast}，其余 ${stats.legibility.rhythm_min_contrast}${track.scene ? '' : '，没有画面亮度数据，按偏亮处理'}`);
  console.log(JSON.stringify(stats));
}

// 从管道里按整帧读
function frameReader(stream, frameBytes) {
  let chunks = [], have = 0, ended = false, waiter = null;
  stream.on('data', (c) => { chunks.push(c); have += c.length; if (have >= frameBytes * 3) stream.pause(); if (waiter) { const w = waiter; waiter = null; w(); } });
  stream.on('end', () => { ended = true; if (waiter) { const w = waiter; waiter = null; w(); } });
  return {
    // into：可选，拿来装这一帧的 ArrayBuffer（回收的旧帧），没有就新分配
    async next(into) {
      while (have < frameBytes && !ended) { stream.resume(); await new Promise((r) => (waiter = r)); }
      if (have < frameBytes) return null;
      const out = new Uint8Array(into || new ArrayBuffer(frameBytes));
      let off = 0;
      while (off < frameBytes) {
        const c = chunks[0];
        const take = Math.min(c.length, frameBytes - off);
        out.set(c.subarray(0, take), off);
        off += take;
        if (take === c.length) chunks.shift(); else chunks[0] = c.subarray(take);
      }
      have -= frameBytes;
      if (have < frameBytes * 2) stream.resume();
      return out;
    },
  };
}

async function workerMain() {
  const { createCanvas, GlobalFonts, ImageData } = await import('@napi-rs/canvas');
  const { rawVideoSource } = await import('./video_source.mjs');
  const { registerFonts } = await import('./fonts.mjs');
  registerFonts(GlobalFonts);
  const { track, styles, style, endcardText, nickname, look, pix } = workerData;
  // 跳绳的轨迹（combocam/jumprope_track.py）用跳绳特效（双人 PK 另一套），其余是颠球
  const { buildTimeline, drawFrame, drawEndCard, legibility, lightScrimLUT, stateAt } = await import(fxModule(track));
  const tl = buildTimeline(track, styles, (w, h) => createCanvas(w, h), look);
  const { W, H } = tl;
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');
  // 保护层渐变改成拷视频帧时查表压暗（见 fx.js 的 lightScrimLUT），结果逐字节相同
  const video = rawVideoSource(createCanvas, ImageData, W, H, Math.ceil(tl.rBall * 2.2) + 4, { canvas, ctx, scrim: lightScrimLUT(tl), lutCopy });
  parentPort.on('message', (task) => {
    const p0 = performance.now();
    if (task.buf) video.set(new Uint8ClampedArray(task.buf));
    const pSet = performance.now() - p0;
    task.jobs.forEach((job, k) => {
      const a = performance.now();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = 'source-over';
      if (job.kind === 'video') drawFrame(ctx, tl, job.t, style, video);
      else {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, W, H);
        drawEndCard(ctx, tl, job.t, { text: endcardText, nickname, styleId: style, lastFrame: task.buf ? video : null });
      }
      const b = performance.now();
      const data = canvas.data(); // RGBA，不透明帧与 getImageData 相同，但不额外转换
      // 输出像素装进哪块内存：这批的最后一帧直接用传进来的视频帧内存（已经画完，用不着了），
      // 主线程写给编码器后再拿去装下一帧解码结果，省掉每帧新分配 8 MB（Linux 上分配 + 缺页约 1–2 ms）
      const last = k === task.jobs.length - 1;
      const size = pix === 'rgba' ? data.length : i420Bytes(W, H);
      let copy;
      if (last && task.buf) { video.release(); copy = new Uint8Array(task.buf, 0, size); } else copy = new Uint8Array(size);
      if (pix === 'rgba') copy.set(data);
      else rgbaToI420(data, W, H, copy);
      const c = performance.now();
      // 可读性检查：HUD 文字在压暗后的背景上的对比度（节奏条出现后才算它）
      let leg = null;
      if (job.kind === 'video') {
        // 计数（标签 + 数字）单算；其余各行（面板、最高一颠、节奏）在第 1 次触球后才出现
        const L = legibility(tl, job.t);
        const items = Object.entries(L.items);
        const hud = Math.min(...items.filter(([n]) => n === '标签' || n === '数字').map(([, c]) => c));
        const rest = items.filter(([n]) => n !== '标签' && n !== '数字').map(([, c]) => c);
        const shade = Math.max(L.aTop, L.aBot, ...Object.values(L.plates));
        leg = [hud, stateAt(tl, job.t).count >= 1 && rest.length ? Math.min(...rest) : null, L.lumMax, shade];
      }
      const d = performance.now();
      // prof：[拿到视频帧, 画特效, 拷出像素, 可读性统计]（毫秒），用来找渲染瓶颈
      const prof = [k === 0 ? pSet : 0, b - a, c - b, d - c];
      parentPort.postMessage({ id: job.id, buf: copy.buffer, ms: d - a + (k === 0 ? pSet : 0), last, leg, prof }, [copy.buffer]);
    });
  });
}

function fxModule(track) {
  if (track.sport !== 'jumprope') return './fx.js';
  return track.mode === 'pk' ? './jumprope_pk_fx.js' : './jumprope_fx.js';
}
