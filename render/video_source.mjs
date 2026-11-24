// 视频帧提供者（fx.js drawFrame 的 video 参数）：成片渲染和单帧预览共用。
// - full：不变换时直接写像素；震屏时画整帧
// - patch：抠球只拷贝球附近的一小块
// - small：1/DS 分辨率的整帧（按步长取样，不经过整帧绘制），给渐进模糊用

export const SMALL_DS = 4;

// direct：可选，{ canvas, ctx, scrim, lutCopy }，ctx 所在的画布。给了就不建 ImageData（napi-rs 建一个 1080×1920 的要 10 ms），
// 不变换时把像素直接写进画布内存（canvas.data() 是实时视图），和 putImageData 结果一样（视频帧不透明）。
// 再给了 scrim（fx.js 的 lightScrimLUT）和 lutCopy（pixels.mjs），写像素时顺手查表压暗，并把 scrimApplied 设为 true，
// drawFrame 就不再画 light 保护层的渐变（结果逐字节相同）。震屏帧照旧画。
export function rawVideoSource(createCanvas, ImageData, W, H, maxPatch, direct = null) {
  let raw = null, img = null, fcDirty = true, smallDirty = true;
  const image = () => img || (img = new ImageData(raw, W, H));
  const frameCanvas = createCanvas(W, H);
  const fctx = frameCanvas.getContext('2d');
  const patchCanvas = createCanvas(maxPatch, maxPatch);
  const pctx = patchCanvas.getContext('2d');
  const sw = Math.ceil(W / SMALL_DS), sh = Math.ceil(H / SMALL_DS);
  const smallCanvas = createCanvas(sw, sh);
  const sctx = smallCanvas.getContext('2d');
  const smallBuf = new Uint8ClampedArray(sw * sh * 4);
  const src = {
    scrimApplied: false,
    set(buf) { raw = buf; img = null; fcDirty = true; smallDirty = true; },
    // 帧的内存要另作他用（渲染器拿它装输出像素）之前调用
    release() { raw = null; img = null; },
    // scrim：调用方是 drawFrame、后面要画保护层时为 true（片尾贴最后一帧时不能压暗）
    full(ctx, transformed, scrim = false) {
      src.scrimApplied = false;
      if (!transformed) {
        if (direct && ctx === direct.ctx) {
          const out = direct.canvas.data();
          if (scrim && direct.scrim && direct.lutCopy) {
            direct.lutCopy(raw, out, direct.scrim.lut, direct.scrim.active, W, H);
            src.scrimApplied = true;
          } else out.set(raw);
        } else ctx.putImageData(image(), 0, 0);
        return;
      }
      if (fcDirty) { fctx.putImageData(image(), 0, 0); fcDirty = false; }
      ctx.drawImage(frameCanvas, 0, 0);
    },
    patch(ctx, x, y, r) {
      const x0 = Math.max(0, Math.floor(x - r)), y0 = Math.max(0, Math.floor(y - r));
      const x1 = Math.min(W, Math.ceil(x + r), x0 + maxPatch), y1 = Math.min(H, Math.ceil(y + r), y0 + maxPatch);
      const w = x1 - x0, h = y1 - y0;
      if (w <= 0 || h <= 0) return;
      const sub = new Uint8ClampedArray(w * h * 4);
      for (let j = 0; j < h; j++) sub.set(raw.subarray(((y0 + j) * W + x0) * 4, ((y0 + j) * W + x1) * 4), j * w * 4);
      // 先整块清空：震屏时贴回要放大，插值会读到这一小块外面一圈，那里不能留着之前帧的像素
      pctx.clearRect(0, 0, maxPatch, maxPatch);
      pctx.putImageData(new ImageData(sub, w, h), 0, 0);
      ctx.drawImage(patchCanvas, 0, 0, w, h, x0, y0, w, h);
    },
    small() {
      if (smallDirty) {
        // 每个 4×4 块取 4 个点求平均（只隔点取样会有锯齿，模糊后显出马赛克）
        let o = 0;
        for (let y = 0; y < sh; y++) {
          const r1 = Math.min(H - 1, y * SMALL_DS + 1) * W, r2 = Math.min(H - 1, y * SMALL_DS + 3) * W;
          for (let x = 0; x < sw; x++) {
            const c1 = Math.min(W - 1, x * SMALL_DS + 1), c2 = Math.min(W - 1, x * SMALL_DS + 3);
            const a = (r1 + c1) * 4, b = (r1 + c2) * 4, c = (r2 + c1) * 4, d = (r2 + c2) * 4;
            smallBuf[o++] = (raw[a] + raw[b] + raw[c] + raw[d]) >> 2;
            smallBuf[o++] = (raw[a + 1] + raw[b + 1] + raw[c + 1] + raw[d + 1]) >> 2;
            smallBuf[o++] = (raw[a + 2] + raw[b + 2] + raw[c + 2] + raw[d + 2]) >> 2;
            smallBuf[o++] = 255;
          }
        }
        sctx.putImageData(new ImageData(smallBuf, sw, sh), 0, 0);
        smallDirty = false;
      }
      return { canvas: smallCanvas, ds: SMALL_DS };
    },
  };
  return src;
}
