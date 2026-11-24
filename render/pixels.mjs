// 渲染工作线程里逐像素的两件活。优先用 C 写的原生模块（native/pixels.c，直接读写画布内存、不拷贝），
// 没编译（例如本机 Windows）或设了 COMBOCAM_NO_NATIVE=1 时用下面的 JS 版本，两者结果逐字节相同。
//
// 1. lutCopy：把视频帧拷进画布，同时按行查表压暗（light 保护层的上下两段渐变）。
//    查找表是 Skia 自己在测试图上画一遍渐变读出来的（fx.js 的 lightScrimLUT），所以和每帧用 Skia 画渐变结果逐字节相同，
//    但每帧省掉两块全宽渐变（V100 机器上单线程 11 ms）。
// 2. rgbaToI420：RGBA → YUV420P，编码器直接吃 YUV。
//    为什么不交给 FFmpeg：FFmpeg 4.4 的格式转换（swscale）在编码进程里单线程跑，1080×1920 每帧约 10 ms，
//    渲染时 CPU 被占满，它就成了整条流水线的上限（实测 36 帧/秒）。放到各个工作线程里并行做，管道里每帧也从 8.3 MB 降到 3.1 MB。
//    公式：BT.601 有限范围（16–235），和 swscale 默认一致；色度取 2×2 平均。和 swscale 比亮度最多差 1，色度滤波略有不同，
//    成片和无损渲染帧比的 PSNR 与原来相同（36.0 dB vs 35.9 dB）。输入要求不透明，alpha 忽略。

import { createRequire } from 'node:module';

let native = null;
if (process.env.COMBOCAM_NO_NATIVE !== '1') {
  try {
    native = createRequire(import.meta.url)('./native/pixels.node');
  } catch {
    native = null;
  }
}
export const nativePixels = !!native;

export const i420Bytes = (W, H) => W * H + 2 * (W >> 1) * (H >> 1);

// 纯黑的一帧（Y=16，U=V=128）：原视频少帧时补位用
export function blackI420(W, H) {
  const out = new Uint8Array(i420Bytes(W, H));
  out.fill(16, 0, W * H);
  out.fill(128, W * H);
  return out;
}

/** src、dst：W×H 的 RGBA；lut：H×256 字节，第 y 行的输出 = lut[y*256 + 输入]；active[y] 为 0 的行原样拷贝。 */
export function lutCopy(src, dst, lut, active, W, H) {
  if (native) return native.lutCopy(src, dst, lut, active, W, H);
  const row = W * 4;
  for (let y = 0; y < H; y++) {
    const o = y * row;
    if (!active[y]) { dst.set(src.subarray(o, o + row), o); continue; }
    const L = y * 256;
    for (let i = o, e = o + row; i < e; i += 4) {
      dst[i] = lut[L + src[i]];
      dst[i + 1] = lut[L + src[i + 1]];
      dst[i + 2] = lut[L + src[i + 2]];
      dst[i + 3] = 255;
    }
  }
}

export function rgbaToI420(src, W, H, dst) {
  if (native) { native.rgbaToI420(src, W, H, dst); return dst; }
  const cw = W >> 1, ch = H >> 1, ySize = W * H, cSize = cw * ch;
  const stride = W * 4;
  for (let cy = 0; cy < ch; cy++) {
    let p0 = 2 * cy * stride, p1 = p0 + stride;
    let y0 = 2 * cy * W, y1 = y0 + W;
    let u = ySize + cy * cw, v = u + cSize;
    for (let cx = 0; cx < cw; cx++) {
      const r00 = src[p0], g00 = src[p0 + 1], b00 = src[p0 + 2];
      const r01 = src[p0 + 4], g01 = src[p0 + 5], b01 = src[p0 + 6];
      const r10 = src[p1], g10 = src[p1 + 1], b10 = src[p1 + 2];
      const r11 = src[p1 + 4], g11 = src[p1 + 5], b11 = src[p1 + 6];
      // Y = 16 + (65.738 R + 129.057 G + 25.064 B) / 256，定点 ×2^8 后四舍五入
      dst[y0++] = (66 * r00 + 129 * g00 + 25 * b00 + 4224) >> 8;
      dst[y0++] = (66 * r01 + 129 * g01 + 25 * b01 + 4224) >> 8;
      dst[y1++] = (66 * r10 + 129 * g10 + 25 * b10 + 4224) >> 8;
      dst[y1++] = (66 * r11 + 129 * g11 + 25 * b11 + 4224) >> 8;
      const r = r00 + r01 + r10 + r11, g = g00 + g01 + g10 + g11, b = b00 + b01 + b10 + b11;
      // 4 个像素的和，所以再多右移 2 位；+128×2^10 让结果为正再移位，+2^9 四舍五入（结果恒为正，C 版本同样的整数运算）
      dst[u++] = (-38 * r - 74 * g + 112 * b + 131584) >> 10;
      dst[v++] = (112 * r - 94 * g - 18 * b + 131584) >> 10;
      p0 += 8;
      p1 += 8;
    }
  }
  return dst;
}
