// 渲染工作线程里逐像素的两件活（JS 版本和说明见 ../pixels.mjs，两边结果逐字节相同）。
// Node-API 模块，直接读写传进来的 Buffer / TypedArray / ArrayBuffer，不拷贝。编译：bash render/native/build.sh
#include <node_api.h>
#include <stdint.h>
#include <string.h>

// 取 Uint8Array / Uint8ClampedArray / Buffer / ArrayBuffer 的数据指针和字节数
static int bytes(napi_env env, napi_value v, uint8_t **p, size_t *len) {
  bool is = false;
  napi_is_typedarray(env, v, &is);
  if (is) {
    napi_typedarray_type t;
    size_t n, off;
    void *data;
    napi_value ab;
    if (napi_get_typedarray_info(env, v, &t, &n, &data, &ab, &off) != napi_ok) return 0;
    if (t != napi_uint8_array && t != napi_uint8_clamped_array) return 0;
    *p = data;
    *len = n;
    return 1;
  }
  napi_is_arraybuffer(env, v, &is);
  if (is) return napi_get_arraybuffer_info(env, v, (void **)p, len) == napi_ok;
  return 0;
}

static int32_t i32(napi_env env, napi_value v) {
  int32_t x = 0;
  napi_get_value_int32(env, v, &x);
  return x;
}

static napi_value fail(napi_env env, const char *msg) {
  napi_throw_error(env, NULL, msg);
  return NULL;
}

// lutCopy(src, dst, lut, active, W, H)
static napi_value lut_copy(napi_env env, napi_callback_info info) {
  size_t argc = 6;
  napi_value a[6];
  napi_get_cb_info(env, info, &argc, a, NULL, NULL);
  uint8_t *src, *dst, *lut, *act;
  size_t ns, nd, nl, na;
  if (argc < 6 || !bytes(env, a[0], &src, &ns) || !bytes(env, a[1], &dst, &nd) || !bytes(env, a[2], &lut, &nl) ||
      !bytes(env, a[3], &act, &na))
    return fail(env, "lutCopy: 参数类型不对");
  int32_t W = i32(env, a[4]), H = i32(env, a[5]);
  size_t row = (size_t)W * 4;
  if (W <= 0 || H <= 0 || ns < row * H || nd < row * H || nl < (size_t)H * 256 || na < (size_t)H)
    return fail(env, "lutCopy: 尺寸不对");
  for (int32_t y = 0; y < H; y++) {
    const uint8_t *s = src + row * y;
    uint8_t *d = dst + row * y;
    if (!act[y]) {
      memcpy(d, s, row);
      continue;
    }
    const uint8_t *L = lut + (size_t)y * 256;
    for (int32_t x = 0; x < W; x++, s += 4, d += 4) {
      d[0] = L[s[0]];
      d[1] = L[s[1]];
      d[2] = L[s[2]];
      d[3] = 255;
    }
  }
  return NULL;
}

// rgbaToI420(src, W, H, dst)
static napi_value rgba_to_i420(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value a[4];
  napi_get_cb_info(env, info, &argc, a, NULL, NULL);
  uint8_t *src, *dst;
  size_t ns, nd;
  if (argc < 4 || !bytes(env, a[0], &src, &ns) || !bytes(env, a[3], &dst, &nd)) return fail(env, "rgbaToI420: 参数类型不对");
  int32_t W = i32(env, a[1]), H = i32(env, a[2]);
  int32_t cw = W >> 1, ch = H >> 1;
  size_t ysize = (size_t)W * H, csize = (size_t)cw * ch;
  if (W <= 0 || H <= 0 || ns < ysize * 4 || nd < ysize + 2 * csize) return fail(env, "rgbaToI420: 尺寸不对");
  uint8_t *U = dst + ysize, *V = U + csize;
  for (int32_t cy = 0; cy < ch; cy++) {
    const uint8_t *p0 = src + (size_t)2 * cy * W * 4, *p1 = p0 + (size_t)W * 4;
    uint8_t *y0 = dst + (size_t)2 * cy * W, *y1 = y0 + W;
    uint8_t *u = U + (size_t)cy * cw, *v = V + (size_t)cy * cw;
    // 亮度：一行一行算，循环简单，gcc 能自动向量化
    for (int32_t x = 0; x < 2 * cw; x++) {
      y0[x] = (uint8_t)((66 * p0[4 * x] + 129 * p0[4 * x + 1] + 25 * p0[4 * x + 2] + 4224) >> 8);
      y1[x] = (uint8_t)((66 * p1[4 * x] + 129 * p1[4 * x + 1] + 25 * p1[4 * x + 2] + 4224) >> 8);
    }
    for (int32_t cx = 0; cx < cw; cx++) {
      const uint8_t *a0 = p0 + 8 * cx, *a1 = p1 + 8 * cx;
      int32_t r = a0[0] + a0[4] + a1[0] + a1[4];
      int32_t g = a0[1] + a0[5] + a1[1] + a1[5];
      int32_t b = a0[2] + a0[6] + a1[2] + a1[6];
      u[cx] = (uint8_t)((-38 * r - 74 * g + 112 * b + 131584) >> 10);
      v[cx] = (uint8_t)((112 * r - 94 * g - 18 * b + 131584) >> 10);
    }
  }
  return NULL;
}

NAPI_MODULE_INIT() {
  napi_value fn;
  napi_create_function(env, "lutCopy", NAPI_AUTO_LENGTH, lut_copy, NULL, &fn);
  napi_set_named_property(env, exports, "lutCopy", fn);
  napi_create_function(env, "rgbaToI420", NAPI_AUTO_LENGTH, rgba_to_i420, NULL, &fn);
  napi_set_named_property(env, exports, "rgbaToI420", fn);
  return exports;
}
