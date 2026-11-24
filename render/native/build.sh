#!/usr/bin/env bash
# 编译渲染用的原生模块 render/native/pixels.node（Linux，需要 gcc；头文件用当前 node 自带的）。
# 编译失败或没编译也能渲染，只是逐像素的两步走 JS 版本，慢一些（见 render/pixels.mjs）。
set -euo pipefail
cd "$(dirname "$0")"
INC="$(dirname "$(dirname "$(command -v node)")")/include/node"
[ -f "$INC/node_api.h" ] || { echo "找不到 node_api.h（$INC）"; exit 1; }
# x86-64-v3 = AVX2，2015 年以后的服务器 CPU 都支持；-O3 让 gcc 把亮度循环自动向量化
gcc -O3 -march=x86-64-v3 -shared -fPIC -I"$INC" -o pixels.node pixels.c
echo "已编译 $(pwd)/pixels.node"
