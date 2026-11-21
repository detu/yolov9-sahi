// Per-element region pre/post-processing, shared by the CUDA kernels
// (gpu_ops_cuda.cu, compiled with nvcc) and the CPU fallback (gpu_ops_cpu.cpp),
// so both compute exactly the same thing.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define TILE_HD __host__ __device__ __forceinline__
#else
#define TILE_HD inline
#endif

// One detection candidate of one tile, in tile pixels.
struct Candidate {
    float score;
    float x1, y1, x2, y2;
    int cls;
    int anchor;  // anchor index, used to restore a deterministic order
};

// How one image region maps onto the size x size network input: resized by
// `scale` to nw x nh and placed at (pad_l, pad_t); the rest is gray (114).
struct RegionXform {
    int rx, ry, rw, rh;  // region in the image
    int nw, nh;          // resized size
    int pad_l, pad_t;    // offset of the resized region in the input
    float scale;         // nw / rw (= nh / rh)
};

// Network input value at channel c (0 = R, 1 = G, 2 = B), row y, column x for
// one region: letterboxed, RGB, scaled to 0..1. `bgr` is an 8-bit BGR image
// with rows `step` bytes apart. Resizing is bilinear with half-pixel centres
// (as cv::resize INTER_LINEAR); a region that is exactly the input size is
// copied 1:1.
TILE_HD float letterbox_input_value(const uint8_t* bgr, size_t step, const RegionXform& t, int c, int y, int x) {
    const int lx = x - t.pad_l, ly = y - t.pad_t;
    if (lx < 0 || ly < 0 || lx >= t.nw || ly >= t.nh) return 114.0f / 255.0f;
    const int ch = 2 - c;
    if (t.nw == t.rw && t.nh == t.rh) {
        return bgr[static_cast<size_t>(t.ry + ly) * step + static_cast<size_t>(t.rx + lx) * 3 + ch] * (1.0f / 255.0f);
    }
    float u = (lx + 0.5f) * (static_cast<float>(t.rw) / t.nw) - 0.5f;
    float v = (ly + 0.5f) * (static_cast<float>(t.rh) / t.nh) - 0.5f;
    u = u < 0.f ? 0.f : (u > t.rw - 1 ? static_cast<float>(t.rw - 1) : u);
    v = v < 0.f ? 0.f : (v > t.rh - 1 ? static_cast<float>(t.rh - 1) : v);
    const int x0 = static_cast<int>(u), y0 = static_cast<int>(v);
    const int x1 = x0 + 1 < t.rw ? x0 + 1 : x0, y1 = y0 + 1 < t.rh ? y0 + 1 : y0;
    const float fx = u - x0, fy = v - y0;
    auto px = [&](int yy, int xx) {
        return static_cast<float>(bgr[static_cast<size_t>(t.ry + yy) * step + static_cast<size_t>(t.rx + xx) * 3 + ch]);
    };
    const float top = px(y0, x0) + (px(y0, x1) - px(y0, x0)) * fx;
    const float bot = px(y1, x0) + (px(y1, x1) - px(y1, x0)) * fx;
    return (top + (bot - top) * fy) * (1.0f / 255.0f);
}

// Best class and box of anchor `a` in one tile's head output: F = 4 + classes
// features by A anchors. feature_rows: stored [F][A] (the usual layout), else
// [A][F]. xyxy: boxes are x1,y1,x2,y2, else cx,cy,w,h. Scores are already
// sigmoid-activated. Boxes are clamped to the size x size tile.
TILE_HD Candidate decode_anchor(const float* out, int F, int A, bool feature_rows, bool xyxy, int size, int a) {
    const size_t fs = feature_rows ? static_cast<size_t>(A) : 1;  // stride between features
    const float* p = feature_rows ? out + a : out + static_cast<size_t>(a) * F;

    Candidate c;
    c.cls = 0;
    c.score = p[4 * fs];
    for (int k = 1; k < F - 4; ++k) {
        const float s = p[(4 + k) * fs];
        if (s > c.score) {
            c.score = s;
            c.cls = k;
        }
    }
    float b0 = p[0], b1 = p[fs], b2 = p[2 * fs], b3 = p[3 * fs];
    if (!xyxy) {  // cx, cy, w, h -> corners
        const float hw = 0.5f * b2, hh = 0.5f * b3;
        const float cx = b0, cy = b1;
        b0 = cx - hw; b1 = cy - hh; b2 = cx + hw; b3 = cy + hh;
    }
    const float lim = static_cast<float>(size);
    c.x1 = b0 < 0.f ? 0.f : (b0 > lim ? lim : b0);
    c.y1 = b1 < 0.f ? 0.f : (b1 > lim ? lim : b1);
    c.x2 = b2 < 0.f ? 0.f : (b2 > lim ? lim : b2);
    c.y2 = b3 < 0.f ? 0.f : (b3 > lim ? lim : b3);
    c.anchor = a;
    return c;
}
