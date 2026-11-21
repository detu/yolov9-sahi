// Per-element tile pre/post-processing, shared by the CUDA kernels
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

// Network input value at channel c (0 = R, 1 = G, 2 = B), row y, column x of
// the tile whose top-left corner is (tile_x, tile_y): RGB, scaled to 0..1.
// `bgr` is an 8-bit BGR image with rows `step` bytes apart. Tiles are exactly
// the input size, so there is no resizing or letterbox padding.
TILE_HD float tile_input_value(const uint8_t* bgr, size_t step, int tile_x, int tile_y, int c, int y, int x) {
    return bgr[static_cast<size_t>(tile_y + y) * step + static_cast<size_t>(tile_x + x) * 3 + (2 - c)] *
           (1.0f / 255.0f);
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
