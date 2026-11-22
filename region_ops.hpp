// Region preprocessing: letterbox + normalization into a planar float tensor.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Per-channel normalization of the network input, channel by channel in the
// order the network expects: value = (pixel - mean[c]) / std[c].
struct InputNorm {
    bool bgr = true;  // channel 0 is blue (RTMDet, mmdet bgr_to_rgb=False), else red
    // Resize filter: Pillow BILINEAR (antialiased when downscaling; LibreYOLO RTMDet),
    // else OpenCV INTER_LINEAR (2x2 bilinear, half-pixel centres; YOLOv9).
    bool pillow = true;
    float mean[3] = {103.53f, 116.28f, 123.675f};
    float std[3] = {57.375f, 57.12f, 58.395f};

    static InputNorm rtmdet() { return InputNorm{}; }
    static InputNorm yolo() {  // RGB, scaled to 0..1
        InputNorm n;
        n.bgr = false;
        n.pillow = false;
        for (int c = 0; c < 3; ++c) { n.mean[c] = 0.f; n.std[c] = 255.f; }
        return n;
    }
};

// How one image region maps onto the size x size network input: resized to
// nw x nh and placed at (pad_l, pad_t); the rest is gray (114).
struct RegionXform {
    int rx, ry, rw, rh;  // region in the image
    int nw, nh;          // resized size
    int pad_l, pad_t;    // offset of the resized region in the input
    float scale;         // nw / rw (= nh / rh)
};

inline RegionXform make_xform(int x, int y, int w, int h, int size, bool topleft) {
    RegionXform t;
    t.rx = x; t.ry = y; t.rw = w; t.rh = h;
    t.scale = std::min(static_cast<float>(size) / w, static_cast<float>(size) / h);
    t.nw = std::min(size, std::max(1, static_cast<int>(w * t.scale)));
    t.nh = std::min(size, std::max(1, static_cast<int>(h * t.scale)));
    t.pad_l = topleft ? 0 : (size - t.nw) / 2;
    t.pad_t = topleft ? 0 : (size - t.nh) / 2;
    return t;
}

// Resampling weights of Pillow's BILINEAR filter (what LibreYOLO uses): a
// triangle filter whose support grows with the downscale factor, so shrinking
// averages over the covered pixels (antialiasing) instead of sampling 2x2.
inline float to_u8(float v) {
    return v <= 0.f ? 0.f : v >= 255.f ? 255.f : std::floor(v + 0.5f);
}

struct ResampleAxis {
    std::vector<int> first;     // first source index per output pixel
    std::vector<int> count;     // number of source pixels per output pixel
    std::vector<float> weight;  // count[i] weights per output pixel, at i * max_count
    int max_count = 0;

    ResampleAxis(int in_size, int out_size) {
        const double scale = static_cast<double>(in_size) / out_size;
        const double filterscale = std::max(1.0, scale);
        const double support = 1.0 * filterscale;  // bilinear support is 1
        max_count = static_cast<int>(std::ceil(support)) * 2 + 1;
        first.resize(out_size);
        count.resize(out_size);
        weight.assign(static_cast<size_t>(out_size) * max_count, 0.f);
        for (int o = 0; o < out_size; ++o) {
            const double center = (o + 0.5) * scale;
            int xmin = static_cast<int>(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = static_cast<int>(center + support + 0.5);
            if (xmax > in_size) xmax = in_size;
            const int n = std::min(xmax - xmin, max_count);
            double total = 0.0;
            float* w = &weight[static_cast<size_t>(o) * max_count];
            for (int k = 0; k < n; ++k) {
                const double t = std::fabs((k + xmin - center + 0.5) / filterscale);
                const double v = t < 1.0 ? 1.0 - t : 0.0;
                w[k] = static_cast<float>(v);
                total += v;
            }
            if (total > 0.0)
                for (int k = 0; k < n; ++k) w[k] = static_cast<float>(w[k] / total);
            first[o] = xmin;
            count[o] = n;
        }
    }
};

// Write one region's network input (3 x size x size floats, planar) from an
// 8-bit BGR image with rows `step` bytes apart: letterboxed (gray 114 pad),
// resized with Pillow's bilinear filter when needed (regions of the input
// size are copied 1:1), then normalized.
inline void fill_region_input(const uint8_t* bgr, size_t step, const RegionXform& t, int size,
                              const InputNorm& norm, float* dst) {
    const size_t plane = static_cast<size_t>(size) * size;
    // Source channel (in BGR memory order) for each network channel
    int src_ch[3];
    for (int c = 0; c < 3; ++c) src_ch[c] = norm.bgr ? c : 2 - c;
    float pad_val[3], inv_std[3];
    for (int c = 0; c < 3; ++c) {
        inv_std[c] = 1.0f / norm.std[c];
        pad_val[c] = (114.0f - norm.mean[c]) * inv_std[c];
    }
    for (int c = 0; c < 3; ++c) std::fill(dst + c * plane, dst + (c + 1) * plane, pad_val[c]);

    if (t.nw == t.rw && t.nh == t.rh) {
        for (int y = 0; y < t.nh; ++y) {
            const uint8_t* row = bgr + static_cast<size_t>(t.ry + y) * step + static_cast<size_t>(t.rx) * 3;
            for (int c = 0; c < 3; ++c) {
                float* out = dst + c * plane + static_cast<size_t>(t.pad_t + y) * size + t.pad_l;
                const int sc = src_ch[c];
                const float m = norm.mean[c], is = inv_std[c];
                for (int x = 0; x < t.nw; ++x) out[x] = (row[x * 3 + sc] - m) * is;
            }
        }
        return;
    }

    if (!norm.pillow) {
        // OpenCV INTER_LINEAR: each output pixel from the 2x2 source pixels around
        // its (half-pixel) centre, no antialiasing
        std::vector<int> x0s(t.nw), x1s(t.nw);
        std::vector<float> fxs(t.nw);
        const float sx = static_cast<float>(t.rw) / t.nw, sy = static_cast<float>(t.rh) / t.nh;
        for (int x = 0; x < t.nw; ++x) {
            float u = (x + 0.5f) * sx - 0.5f;
            u = u < 0.f ? 0.f : (u > t.rw - 1 ? static_cast<float>(t.rw - 1) : u);
            x0s[x] = static_cast<int>(u);
            x1s[x] = x0s[x] + 1 < t.rw ? x0s[x] + 1 : x0s[x];
            fxs[x] = u - x0s[x];
        }
        for (int y = 0; y < t.nh; ++y) {
            float v = (y + 0.5f) * sy - 0.5f;
            v = v < 0.f ? 0.f : (v > t.rh - 1 ? static_cast<float>(t.rh - 1) : v);
            const int y0 = static_cast<int>(v), y1 = y0 + 1 < t.rh ? y0 + 1 : y0;
            const float fy = v - y0;
            const uint8_t* r0 = bgr + static_cast<size_t>(t.ry + y0) * step + static_cast<size_t>(t.rx) * 3;
            const uint8_t* r1 = bgr + static_cast<size_t>(t.ry + y1) * step + static_cast<size_t>(t.rx) * 3;
            for (int c = 0; c < 3; ++c) {
                float* out = dst + c * plane + static_cast<size_t>(t.pad_t + y) * size + t.pad_l;
                const int sc = src_ch[c];
                const float m = norm.mean[c], is = inv_std[c];
                for (int x = 0; x < t.nw; ++x) {
                    const int a = x0s[x] * 3 + sc, b = x1s[x] * 3 + sc;
                    const float fx = fxs[x];
                    const float top = r0[a] + (static_cast<float>(r0[b]) - r0[a]) * fx;
                    const float bot = r1[a] + (static_cast<float>(r1[b]) - r1[a]) * fx;
                    out[x] = (top + (bot - top) * fy - m) * is;
                }
            }
        }
        return;
    }

    // Pillow: separable resample, horizontal pass into a float buffer, then vertical.
    const ResampleAxis ax(t.rw, t.nw), ay(t.rh, t.nh);
    std::vector<float> tmp(static_cast<size_t>(t.rh) * t.nw * 3);
    for (int y = 0; y < t.rh; ++y) {
        const uint8_t* row = bgr + static_cast<size_t>(t.ry + y) * step + static_cast<size_t>(t.rx) * 3;
        float* out = &tmp[static_cast<size_t>(y) * t.nw * 3];
        for (int x = 0; x < t.nw; ++x) {
            const float* w = &ax.weight[static_cast<size_t>(x) * ax.max_count];
            const uint8_t* p = row + static_cast<size_t>(ax.first[x]) * 3;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f;
            for (int k = 0; k < ax.count[x]; ++k) {
                s0 += w[k] * p[k * 3 + 0];
                s1 += w[k] * p[k * 3 + 1];
                s2 += w[k] * p[k * 3 + 2];
            }
            // Pillow stores each pass as 8-bit: round and clamp
            out[x * 3 + 0] = to_u8(s0); out[x * 3 + 1] = to_u8(s1); out[x * 3 + 2] = to_u8(s2);
        }
    }
    for (int y = 0; y < t.nh; ++y) {
        const float* w = &ay.weight[static_cast<size_t>(y) * ay.max_count];
        const int y0 = ay.first[y], n = ay.count[y];
        for (int x = 0; x < t.nw; ++x) {
            float s[3] = {0.f, 0.f, 0.f};
            for (int k = 0; k < n; ++k) {
                const float* p = &tmp[(static_cast<size_t>(y0 + k) * t.nw + x) * 3];
                s[0] += w[k] * p[0]; s[1] += w[k] * p[1]; s[2] += w[k] * p[2];
            }
            for (int c = 0; c < 3; ++c) {
                const size_t o = c * plane + static_cast<size_t>(t.pad_t + y) * size + t.pad_l + x;
                dst[o] = (to_u8(s[src_ch[c]]) - norm.mean[c]) * inv_std[c];
            }
        }
    }
}

// One detection candidate in letterboxed input pixels.
struct Candidate {
    float score;
    float x1, y1, x2, y2;
    int cls;
};

// Best class per anchor of one region's head output: F = 4 + classes features
// by A anchors, stored [F][A] (feature_rows) or [A][F] (RTMDet's decoded
// export). xyxy: boxes are x1,y1,x2,y2, else cx,cy,w,h. Scores are already
// sigmoid-activated. Keeps anchors scoring at least conf, in anchor order.
inline void decode_head(const float* out, int F, int A, bool feature_rows, bool xyxy, int size, float conf,
                        std::vector<Candidate>& cands) {
    const size_t fs = feature_rows ? static_cast<size_t>(A) : 1;
    const float lim = static_cast<float>(size);
    auto clampf = [lim](float v) { return v < 0.f ? 0.f : (v > lim ? lim : v); };
    for (int a = 0; a < A; ++a) {
        const float* p = feature_rows ? out + a : out + static_cast<size_t>(a) * F;
        int best = 0;
        float score = p[4 * fs];
        for (int k = 1; k < F - 4; ++k) {
            const float s = p[(4 + k) * fs];
            if (s > score) { score = s; best = k; }
        }
        if (score < conf) continue;
        float b0 = p[0], b1 = p[fs], b2 = p[2 * fs], b3 = p[3 * fs];
        if (!xyxy) {
            const float hw = 0.5f * b2, hh = 0.5f * b3, cx = b0, cy = b1;
            b0 = cx - hw; b1 = cy - hh; b2 = cx + hw; b3 = cy + hh;
        }
        cands.push_back({score, clampf(b0), clampf(b1), clampf(b2), clampf(b3), best});
    }
}
