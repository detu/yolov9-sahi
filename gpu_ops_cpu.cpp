// CPU implementation of gpu_ops.hpp, for the ONNX Runtime CPU build. Uses the
// same per-element functions (tile_ops.hpp) as the CUDA kernels.

#include "gpu_ops.hpp"

#include <algorithm>

namespace tileops {

struct Context {
    int max_batch = 0, size = 0, n_feat = 0, n_anchor = 0;
    std::vector<float> input, output;
    const uint8_t* image = nullptr;
    size_t image_step = 0;
};

bool on_gpu() { return false; }

bool init_device(int, std::string& name, std::string&) {
    name = "CPU";
    return true;
}

Context* create(int max_batch, int size, int n_feat, int n_anchor, std::string&) {
    auto* ctx = new Context;
    ctx->max_batch = max_batch;
    ctx->size = size;
    ctx->n_feat = n_feat;
    ctx->n_anchor = n_anchor;
    ctx->input.resize(static_cast<size_t>(max_batch) * 3 * size * size);
    ctx->output.resize(static_cast<size_t>(max_batch) * n_feat * n_anchor);
    return ctx;
}

void destroy(Context* ctx) { delete ctx; }

float* input_buffer(Context* ctx) { return ctx->input.data(); }
float* output_buffer(Context* ctx) { return ctx->output.data(); }

bool set_image(Context* ctx, const uint8_t* bgr, int, int, size_t step, std::string&) {
    ctx->image = bgr;
    ctx->image_step = step;
    return true;
}

bool preprocess(Context* ctx, const RegionXform* regions, int n, std::string& error) {
    if (n <= 0 || n > ctx->max_batch) {
        error = "preprocess: bad batch size";
        return false;
    }
    const int S = ctx->size;
    const size_t plane = static_cast<size_t>(S) * S;
    for (int b = 0; b < n; ++b) {
        float* dst = ctx->input.data() + static_cast<size_t>(b) * 3 * plane;
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < S; ++y)
                for (int x = 0; x < S; ++x)
                    dst[c * plane + static_cast<size_t>(y) * S + x] =
                        letterbox_input_value(ctx->image, ctx->image_step, regions[b], c, y, x);
    }
    return true;
}

bool decode(Context* ctx, int n, bool feature_rows, bool xyxy, float conf,
            std::vector<std::vector<Candidate>>& out, std::string& error) {
    if (n <= 0 || n > ctx->max_batch) {
        error = "decode: bad batch size";
        return false;
    }
    const int F = ctx->n_feat, A = ctx->n_anchor;
    out.assign(n, {});
    for (int b = 0; b < n; ++b) {
        const float* o = ctx->output.data() + static_cast<size_t>(b) * F * A;
        for (int a = 0; a < A; ++a) {
            const Candidate c = decode_anchor(o, F, A, feature_rows, xyxy, ctx->size, a);
            if (c.score >= conf) out[b].push_back(c);  // already in anchor order
        }
    }
    return true;
}

}  // namespace tileops
