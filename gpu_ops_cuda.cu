// CUDA implementation of gpu_ops.hpp (compiled with nvcc).
//
// Per image: the BGR image is copied to the GPU once. Per batch of tiles:
//   preprocess_kernel - letterboxes each region (bilinear resize + gray pad,
//                       or a 1:1 copy for regions of the input size) and writes
//                       the network input directly (BGR -> RGB, 0..1, planar),
//                       one thread per pixel, so regions never pass through
//                       host memory.
//   decode_kernel     - one thread per anchor picks the best class; anchors at
//                       or above the threshold are appended to a per-tile list
//                       with atomicAdd. Only those few candidates are copied
//                       back instead of the whole head output (~2.7 MB/tile).

#include "gpu_ops.hpp"

#include <algorithm>
#include <cuda_runtime.h>

namespace tileops {

struct Context {
    int max_batch = 0, size = 0, n_feat = 0, n_anchor = 0;
    float* d_input = nullptr;       // [max_batch][3][size][size]
    float* d_output = nullptr;      // [max_batch][n_feat * n_anchor]
    Candidate* d_cands = nullptr;   // [max_batch][n_anchor]
    int* d_counts = nullptr;        // [max_batch]
    RegionXform* d_regions = nullptr;  // [max_batch]
    uint8_t* d_image = nullptr;     // current image, rows `image_step` bytes apart
    size_t image_capacity = 0, image_step = 0;
    std::vector<int> h_counts;
};

static bool check(cudaError_t e, const char* what, std::string& error) {
    if (e == cudaSuccess) return true;
    error = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

__global__ void preprocess_kernel(const uint8_t* __restrict__ img, size_t step,
                                  const RegionXform* __restrict__ regions, int size, float* __restrict__ input) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= size || y >= size) return;
    const RegionXform t = regions[b];
    const size_t plane = static_cast<size_t>(size) * size;
    float* dst = input + static_cast<size_t>(b) * 3 * plane + static_cast<size_t>(y) * size + x;
    for (int c = 0; c < 3; ++c) dst[c * plane] = letterbox_input_value(img, step, t, c, y, x);
}

__global__ void decode_kernel(const float* __restrict__ output, int F, int A, bool feature_rows, bool xyxy,
                              int size, float conf, Candidate* __restrict__ cands, int* __restrict__ counts) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (a >= A) return;
    const Candidate c = decode_anchor(output + static_cast<size_t>(b) * F * A, F, A, feature_rows, xyxy, size, a);
    if (c.score >= conf) {
        const int k = atomicAdd(&counts[b], 1);
        cands[static_cast<size_t>(b) * A + k] = c;
    }
}

bool on_gpu() { return true; }

bool init_device(int device, std::string& name, std::string& error) {
    int n = 0;
    if (!check(cudaGetDeviceCount(&n), "cudaGetDeviceCount", error)) return false;
    if (device < 0 || device >= n) {
        error = "CUDA device " + std::to_string(device) + " not found (" + std::to_string(n) + " available)";
        return false;
    }
    if (!check(cudaSetDevice(device), "cudaSetDevice", error)) return false;
    cudaDeviceProp prop{};
    if (!check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties", error)) return false;
    name = prop.name;
    return true;
}

Context* create(int max_batch, int size, int n_feat, int n_anchor, std::string& error) {
    auto* ctx = new Context;
    ctx->max_batch = max_batch;
    ctx->size = size;
    ctx->n_feat = n_feat;
    ctx->n_anchor = n_anchor;
    ctx->h_counts.resize(max_batch);
    const size_t n_in = static_cast<size_t>(max_batch) * 3 * size * size;
    const size_t n_out = static_cast<size_t>(max_batch) * n_feat * n_anchor;
    if (!check(cudaMalloc(&ctx->d_input, n_in * sizeof(float)), "cudaMalloc(input)", error) ||
        !check(cudaMalloc(&ctx->d_output, n_out * sizeof(float)), "cudaMalloc(output)", error) ||
        !check(cudaMalloc(&ctx->d_cands, static_cast<size_t>(max_batch) * n_anchor * sizeof(Candidate)),
               "cudaMalloc(candidates)", error) ||
        !check(cudaMalloc(&ctx->d_counts, max_batch * sizeof(int)), "cudaMalloc(counts)", error) ||
        !check(cudaMalloc(&ctx->d_regions, max_batch * sizeof(RegionXform)), "cudaMalloc(regions)", error)) {
        destroy(ctx);
        return nullptr;
    }
    return ctx;
}

void destroy(Context* ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_input);
    cudaFree(ctx->d_output);
    cudaFree(ctx->d_cands);
    cudaFree(ctx->d_counts);
    cudaFree(ctx->d_regions);
    cudaFree(ctx->d_image);
    delete ctx;
}

float* input_buffer(Context* ctx) { return ctx->d_input; }
float* output_buffer(Context* ctx) { return ctx->d_output; }

bool set_image(Context* ctx, const uint8_t* bgr, int width, int height, size_t step, std::string& error) {
    const size_t row = static_cast<size_t>(width) * 3;
    const size_t bytes = row * height;
    if (bytes > ctx->image_capacity) {
        cudaFree(ctx->d_image);
        ctx->d_image = nullptr;
        ctx->image_capacity = 0;
        if (!check(cudaMalloc(&ctx->d_image, bytes), "cudaMalloc(image)", error)) return false;
        ctx->image_capacity = bytes;
    }
    ctx->image_step = row;  // packed on the GPU
    return check(cudaMemcpy2D(ctx->d_image, row, bgr, step, row, height, cudaMemcpyHostToDevice),
                 "cudaMemcpy2D(image)", error);
}

bool preprocess(Context* ctx, const RegionXform* regions, int n, std::string& error) {
    if (n <= 0 || n > ctx->max_batch) {
        error = "preprocess: bad batch size";
        return false;
    }
    if (!check(cudaMemcpy(ctx->d_regions, regions, n * sizeof(RegionXform), cudaMemcpyHostToDevice),
               "cudaMemcpy(regions)", error))
        return false;
    const dim3 block(32, 8);
    const dim3 grid((ctx->size + block.x - 1) / block.x, (ctx->size + block.y - 1) / block.y, n);
    preprocess_kernel<<<grid, block>>>(ctx->d_image, ctx->image_step, ctx->d_regions, ctx->size, ctx->d_input);
    if (!check(cudaGetLastError(), "preprocess_kernel", error)) return false;
    // ONNX Runtime runs on its own CUDA stream: the input must be complete first.
    return check(cudaDeviceSynchronize(), "preprocess_kernel", error);
}

bool decode(Context* ctx, int n, bool feature_rows, bool xyxy, float conf,
            std::vector<std::vector<Candidate>>& out, std::string& error) {
    if (n <= 0 || n > ctx->max_batch) {
        error = "decode: bad batch size";
        return false;
    }
    if (!check(cudaMemset(ctx->d_counts, 0, n * sizeof(int)), "cudaMemset(counts)", error)) return false;
    const int threads = 256;
    const dim3 grid((ctx->n_anchor + threads - 1) / threads, n);
    decode_kernel<<<grid, threads>>>(ctx->d_output, ctx->n_feat, ctx->n_anchor, feature_rows, xyxy, ctx->size,
                                     conf, ctx->d_cands, ctx->d_counts);
    if (!check(cudaGetLastError(), "decode_kernel", error)) return false;
    if (!check(cudaMemcpy(ctx->h_counts.data(), ctx->d_counts, n * sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy(counts)", error))
        return false;
    out.assign(n, {});
    for (int b = 0; b < n; ++b) {
        const int k = ctx->h_counts[b];
        if (k <= 0) continue;
        out[b].resize(k);
        if (!check(cudaMemcpy(out[b].data(), ctx->d_cands + static_cast<size_t>(b) * ctx->n_anchor,
                              k * sizeof(Candidate), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(candidates)", error))
            return false;
        // atomicAdd order varies between runs; anchor order keeps results reproducible
        std::sort(out[b].begin(), out[b].end(),
                  [](const Candidate& p, const Candidate& q) { return p.anchor < q.anchor; });
    }
    return true;
}

}  // namespace tileops
