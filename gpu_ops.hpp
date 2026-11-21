// Tile buffers and pre/post-processing for the ONNX Runtime backend.
//
// Two implementations with the same interface:
//   gpu_ops_cuda.cu - CUDA (nvcc): buffers in GPU memory, work done by kernels.
//   gpu_ops_cpu.cpp - plain C++: buffers in host memory (ONNX Runtime CPU build).
// Plain C++ header: no CUDA types, so it can be included from any .cpp file.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tile_ops.hpp"

namespace tileops {

struct Context;

// True when buffers live in CUDA device memory (bind them with a "Cuda" OrtMemoryInfo).
bool on_gpu();

// Select the GPU (no-op on CPU) and return its name, or false with an error message.
bool init_device(int device, std::string& name, std::string& error);

// Buffers for up to max_batch tiles of size x size, with a head output of
// n_feat x n_anchor values per tile. Returns null and sets error on failure.
Context* create(int max_batch, int size, int n_feat, int n_anchor, std::string& error);
void destroy(Context* ctx);

// Network input [max_batch][3][size][size] and head output [max_batch][...]
// buffers, to bind to ONNX Runtime.
float* input_buffer(Context* ctx);
float* output_buffer(Context* ctx);

// Make an 8-bit BGR image (rows `step` bytes apart) the source for preprocess().
// The GPU version copies it to device memory; the CPU version keeps the pointer.
bool set_image(Context* ctx, const uint8_t* bgr, int width, int height, size_t step, std::string& error);

// Write the network input for n tiles, tile i having its top-left corner at
// (xy[2 i], xy[2 i + 1]): RGB, 0..1, planar (CHW). Finishes before returning.
bool preprocess(Context* ctx, const int* xy, int n, std::string& error);

// For n tiles of head output: the best class per anchor; keeps anchors scoring
// at least conf. out[i] gets tile i's candidates sorted by anchor index.
bool decode(Context* ctx, int n, bool feature_rows, bool xyxy, float conf,
            std::vector<std::vector<Candidate>>& out, std::string& error);

}  // namespace tileops
