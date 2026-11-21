// ONNX Runtime backend: CUDA execution provider (SAHI_BACKEND=ORT_CUDA) or
// CPU (SAHI_BACKEND=ORT_CPU).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "detector.hpp"

struct OrtConfig {
    std::string model_path;  // .onnx
    int size = 0;            // input size override (0 = metadata, input shape, else 640)
    int batch = 8;           // tiles per run, if the model has a dynamic batch dimension
    int device = 0;          // CUDA device
    int threads = 0;         // CPU build: intra-op threads (0 = ONNX Runtime default)
    std::string box = "auto";
    std::string pad = "auto";  // letterbox placement: auto (metadata, else center), center, topleft
    std::string in_name, out_name;
};

struct OrtModelInfo {
    std::string device;                // e.g. "CUDA (NVIDIA RTX 1000 Ada Generation Laptop GPU)"
    std::string in_name, out_name;
    int size = 640;
    int batch = 1;
    bool dynamic_batch = false;
    bool xyxy = false;
    std::string box_source;            // "--box", "metadata", "detected"
    bool topleft = false;              // letterbox at the top-left (LibreYOLO checkpoints), else centred
    std::string pad_source;            // "--pad", "metadata", "default"
    std::vector<std::string> names;    // from the model's metadata, if any
    std::string family;                // LibreYOLO model_family, if any
};

// Loads the model and checks it on a blank input. Returns null and sets error on failure.
std::unique_ptr<TileDetector> create_ort_detector(const OrtConfig& cfg, OrtModelInfo& info, std::string& error);
