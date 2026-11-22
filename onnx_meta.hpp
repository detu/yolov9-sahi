// Reads what the program needs from an .onnx file without an ONNX library:
// the metadata (class names, input size, ...) and the input/output shapes.
#pragma once

#include <map>
#include <string>
#include <vector>

struct OnnxDim {
    long long value = -1;  // -1 when symbolic (dynamic), e.g. "batch"
    std::string param;
};

struct OnnxInfo {
    std::map<std::string, std::string> metadata;  // metadata_props key -> value
    std::vector<std::string> input_names, output_names;
    std::vector<std::vector<OnnxDim>> input_dims, output_dims;
};

// Returns false and sets error when the file can't be read or isn't ONNX.
bool read_onnx_info(const std::string& path, OnnxInfo& info, std::string& error);

// Parse a JSON object of string values keyed by integers, e.g. LibreYOLO's
// names metadata {"0": "person", "1": ...}, into a vector indexed by key.
std::vector<std::string> parse_names_json(const std::string& s);
