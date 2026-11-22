// Detector on top of an inference engine (runner.hpp): letterboxes the regions
// of an image, runs them in batches, decodes and maps the boxes back.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "detector.hpp"
#include "runner.hpp"

struct DetectorInfo {
    std::string backend;     // what actually runs, e.g. "CUDA"
    std::string in_name, out_name;
    int batch = 1;
    bool xyxy = true;
    std::string box_source;  // "--box", "metadata", "detected"
    int n_feat = 0, n_anchor = 0;
};

class RegionDetector {
public:
    // Opens the model and checks it on a blank input. Returns null and sets
    // error on failure.
    static std::unique_ptr<RegionDetector> create(const EngineConfig& cfg, DetectorInfo& info, std::string& error);

    // Run the model on every region (letterboxed to the input size). per_region[i]
    // gets region i's detections in image coordinates, after class-aware NMS
    // within the region. Returns the number of regions that failed.
    int detect(const cv::Mat& bgr, const std::vector<Region>& regions, float conf, float nms_iou,
               std::vector<std::vector<Detection>>& per_region);

    std::string last_run_desc() const { return last_; }

private:
    RegionDetector() = default;
    std::unique_ptr<Runner> runner_;
    EngineConfig cfg_;
    DetectorInfo info_;
    bool feature_rows_ = false;
    std::string last_;
};
