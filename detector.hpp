// Shared types: detections, regions and class-aware NMS.
#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

struct Detection {
    cv::Rect2f box;  // (x, y, width, height)
    float confidence;
    int class_id;
};

struct Region {
    int x, y, width, height;
};

// Class-aware NMS: classes don't suppress each other. Output order is
// deterministic (sorted by class id, then score).
inline std::vector<Detection> apply_class_aware_nms(const std::vector<Detection>& raw_dets, float conf_threshold,
                                                    float nms_iou_threshold) {
    std::vector<Detection> final_dets;
    if (raw_dets.empty()) return final_dets;

    std::map<int, std::vector<size_t>> class_indices;
    for (size_t i = 0; i < raw_dets.size(); ++i) class_indices[raw_dets[i].class_id].push_back(i);

    for (const auto& kv : class_indices) {
        const auto& indices = kv.second;
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        boxes.reserve(indices.size());
        scores.reserve(indices.size());
        for (size_t idx : indices) {
            const auto& b = raw_dets[idx].box;
            boxes.emplace_back(b.x, b.y, b.width, b.height);
            scores.push_back(raw_dets[idx].confidence);
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_iou_threshold, keep);
        for (int k : keep) final_dets.push_back(raw_dets[indices[k]]);
    }
    return final_dets;
}
