// Regions (SAHI slices, the optional full image) are letterboxed on the CPU
// (OpenMP, one region per thread) straight into the engine's input buffer, run
// in batches, and decoded on the CPU; only boxes above the threshold are kept.

#include "region_detector.hpp"

#include <algorithm>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

std::unique_ptr<RegionDetector> RegionDetector::create(const EngineConfig& cfg, DetectorInfo& info,
                                                       std::string& error) {
    std::unique_ptr<RegionDetector> d(new RegionDetector);
    d->cfg_ = cfg;
    RunnerInfo ri;
    d->runner_ = create_runner(cfg, ri, error);
    if (!d->runner_) return nullptr;
    info.backend = ri.backend;
    info.in_name = ri.in_name;
    info.out_name = ri.out_name;
    info.batch = ri.batch;

    // Probe on a gray input: checks the run and learns the output layout
    const size_t n_in = static_cast<size_t>(info.batch) * 3 * cfg.size * cfg.size;
    std::fill(d->runner_->input(), d->runner_->input() + n_in, 0.f);
    std::string err;
    if (!d->runner_->run(err)) {
        error = "test run failed: " + err;
        return nullptr;
    }
    const std::vector<int64_t> s = d->runner_->output_shape();
    if (s.size() != 3 || s[0] != info.batch) {
        std::string sh;
        for (int64_t v : s) sh += (sh.empty() ? "" : ",") + std::to_string(v);
        error = "output has shape [" + sh + "], expected [" + std::to_string(info.batch) +
                ", anchors, 4 + classes]";
        return nullptr;
    }
    d->feature_rows_ = s[1] <= s[2];  // far more anchors than features
    info.n_feat = static_cast<int>(d->feature_rows_ ? s[1] : s[2]);
    info.n_anchor = static_cast<int>(d->feature_rows_ ? s[2] : s[1]);
    if (info.n_feat < 5) {
        error = "output has " + std::to_string(info.n_feat) + " values per anchor; expected 4 box values + class scores";
        return nullptr;
    }
    if (cfg.box != "auto") {
        info.xyxy = cfg.box == "xyxy";
        info.box_source = "--box";
    } else if (cfg.box_from_metadata) {
        info.xyxy = true;
        info.box_source = "metadata";
    } else {
        // Corner boxes always have x2 >= x1 and y2 >= y1; centre boxes fail that
        // somewhere among the anchors (centre further out than the width).
        const float* p = d->runner_->output();
        const bool fr = d->feature_rows_;
        bool xyxy = true;
        for (int a = 0; a < info.n_anchor && xyxy; ++a) {
            auto at = [&](int f) {
                return fr ? p[static_cast<size_t>(f) * info.n_anchor + a] : p[static_cast<size_t>(a) * info.n_feat + f];
            };
            if (at(2) < at(0) || at(3) < at(1)) xyxy = false;
        }
        info.xyxy = xyxy;
        info.box_source = "detected";
    }
    d->info_ = info;
    return d;
}

int RegionDetector::detect(const cv::Mat& bgr, const std::vector<Region>& regions, float conf, float nms_iou,
                           std::vector<std::vector<Detection>>& per_region) {
    per_region.assign(regions.size(), {});
    const int S = cfg_.size, B = info_.batch, F = info_.n_feat, A = info_.n_anchor;
    const size_t in_per = static_cast<size_t>(3) * S * S, out_per = static_cast<size_t>(F) * A;
    int failed = 0, runs = 0;
    std::vector<RegionXform> xf(B);
    for (size_t start = 0; start < regions.size(); start += B) {
        // Always a full batch: unused slots keep earlier data and are ignored,
        // so the input shape never changes (MIGraphX would recompile).
        const int n = static_cast<int>(std::min<size_t>(B, regions.size() - start));
        ++runs;
        float* in = runner_->input();

        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < n; ++i) {
            const Region& r = regions[start + i];
            xf[i] = make_xform(r.x, r.y, r.width, r.height, S, cfg_.topleft);
            fill_region_input(bgr.data, bgr.step[0], xf[i], S, cfg_.norm, in + i * in_per);
        }

        std::string err;
        if (!runner_->run(err)) {
            std::cerr << "Inference failed on regions " << start << ".." << (start + n - 1) << ": " << err << "\n";
            failed += n;
            continue;
        }

        const float* out = runner_->output();
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < n; ++i) {
            const RegionXform& t = xf[i];
            std::vector<Candidate> cands;
            decode_head(out + i * out_per, F, A, feature_rows_, info_.xyxy, S, conf, cands);
            std::vector<Detection> dets;
            dets.reserve(cands.size());
            for (const Candidate& c : cands) {
                // Undo the letterbox, clamp to the region, then shift to image coordinates
                const float x1 = std::clamp((c.x1 - t.pad_l) / t.scale, 0.0f, static_cast<float>(t.rw));
                const float y1 = std::clamp((c.y1 - t.pad_t) / t.scale, 0.0f, static_cast<float>(t.rh));
                const float x2 = std::clamp((c.x2 - t.pad_l) / t.scale, 0.0f, static_cast<float>(t.rw));
                const float y2 = std::clamp((c.y2 - t.pad_t) / t.scale, 0.0f, static_cast<float>(t.rh));
                if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;
                dets.push_back({cv::Rect2f(x1 + t.rx, y1 + t.ry, x2 - x1, y2 - y1), c.score, c.cls});
            }
            // The head predicts many overlapping boxes per object
            per_region[start + i] = apply_class_aware_nms(dets, conf, nms_iou);
        }
    }
    last_ = "batch " + std::to_string(B) + ", " + std::to_string(runs) + " run(s)";
    return failed;
}
