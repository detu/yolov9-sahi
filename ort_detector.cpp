// ONNX Runtime backend. Regions (SAHI slices, the optional full image) are
// letterboxed to the input size and batched; with the CUDA build, the network
// input and output stay in GPU memory (bound with IoBinding), preprocessing and
// decoding run as CUDA kernels (gpu_ops_cuda.cu), and only the candidate boxes
// come back to the host.

#include "ort_detector.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <onnxruntime_cxx_api.h>

#include "gpu_ops.hpp"

namespace {

// Parse LibreYOLO's names metadata: a JSON object {"0": "person", "1": ...}.
std::vector<std::string> parse_names_json(const std::string& s) {
    std::vector<std::pair<int, std::string>> items;
    size_t i = 0;
    auto skip_ws = [&]() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; };
    auto read_string = [&](std::string& outs) -> bool {
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        while (i < s.size() && s[i] != '"') {
            char ch = s[i++];
            if (ch != '\\') { outs += ch; continue; }
            if (i >= s.size()) return false;
            char e = s[i++];
            switch (e) {
                case 'n': outs += '\n'; break;
                case 't': outs += '\t'; break;
                case 'r': outs += '\r'; break;
                case 'b': outs += '\b'; break;
                case 'f': outs += '\f'; break;
                case 'u': {
                    if (i + 4 > s.size()) return false;
                    unsigned cp = static_cast<unsigned>(std::strtoul(s.substr(i, 4).c_str(), nullptr, 16));
                    i += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        unsigned lo = static_cast<unsigned>(std::strtoul(s.substr(i + 2, 4).c_str(), nullptr, 16));
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            i += 6;
                        }
                    }
                    if (cp < 0x80) outs += static_cast<char>(cp);
                    else if (cp < 0x800) { outs += static_cast<char>(0xC0 | (cp >> 6)); outs += static_cast<char>(0x80 | (cp & 0x3F)); }
                    else if (cp < 0x10000) { outs += static_cast<char>(0xE0 | (cp >> 12)); outs += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); outs += static_cast<char>(0x80 | (cp & 0x3F)); }
                    else { outs += static_cast<char>(0xF0 | (cp >> 18)); outs += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); outs += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); outs += static_cast<char>(0x80 | (cp & 0x3F)); }
                    break;
                }
                default: outs += e;  // \" \\ \/
            }
        }
        if (i >= s.size()) return false;
        ++i;  // closing quote
        return true;
    };
    skip_ws();
    if (i >= s.size() || s[i] != '{') return {};
    ++i;
    while (true) {
        skip_ws();
        if (i < s.size() && s[i] == '}') break;
        std::string key, value;
        if (!read_string(key)) return {};
        skip_ws();
        if (i >= s.size() || s[i] != ':') return {};
        ++i;
        skip_ws();
        if (!read_string(value)) return {};
        if (!key.empty() && key.find_first_not_of("0123456789") == std::string::npos)
            items.emplace_back(std::atoi(key.c_str()), value);
        skip_ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        break;
    }
    std::vector<std::string> names;
    for (const auto& kv : items) {
        if (kv.first >= static_cast<int>(names.size())) names.resize(kv.first + 1);
        names[kv.first] = kv.second;
    }
    return names;
}

class OrtDetector : public TileDetector {
public:
    OrtDetector(Ort::Env&& env, Ort::Session&& session, tileops::Context* ctx, const OrtModelInfo& info,
                int device, int n_feat, int n_anchor, bool feature_rows, bool out_has_batch)
        : env_(std::move(env)),
          session_(std::move(session)),
          io_(session_),
          mem_(tileops::on_gpu() ? Ort::MemoryInfo("Cuda", OrtDeviceAllocator, device, OrtMemTypeDefault)
                                 : Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
          ctx_(ctx),
          info_(info),
          n_feat_(n_feat),
          n_anchor_(n_anchor),
          feature_rows_(feature_rows),
          out_has_batch_(out_has_batch) {}

    ~OrtDetector() override { tileops::destroy(ctx_); }

    int detect(const cv::Mat& work, const std::vector<Region>& tiles, float conf, float nms_iou,
               std::vector<std::vector<Detection>>& per_tile) override {
        per_tile.assign(tiles.size(), {});
        std::string err;
        if (!tileops::set_image(ctx_, work.data, work.cols, work.rows, work.step[0], err)) {
            std::cerr << "Cannot upload image: " << err << "\n";
            return static_cast<int>(tiles.size());
        }
        const int S = info_.size;
        int failed = 0, batches = 0;
        std::vector<RegionXform> xf;
        std::vector<std::vector<Candidate>> cands;
        for (size_t start = 0; start < tiles.size(); start += info_.batch) {
            const int n = static_cast<int>(std::min<size_t>(info_.batch, tiles.size() - start));
            ++batches;
            try {
                xf.resize(n);
                for (int i = 0; i < n; ++i) {
                    // Letterbox: scale the region to fit size x size, keep its aspect ratio
                    const Region& r = tiles[start + i];
                    RegionXform& t = xf[i];
                    t.rx = r.x; t.ry = r.y; t.rw = r.width; t.rh = r.height;
                    t.scale = std::min(static_cast<float>(S) / r.width, static_cast<float>(S) / r.height);
                    t.nw = std::min(S, std::max(1, static_cast<int>(r.width * t.scale)));
                    t.nh = std::min(S, std::max(1, static_cast<int>(r.height * t.scale)));
                    t.pad_l = info_.topleft ? 0 : (S - t.nw) / 2;
                    t.pad_t = info_.topleft ? 0 : (S - t.nh) / 2;
                }
                if (!tileops::preprocess(ctx_, xf.data(), n, err)) throw std::runtime_error(err);

                const std::vector<int64_t> in_shape = {n, 3, S, S};
                Ort::Value in = Ort::Value::CreateTensor<float>(mem_, tileops::input_buffer(ctx_),
                                                                static_cast<size_t>(n) * 3 * S * S,
                                                                in_shape.data(), in_shape.size());
                std::vector<int64_t> out_shape;
                if (out_has_batch_) out_shape.push_back(n);
                if (feature_rows_) { out_shape.push_back(n_feat_); out_shape.push_back(n_anchor_); }
                else { out_shape.push_back(n_anchor_); out_shape.push_back(n_feat_); }
                Ort::Value out = Ort::Value::CreateTensor<float>(mem_, tileops::output_buffer(ctx_),
                                                                 static_cast<size_t>(n) * n_feat_ * n_anchor_,
                                                                 out_shape.data(), out_shape.size());
                io_.ClearBoundInputs();
                io_.ClearBoundOutputs();
                io_.BindInput(info_.in_name.c_str(), in);
                io_.BindOutput(info_.out_name.c_str(), out);
                session_.Run(Ort::RunOptions{nullptr}, io_);
                io_.SynchronizeOutputs();

                if (!tileops::decode(ctx_, n, feature_rows_, info_.xyxy, conf, cands, err)) throw std::runtime_error(err);
                for (int i = 0; i < n; ++i) {
                    const RegionXform& t = xf[i];
                    std::vector<Detection> dets;
                    dets.reserve(cands[i].size());
                    for (const Candidate& c : cands[i]) {
                        // Undo the letterbox, clamp to the region, then shift to image coordinates
                        const float x1 = std::clamp((c.x1 - t.pad_l) / t.scale, 0.0f, static_cast<float>(t.rw));
                        const float y1 = std::clamp((c.y1 - t.pad_t) / t.scale, 0.0f, static_cast<float>(t.rh));
                        const float x2 = std::clamp((c.x2 - t.pad_l) / t.scale, 0.0f, static_cast<float>(t.rw));
                        const float y2 = std::clamp((c.y2 - t.pad_t) / t.scale, 0.0f, static_cast<float>(t.rh));
                        if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;
                        dets.push_back({cv::Rect2f(x1 + t.rx, y1 + t.ry, x2 - x1, y2 - y1), c.score, c.cls});
                    }
                    // The head predicts many overlapping boxes per object
                    per_tile[start + i] = apply_class_aware_nms(dets, conf, nms_iou);
                }
            } catch (const std::exception& e) {
                std::cerr << "Inference failed on regions " << start << ".." << (start + n - 1) << ": " << e.what() << "\n";
                failed += n;
            }
        }
        last_ = "batch " + std::to_string(info_.batch) + ", " + std::to_string(batches) + " run(s)";
        return failed;
    }

    std::string last_run_desc() const override { return last_; }

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::IoBinding io_;
    Ort::MemoryInfo mem_;
    tileops::Context* ctx_;
    OrtModelInfo info_;
    int n_feat_, n_anchor_;
    bool feature_rows_, out_has_batch_;
    std::string last_;
};

}  // namespace

std::unique_ptr<TileDetector> create_ort_detector(const OrtConfig& cfg, OrtModelInfo& info, std::string& error) {
    try {
        std::string dev_name;
        if (!tileops::init_device(cfg.device, dev_name, error)) return nullptr;

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov9_sahi");
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (tileops::on_gpu()) {
            const OrtApi& api = Ort::GetApi();
            OrtCUDAProviderOptionsV2* cuda = nullptr;
            Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
            const std::string dev = std::to_string(cfg.device);
            const char* keys[] = {"device_id"};
            const char* values[] = {dev.c_str()};
            OrtStatus* st = api.UpdateCUDAProviderOptions(cuda, keys, values, 1);
            if (st) {
                api.ReleaseCUDAProviderOptions(cuda);
                Ort::ThrowOnError(st);
            }
            try {
                so.AppendExecutionProvider_CUDA_V2(*cuda);
            } catch (...) {
                api.ReleaseCUDAProviderOptions(cuda);
                throw;
            }
            api.ReleaseCUDAProviderOptions(cuda);
            info.device = "CUDA (" + dev_name + ")";
        } else {
            if (cfg.threads > 0) so.SetIntraOpNumThreads(cfg.threads);
            info.device = "CPU (ONNX Runtime)";
        }

        Ort::Session session(env, cfg.model_path.c_str(), so);
        Ort::AllocatorWithDefaultOptions alloc;

        // Names, input size and box format: options win, then the model's metadata
        Ort::ModelMetadata md = session.GetModelMetadata();
        auto meta = [&](const char* key) {
            Ort::AllocatedStringPtr v = md.LookupCustomMetadataMapAllocated(key, alloc);
            return v ? std::string(v.get()) : std::string();
        };
        info.names = parse_names_json(meta("names"));
        info.family = meta("model_family");
        const bool libreyolo = !meta("libreyolo_version").empty();
        const int meta_imgsz = std::atoi(meta("imgsz").c_str());
        const std::string meta_pad = meta("letterbox_pad");
        if (cfg.pad != "auto") {
            info.topleft = cfg.pad == "topleft";
            info.pad_source = "--pad";
        } else if (meta_pad == "topleft" || meta_pad == "center" || meta_pad == "centre") {
            info.topleft = meta_pad == "topleft";
            info.pad_source = "metadata";
        } else {
            info.topleft = false;
            info.pad_source = "default";
        }

        if (session.GetInputCount() < 1 || session.GetOutputCount() < 1) {
            error = "model needs at least one input and one output";
            return nullptr;
        }
        info.in_name = cfg.in_name.empty() ? session.GetInputNameAllocated(0, alloc).get() : cfg.in_name;
        info.out_name = cfg.out_name.empty() ? session.GetOutputNameAllocated(0, alloc).get() : cfg.out_name;

        // Input shape [batch, 3, H, W]; -1 means dynamic
        size_t in_index = 0;
        for (size_t k = 0; k < session.GetInputCount(); ++k)
            if (info.in_name == session.GetInputNameAllocated(k, alloc).get()) in_index = k;
        const std::vector<int64_t> in_shape =
            session.GetInputTypeInfo(in_index).GetTensorTypeAndShapeInfo().GetShape();
        if (in_shape.size() != 4) {
            error = "input '" + info.in_name + "' has rank " + std::to_string(in_shape.size()) + ", expected [batch, 3, H, W]";
            return nullptr;
        }
        info.size = cfg.size > 0 ? cfg.size
                  : meta_imgsz >= 32 ? meta_imgsz
                  : in_shape[2] > 0 ? static_cast<int>(in_shape[2])
                  : 640;
        if ((in_shape[2] > 0 && in_shape[2] != info.size) || (in_shape[3] > 0 && in_shape[3] != info.size)) {
            error = "model input is " + std::to_string(in_shape[2]) + "x" + std::to_string(in_shape[3]) +
                    " but the tile size is " + std::to_string(info.size) + "; export a square model or pass --size";
            return nullptr;
        }
        info.dynamic_batch = in_shape[0] < 0;
        info.batch = info.dynamic_batch ? std::max(1, cfg.batch) : static_cast<int>(std::max<int64_t>(1, in_shape[0]));

        // Probe on a blank input: checks the names and learns the output layout
        const int S = info.size;
        std::vector<float> probe(static_cast<size_t>(3) * S * S, 0.5f);
        const std::vector<int64_t> probe_shape = {1, 3, S, S};
        Ort::MemoryInfo cpu = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        Ort::Value pin = Ort::Value::CreateTensor<float>(cpu, probe.data(), probe.size(), probe_shape.data(), probe_shape.size());
        const char* in_names[] = {info.in_name.c_str()};
        const char* out_names[] = {info.out_name.c_str()};
        std::vector<Ort::Value> outs = session.Run(Ort::RunOptions{nullptr}, in_names, &pin, 1, out_names, 1);
        const std::vector<int64_t> oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        bool out_has_batch;
        int64_t d0, d1;
        if (oshape.size() == 3 && oshape[0] == 1) { out_has_batch = true; d0 = oshape[1]; d1 = oshape[2]; }
        else if (oshape.size() == 2) { out_has_batch = false; d0 = oshape[0]; d1 = oshape[1]; }
        else {
            std::string sh;
            for (auto d : oshape) sh += (sh.empty() ? "" : ",") + std::to_string(d);
            error = "output '" + info.out_name + "' has shape [" + sh + "], expected [1, 4+classes, anchors]";
            return nullptr;
        }
        if (!out_has_batch) info.batch = 1;
        const bool feature_rows = d0 <= d1;  // far more anchors than features
        const int n_feat = static_cast<int>(feature_rows ? d0 : d1);
        const int n_anchor = static_cast<int>(feature_rows ? d1 : d0);
        if (n_feat < 5) {
            error = "output has " + std::to_string(n_feat) + " values per anchor; expected 4 box values + class scores";
            return nullptr;
        }

        if (cfg.box != "auto") {
            info.xyxy = cfg.box == "xyxy";
            info.box_source = "--box";
        } else if (libreyolo && info.family.rfind("yolo9", 0) == 0) {
            info.xyxy = true;  // LibreYOLO's YOLOv9 head exports x1,y1,x2,y2
            info.box_source = "metadata";
        } else {
            // Corner boxes always have x2 >= x1 and y2 >= y1; centre boxes fail
            // that somewhere among the anchors (centre further out than the width).
            const float* p = outs[0].GetTensorData<float>();
            bool xyxy = true;
            for (int a = 0; a < n_anchor && xyxy; ++a) {
                auto at = [&](int f) { return feature_rows ? p[static_cast<size_t>(f) * n_anchor + a] : p[static_cast<size_t>(a) * n_feat + f]; };
                if (at(2) < at(0) || at(3) < at(1)) xyxy = false;
            }
            info.xyxy = xyxy;
            info.box_source = "detected";
        }

        tileops::Context* ctx = tileops::create(info.batch, S, n_feat, n_anchor, error);
        if (!ctx) return nullptr;
        return std::make_unique<OrtDetector>(std::move(env), std::move(session), ctx, info, cfg.device, n_feat, n_anchor,
                                             feature_rows, out_has_batch);
    } catch (const Ort::Exception& e) {
        error = e.what();
        if (tileops::on_gpu() && (error.find("libonnxruntime_providers_cuda") != std::string::npos ||
                                  error.find("cudnn") != std::string::npos || error.find("CUDA") != std::string::npos)) {
            error += "\n(The CUDA execution provider needs libonnxruntime_providers_cuda.so next to libonnxruntime.so,"
                     " and the CUDA and cuDNN 9 libraries matching the ONNX Runtime GPU package.)";
        }
        return nullptr;
    }
}
