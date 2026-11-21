// YOLOv9 + SAHI (Slicing Aided Hyper Inference) with ncnn.
//
// The image is cut into overlapping tiles at the model's input size, every tile
// is letterboxed and run through YOLOv9, detections are mapped back to the full
// image and duplicates across tile borders are merged with class-aware NMS.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cpu.h"  // ncnn
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif

namespace fs = std::filesystem;

struct Detection {
    cv::Rect2f box;  // Global (x, y, width, height)
    float confidence;
    int class_id;
};

struct Region {
    int x, y, width, height;
};

struct Options {
    std::string param;                // model structure (.param)
    std::string bin;                  // model weights (.bin)
    std::vector<std::string> inputs;  // image files and/or directories
    std::string output;               // -o: output file (single image) or directory
    std::string csv_path;             // --csv: optional detections CSV
    std::string names_path;           // --names: class names, one per line (line 1 = class 0)
    std::string in_blob, out_blob;    // ncnn blob names (default: first input/output)
    std::string box = "auto";         // box format in the output: auto, xyxy or cxcywh
    std::string pad = "auto";         // letterbox placement: auto, center or topleft
    bool recursive = false;
    bool full = false;                // also run on the whole (letterboxed) image
    bool gpu = false;                 // Vulkan, if ncnn was built with it
    int size = 0;                     // model input size (0 = metadata.yaml or 640)
    int slice = 0;                    // tile size (0 = model input size)
    int workers = 0;                  // parallel tile workers (0 = auto: one per core, up to the tile count)
    int threads_per_worker = 0;       // ncnn threads per worker (0 = auto: cores / workers)
    int cores = 0;                    // thread budget (0 = physical cores)
    float overlap = 0.20f;
    float conf = 0.25f;
    float nms_iou = 0.45f;
    std::string merge = "nmm";        // final cross-tile merge: nmm (greedy merging) or nms
    std::string match = "ios";        // overlap metric for the final merge: ios or iou
    float match_thresh = 0.5f;
};

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

// Thread budget: physical cores. ncnn's kernels share each core's vector units,
// so hyperthreads add little and mostly cause contention.
static int physical_cores() {
    int n = ncnn::get_physical_big_cpu_count();
    if (n <= 0) n = ncnn::get_physical_cpu_count();
    if (n <= 0) n = static_cast<int>(std::thread::hardware_concurrency());
    return std::max(1, n);
}

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <model.param> <model.bin> <image|dir> [<image|dir> ...] [options]\n"
        << "\n"
        << "Loads the YOLOv9 ncnn model (.param, then .bin) once and runs SAHI on every image given.\n"
        << "Directories are scanned for .jpg/.jpeg/.png/.bmp/.tif/.tiff/.webp files (sorted by name).\n"
        << "\n"
        << "  -o, --output <path>    single image: output file (default: sahi_output.jpg)\n"
        << "                         several images: output directory (default: sahi_output/)\n"
        << "                         files are written as <name>_sahi.<ext>\n"
        << "      --csv <path>       also write all detections to a CSV file\n"
        << "  -n, --names <file>     class names, one per line: line 1 = class 0, line 2 = class 1, ...\n"
        << "                         (default: names from a metadata.yaml next to the .param, if present)\n"
        << "  -r, --recursive        scan directories recursively\n"
        << "  -s, --size <n>         model input size (default: imgsz from metadata.yaml, else 640)\n"
        << "      --slice <n>        tile size (default: model input size)\n"
        << "      --full             also detect on the whole image (finds large objects)\n"
        << "  -w, --workers <n>      tiles run in parallel, max (default: one per core, never more than tiles)\n"
        << "  -t, --threads <n>      ncnn threads per tile (default: cores / workers for each image)\n"
        << "  -j, --cores <n>        total thread budget (default: physical CPU cores)\n"
        << "      --gpu              use Vulkan (needs a build with -DSAHI_VULKAN=ON)\n"
        << "      --overlap <r>      tile overlap ratio (default: 0.20)\n"
        << "      --conf <c>         confidence threshold (default: 0.25)\n"
        << "      --iou <i>          NMS IoU threshold within a tile (default: 0.45)\n"
        << "      --merge <m>        merging across tiles: nmm = merge overlapping boxes into one (default),\n"
        << "                         nms = keep only the best box\n"
        << "      --match <m>        overlap metric for the merge: ios = intersection over the smaller box\n"
        << "                         (default; joins objects cut by a tile edge), iou = intersection over union\n"
        << "      --match-thresh <t> overlap above which boxes are merged (default: 0.5)\n"
        << "      --box <fmt>        box format in the model output: xyxy (LibreYOLO) or cxcywh\n"
        << "                         (default: from metadata.yaml, else detected from the output)\n"
        << "      --pad <mode>       letterbox placement: center or topleft (LibreYOLO checkpoints)\n"
        << "                         (default: letterbox_pad from metadata.yaml, else center)\n"
        << "      --in <blob>        input blob name (default: the model's first input, e.g. in0)\n"
        << "      --out <blob>       output blob name (default: the model's first output, e.g. out0)\n";
}

static bool parse_args(int argc, char** argv, Options& opt) {
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };
        const char* v = nullptr;
        if (a == "-h" || a == "--help") return false;
        else if (a == "-o" || a == "--output")    { if (!(v = next("--output")))  return false; opt.output = v; }
        else if (a == "--csv")                    { if (!(v = next("--csv")))     return false; opt.csv_path = v; }
        else if (a == "-n" || a == "--names")     { if (!(v = next("--names")))   return false; opt.names_path = v; }
        else if (a == "-r" || a == "--recursive") { opt.recursive = true; }
        else if (a == "-s" || a == "--size")      { if (!(v = next("--size")))    return false; opt.size = std::atoi(v); }
        else if (a == "--slice")                  { if (!(v = next("--slice")))   return false; opt.slice = std::atoi(v); }
        else if (a == "--full")                   { opt.full = true; }
        else if (a == "-w" || a == "--workers")   { if (!(v = next("--workers"))) return false; opt.workers = std::atoi(v); }
        else if (a == "-t" || a == "--threads")   { if (!(v = next("--threads"))) return false; opt.threads_per_worker = std::atoi(v); }
        else if (a == "-j" || a == "--cores")     { if (!(v = next("--cores")))   return false; opt.cores = std::atoi(v); }
        else if (a == "--gpu")                    { opt.gpu = true; }
        else if (a == "--overlap")                { if (!(v = next("--overlap"))) return false; opt.overlap = std::strtof(v, nullptr); }
        else if (a == "--conf")                   { if (!(v = next("--conf")))    return false; opt.conf = std::strtof(v, nullptr); }
        else if (a == "--iou")                    { if (!(v = next("--iou")))     return false; opt.nms_iou = std::strtof(v, nullptr); }
        else if (a == "--merge")                  { if (!(v = next("--merge")))   return false; opt.merge = v; }
        else if (a == "--match")                  { if (!(v = next("--match")))   return false; opt.match = v; }
        else if (a == "--match-thresh")           { if (!(v = next("--match-thresh"))) return false; opt.match_thresh = std::strtof(v, nullptr); }
        else if (a == "--merge")                  { if (!(v = next("--merge")))   return false; opt.merge = v; }
        else if (a == "--match")                  { if (!(v = next("--match")))   return false; opt.match = v; }
        else if (a == "--match-thresh")           { if (!(v = next("--match-thresh"))) return false; opt.match_thresh = std::strtof(v, nullptr); }
        else if (a == "--box")                    { if (!(v = next("--box")))     return false; opt.box = v; }
        else if (a == "--pad")                    { if (!(v = next("--pad")))     return false; opt.pad = v; }
        else if (a == "--in")                     { if (!(v = next("--in")))      return false; opt.in_blob = v; }
        else if (a == "--out")                    { if (!(v = next("--out")))     return false; opt.out_blob = v; }
        else if (!a.empty() && a[0] == '-')       { std::cerr << "Unknown option: " << a << "\n"; return false; }
        else positional.push_back(a);
    }
    if (opt.size != 0 && opt.size < 32) {
        std::cerr << "--size must be at least 32\n";
        return false;
    }
    if (opt.overlap < 0.0f || opt.overlap >= 1.0f) {
        std::cerr << "--overlap must be in [0, 1)\n";
        return false;
    }
    if (opt.box != "auto" && opt.box != "xyxy" && opt.box != "cxcywh") { std::cerr << "--box must be xyxy or cxcywh\n"; return false; }
    if (opt.pad != "auto" && opt.pad != "center" && opt.pad != "topleft") { std::cerr << "--pad must be center or topleft\n"; return false; }
    if (opt.merge != "nmm" && opt.merge != "nms") { std::cerr << "--merge must be nmm or nms\n"; return false; }
    if (opt.match != "ios" && opt.match != "iou") { std::cerr << "--match must be ios or iou\n"; return false; }
    // Fixed order: <model.param> <model.bin> <image|dir> ...
    if (positional.size() < 3) {
        std::cerr << "Expected <model.param> <model.bin> and at least one image or directory\n";
        return false;
    }
    auto has_ext = [](const std::string& f, const char* ext) {
        std::string e = fs::path(f).extension().string();
        std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return e == ext;
    };
    if (!has_ext(positional[0], ".param")) {
        std::cerr << "First argument must be the model's .param file (got " << positional[0] << ")\n";
        return false;
    }
    if (!has_ext(positional[1], ".bin")) {
        std::cerr << "Second argument must be the model's .bin file (got " << positional[1] << ")\n";
        return false;
    }
    opt.param = positional[0];
    opt.bin = positional[1];
    opt.inputs.assign(positional.begin() + 2, positional.end());
    return true;
}

static bool check_model_files(const fs::path& param, const fs::path& bin) {
    std::error_code ec;
    if (!fs::is_regular_file(param, ec)) { std::cerr << "Model file not found: " << param << "\n"; return false; }
    if (!fs::is_regular_file(bin, ec))   { std::cerr << "Weights file not found: " << bin << "\n"; return false; }
    return true;
}

static bool is_image_file(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    static const char* exts[] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"};
    for (const char* e : exts) if (ext == e) return true;
    return false;
}

// Expand the command-line inputs into a list of image files.
static std::vector<fs::path> collect_images(const std::vector<std::string>& inputs, bool recursive) {
    std::vector<fs::path> files;
    for (const auto& in : inputs) {
        std::error_code ec;
        fs::path p(in);
        if (fs::is_directory(p, ec)) {
            std::vector<fs::path> found;
            auto add = [&](const fs::directory_entry& e) {
                if (e.is_regular_file(ec) && is_image_file(e.path())) found.push_back(e.path());
            };
            if (recursive) {
                for (const auto& e : fs::recursive_directory_iterator(p, fs::directory_options::skip_permission_denied, ec)) add(e);
            } else {
                for (const auto& e : fs::directory_iterator(p, ec)) add(e);
            }
            std::sort(found.begin(), found.end());
            if (found.empty()) std::cerr << "Warning: no images found in " << p << "\n";
            files.insert(files.end(), found.begin(), found.end());
        } else if (fs::exists(p, ec)) {
            files.push_back(p);
        } else {
            std::cerr << "Warning: input not found, skipping: " << in << "\n";
        }
    }
    return files;
}

// Read class names: line N (1-based) is the name of class N-1. Surrounding
// whitespace and Windows line endings are stripped. Empty lines are kept, so
// line numbers stay aligned with class ids. A UTF-8 byte-order mark is ignored.
static bool load_class_names(const std::string& path, std::vector<std::string>& names) {
    std::ifstream f(path);
    if (!f) return false;
    names.clear();
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (first && line.size() >= 3 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
        first = false;
        const auto b = line.find_first_not_of(" \t\r\n");
        const auto e = line.find_last_not_of(" \t\r\n");
        names.push_back(b == std::string::npos ? std::string() : line.substr(b, e - b + 1));
    }
    while (!names.empty() && names.back().empty()) names.pop_back();
    return true;
}

// What a metadata.yaml next to the .param can tell us. Exporters such as
// LibreYOLO write one with the class names, input size and letterbox placement.
struct ModelMetadata {
    std::vector<std::string> names;
    int imgsz = 0;
    std::string letterbox_pad;  // "center" / "topleft" (LibreYOLO)
    std::string model_family;   // e.g. "yolo9", "yolo9_e2e" (LibreYOLO)
    bool libreyolo = false;     // written by LibreYOLO (boxes are xyxy)
};

// Minimal reader for that file: flat "key: value" lines plus the `names`
// mapping (0: person / '0': person) and `imgsz` (640, [640, 640] or a list).
static ModelMetadata read_model_metadata(const fs::path& path) {
    ModelMetadata md;
    std::ifstream f(path);
    if (!f) return md;
    auto trim = [](std::string t) {
        const auto b = t.find_first_not_of(" \t\r");
        const auto e = t.find_last_not_of(" \t\r");
        return b == std::string::npos ? std::string() : t.substr(b, e - b + 1);
    };
    auto unquote = [](std::string t) {
        if (t.size() >= 2 && (t.front() == '\'' || t.front() == '"') && t.back() == t.front()) t = t.substr(1, t.size() - 2);
        return t;
    };
    std::map<int, std::string> by_id;
    std::string section, line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (trim(line).empty() || trim(line)[0] == '#') continue;
        const bool indented = line[0] == ' ' || line[0] == '\t' || line[0] == '-';
        const std::string t = trim(line);
        if (!indented) {
            const auto colon = t.find(':');
            section = colon == std::string::npos ? "" : unquote(trim(t.substr(0, colon)));
            const std::string rest = colon == std::string::npos ? "" : unquote(trim(t.substr(colon + 1)));
            if (section == "imgsz" && !rest.empty()) {  // imgsz: 640 or imgsz: [640, 640]
                const auto d = rest.find_first_of("0123456789");
                if (d != std::string::npos) md.imgsz = std::atoi(rest.c_str() + d);
            } else if (section == "letterbox_pad") {
                md.letterbox_pad = rest;
            } else if (section == "model_family") {
                md.model_family = rest;
            } else if (section == "libreyolo_version") {
                md.libreyolo = true;
            }
            continue;
        }
        if (section == "imgsz" && t[0] == '-' && md.imgsz == 0) {  // block list: "- 640"
            md.imgsz = std::atoi(trim(t.substr(1)).c_str());
        } else if (section == "names") {
            const auto colon = t.find(':');
            if (colon == std::string::npos) continue;
            const std::string key = unquote(trim(t.substr(0, colon)));
            if (key.empty() || key.find_first_not_of("0123456789") != std::string::npos) continue;
            by_id[std::atoi(key.c_str())] = unquote(trim(t.substr(colon + 1)));
        }
    }
    if (!by_id.empty()) {
        md.names.assign(static_cast<size_t>(by_id.rbegin()->first) + 1, std::string());
        for (const auto& kv : by_id) md.names[kv.first] = kv.second;
    }
    return md;
}

static std::string class_label(const std::vector<std::string>& names, int id) {
    if (id >= 0 && static_cast<size_t>(id) < names.size() && !names[id].empty()) return names[id];
    return "class " + std::to_string(id);
}

static std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += c; }
    return out + "\"";
}

// ---------------------------------------------------------------------------
// SAHI
// ---------------------------------------------------------------------------

// Generate overlapping SAHI tile coordinates
static std::vector<Region> generate_slices(int img_w, int img_h, int slice_w, int slice_h, float overlap_ratio) {
    std::vector<Region> slices;
    int step_x = std::max(1, static_cast<int>(slice_w * (1.0f - overlap_ratio)));
    int step_y = std::max(1, static_cast<int>(slice_h * (1.0f - overlap_ratio)));

    for (int y = 0; y < img_h; y += step_y) {
        int h = std::min(slice_h, img_h - y);
        // Shift window back if we hit the bottom edge so slice keeps full native resolution
        int start_y = (h < slice_h && img_h >= slice_h) ? (img_h - slice_h) : y;
        h = std::min(slice_h, img_h - start_y);

        for (int x = 0; x < img_w; x += step_x) {
            int w = std::min(slice_w, img_w - x);
            // Shift window back if we hit the right edge
            int start_x = (w < slice_w && img_w >= slice_w) ? (img_w - slice_w) : x;
            w = std::min(slice_w, img_w - start_x);

            slices.push_back({start_x, start_y, w, h});

            if (start_x + w >= img_w) break;
        }
        if (start_y + h >= img_h) break;
    }
    return slices;
}

// Class-aware NMS within one tile: classes don't suppress each other.
// Output order is deterministic (sorted by class id).
static std::vector<Detection> apply_class_aware_nms(const std::vector<Detection>& raw_dets,
                                                    float conf_threshold, float nms_iou_threshold) {
    std::vector<Detection> final_dets;
    if (raw_dets.empty()) return final_dets;

    std::map<int, std::vector<size_t>> class_indices;
    for (size_t i = 0; i < raw_dets.size(); ++i) class_indices[raw_dets[i].class_id].push_back(i);

    std::vector<const std::vector<size_t>*> groups;
    groups.reserve(class_indices.size());
    for (const auto& kv : class_indices) groups.push_back(&kv.second);

    std::vector<std::vector<Detection>> kept(groups.size());

    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        const auto& indices = *groups[g];
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        boxes.reserve(indices.size());
        scores.reserve(indices.size());
        for (size_t idx : indices) {
            const auto& b = raw_dets[idx].box;
            boxes.emplace_back(b.x, b.y, b.width, b.height);
            scores.push_back(raw_dets[idx].confidence);
        }
        std::vector<int> keep_indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_iou_threshold, keep_indices);
        kept[g].reserve(keep_indices.size());
        for (int k : keep_indices) kept[g].push_back(raw_dets[indices[k]]);
    }

    for (auto& v : kept) final_dets.insert(final_dets.end(), v.begin(), v.end());
    return final_dets;
}

// Final cross-tile merge, as in the SAHI library (GREEDYNMM / NMS with an IOS
// or IOU match metric). Objects cut by a tile edge show up as a full box from
// one tile plus a fragment from the neighbour; a fragment lies mostly inside
// the full box, so intersection-over-smaller-area (IOS) catches it where IoU
// does not. With NMM, matched boxes are merged into their union (two halves
// of one object become one box) and keep the best score; with NMS they are
// dropped. Per class, highest score first; classes run in parallel; output
// order is deterministic.
static std::vector<Detection> merge_detections(const std::vector<Detection>& dets, const Options& opt) {
    const bool use_ios = opt.match == "ios";
    const bool merge_boxes = opt.merge == "nmm";
    auto overlap = [&](const cv::Rect2f& a, const cv::Rect2f& b) {
        const float ix = std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x);
        const float iy = std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y);
        if (ix <= 0.0f || iy <= 0.0f) return 0.0f;
        const float inter = ix * iy, area_a = a.width * a.height, area_b = b.width * b.height;
        const float denom = use_ios ? std::min(area_a, area_b) : (area_a + area_b - inter);
        return denom > 0.0f ? inter / denom : 0.0f;
    };

    std::map<int, std::vector<size_t>> by_class;
    for (size_t i = 0; i < dets.size(); ++i) by_class[dets[i].class_id].push_back(i);
    std::vector<std::vector<size_t>> groups;
    for (auto& kv : by_class) groups.push_back(std::move(kv.second));
    std::vector<std::vector<Detection>> kept(groups.size());

    // Worth threads only when there are many boxes; usually this takes microseconds.
    #pragma omp parallel for schedule(dynamic, 1) if (groups.size() > 1 && dets.size() > 4000)
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        std::vector<size_t> order = groups[g];
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return dets[a].confidence > dets[b].confidence; });
        std::vector<char> used(order.size(), 0);
        for (size_t i = 0; i < order.size(); ++i) {
            if (used[i]) continue;
            Detection keep = dets[order[i]];
            const cv::Rect2f ref = keep.box;  // match against the original box, as SAHI does
            float x1 = ref.x, y1 = ref.y, x2 = ref.x + ref.width, y2 = ref.y + ref.height;
            for (size_t j = i + 1; j < order.size(); ++j) {
                if (used[j]) continue;
                const cv::Rect2f& b = dets[order[j]].box;
                if (overlap(ref, b) <= opt.match_thresh) continue;
                used[j] = 1;
                if (merge_boxes) {
                    x1 = std::min(x1, b.x); y1 = std::min(y1, b.y);
                    x2 = std::max(x2, b.x + b.width); y2 = std::max(y2, b.y + b.height);
                }
            }
            keep.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
            kept[g].push_back(keep);
        }
    }

    std::vector<Detection> out;
    for (auto& v : kept) out.insert(out.end(), v.begin(), v.end());
    return out;
}

struct ModelInfo {
    std::string in_blob;
    std::string out_blob;
    int size = 640;
    bool xyxy = false;     // output boxes are x1,y1,x2,y2 (else cx,cy,w,h)
    bool topleft = false;  // letterbox image in the top-left corner (else centred)
};

// View the head output as (features x anchors). ncnn gives [4 + nc, N] as
// h = 4 + nc, w = N; the transposed layout (anchors x features) is detected
// automatically, since there are always far more anchors than features.
struct HeadView {
    ncnn::Mat out;
    bool feature_rows = true;
    int n_feat = 0, n_anchor = 0;
    float at(int f, int a) const { return feature_rows ? out.row(f)[a] : out.row(a)[f]; }
};

static bool make_head_view(const ncnn::Mat& out_raw, HeadView& v) {
    v.out = out_raw;
    if (v.out.dims == 3 && v.out.c == 1) v.out = v.out.channel(0);
    if (v.out.dims != 2) {
        std::fprintf(stderr, "Unexpected output shape (dims=%d, w=%d, h=%d, c=%d); expected 2D (4+classes) x anchors\n",
                     v.out.dims, v.out.w, v.out.h, v.out.c);
        return false;
    }
    v.feature_rows = v.out.h <= v.out.w;
    v.n_feat = v.feature_rows ? v.out.h : v.out.w;
    v.n_anchor = v.feature_rows ? v.out.w : v.out.h;
    if (v.n_feat < 5) {
        std::fprintf(stderr, "Output has %d features per anchor; expected 4 box values + class scores\n", v.n_feat);
        return false;
    }
    return true;
}

// Corner boxes always satisfy x2 >= x1 and y2 >= y1. With centre boxes, many
// anchors have a centre further from the origin than the box is wide, so the
// same test fails somewhere among the thousands of anchors.
static bool looks_like_xyxy(const HeadView& v) {
    for (int a = 0; a < v.n_anchor; ++a) {
        if (v.at(2, a) < v.at(0, a) || v.at(3, a) < v.at(1, a)) return false;
    }
    return true;
}

// Decode a YOLOv9-style head: (4 + num_classes) x num_anchors. Rows are the
// box (x1,y1,x2,y2 or cx,cy,w,h, in letterboxed input pixels), then one
// sigmoid-activated score per class.
static bool decode_yolo(const ncnn::Mat& out_raw, bool xyxy, float conf, float scale, int pad_l, int pad_t,
                        const Region& r, std::vector<Detection>& dets) {
    HeadView v;
    if (!make_head_view(out_raw, v)) return false;
    const int n_cls = v.n_feat - 4;
    auto at = [&](int f, int a) { return v.at(f, a); };

    // Best class per anchor. In the usual (features x anchors) layout each class
    // is one contiguous row, so scan row by row (sequential memory, vectorizes)
    // instead of jumping a whole row per class for every anchor.
    std::vector<float> best_score(v.n_anchor);
    std::vector<int> best_cls(v.n_anchor, 0);
    if (v.feature_rows) {
        const float* row = v.out.row(4);
        std::copy(row, row + v.n_anchor, best_score.begin());
        for (int c = 1; c < n_cls; ++c) {
            row = v.out.row(4 + c);
            for (int a = 0; a < v.n_anchor; ++a) {
                if (row[a] > best_score[a]) { best_score[a] = row[a]; best_cls[a] = c; }
            }
        }
    } else {
        for (int a = 0; a < v.n_anchor; ++a) {
            const float* p = v.out.row(a) + 4;
            int best = 0;
            for (int c = 1; c < n_cls; ++c) if (p[c] > p[best]) best = c;
            best_score[a] = p[best];
            best_cls[a] = best;
        }
    }

    for (int a = 0; a < v.n_anchor; ++a) {
        if (best_score[a] < conf) continue;
        float bx1, by1, bx2, by2;
        if (xyxy) {
            bx1 = at(0, a); by1 = at(1, a); bx2 = at(2, a); by2 = at(3, a);
        } else {
            const float cx = at(0, a), cy = at(1, a), w = at(2, a), h = at(3, a);
            bx1 = cx - 0.5f * w; by1 = cy - 0.5f * h; bx2 = cx + 0.5f * w; by2 = cy + 0.5f * h;
        }
        // Undo the letterbox, clamp to the region, then shift to image coordinates
        float x1 = std::clamp((bx1 - pad_l) / scale, 0.0f, static_cast<float>(r.width));
        float y1 = std::clamp((by1 - pad_t) / scale, 0.0f, static_cast<float>(r.height));
        float x2 = std::clamp((bx2 - pad_l) / scale, 0.0f, static_cast<float>(r.width));
        float y2 = std::clamp((by2 - pad_t) / scale, 0.0f, static_cast<float>(r.height));
        if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;

        Detection d;
        d.box = cv::Rect2f(x1 + r.x, y1 + r.y, x2 - x1, y2 - y1);
        d.confidence = best_score[a];
        d.class_id = best_cls[a];
        dets.push_back(d);
    }
    return true;
}

// Letterbox one region of the image to size x size (gray 114 padding, centred or top-left),
// run YOLOv9 and return its detections (after per-region NMS) in image coordinates.
// Uses its own Extractor, so it is safe to call concurrently on the same Net.
// Per-worker memory pools, reused for every tile and image. By default all
// extractors share the network's one locked pool, so concurrent tiles would
// queue on its mutex; this is the setup ncnn's own benchmark uses.
struct WorkerPools {
    ncnn::UnlockedPoolAllocator blob;
    ncnn::PoolAllocator workspace;
    WorkerPools() {
        blob.set_size_compare_ratio(0.f);
        workspace.set_size_compare_ratio(0.f);
    }
};

static bool detect_region(const ncnn::Net& net, const ModelInfo& mi, const cv::Mat& bgr,
                          const Region& r, float conf, float nms_iou, WorkerPools* pools,
                          std::vector<Detection>& out) {
    const int S = mi.size;
    const float scale = std::min(static_cast<float>(S) / r.width, static_cast<float>(S) / r.height);
    const int nw = std::min(S, std::max(1, static_cast<int>(r.width * scale)));
    const int nh = std::min(S, std::max(1, static_cast<int>(r.height * scale)));
    const int pad_l = mi.topleft ? 0 : (S - nw) / 2;
    const int pad_t = mi.topleft ? 0 : (S - nh) / 2;

    // Read the region straight from the BGR image (row stride = full image row), convert to RGB
    const unsigned char* src = bgr.ptr<unsigned char>(r.y) + static_cast<size_t>(r.x) * 3;
    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(src, ncnn::Mat::PIXEL_BGR2RGB, r.width, r.height,
                                                      static_cast<int>(bgr.step[0]), nw, nh);
    ncnn::Mat in;
    ncnn::copy_make_border(resized, in, pad_t, S - nh - pad_t, pad_l, S - nw - pad_l,
                           ncnn::BORDER_CONSTANT, 114.0f);
    const float norm[3] = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f};
    in.substract_mean_normalize(nullptr, norm);

    ncnn::Extractor ex = net.create_extractor();
    if (pools) {
        ex.set_blob_allocator(&pools->blob);
        ex.set_workspace_allocator(&pools->workspace);
    }
    if (ex.input(mi.in_blob.c_str(), in) != 0) {
        std::fprintf(stderr, "Cannot feed input blob '%s'\n", mi.in_blob.c_str());
        return false;
    }
    ncnn::Mat raw;
    if (ex.extract(mi.out_blob.c_str(), raw) != 0) {
        std::fprintf(stderr, "Inference failed on region (%d,%d %dx%d)\n", r.x, r.y, r.width, r.height);
        return false;
    }

    std::vector<Detection> dets;
    if (!decode_yolo(raw, mi.xyxy, conf, scale, pad_l, pad_t, r, dets)) return false;

    // The head predicts many overlapping boxes per object: suppress within the region first
    std::vector<Detection> kept = apply_class_aware_nms(dets, conf, nms_iou);
    out.insert(out.end(), kept.begin(), kept.end());
    return true;
}

// How the thread budget is spread over the tiles. ncnn's convolution layers
// fix their thread count when the model is loaded (weights are pre-arranged
// for it), so the split is chosen once at startup, not per image.
struct ThreadPlan {
    int workers = 1;   // tiles in flight at once (outer OpenMP)
    int threads = 1;   // ncnn threads per tile (inner OpenMP)
};

// Default: one tile per core with ncnn single-threaded. SAHI produces many
// equal tiles, and this scales almost linearly with no nested OpenMP teams and
// no idle spin-waiting; ncnn's per-layer threading scales worse on one small
// 640x640 tile. -w / -t override.
static ThreadPlan plan_threads(int cores, const Options& opt) {
    ThreadPlan p;
    p.workers = opt.workers > 0 ? opt.workers : cores;
    p.threads = opt.threads_per_worker > 0 ? opt.threads_per_worker : std::max(1, cores / p.workers);
#ifndef _OPENMP
    p.workers = 1;
#endif
    return p;
}

// Run SAHI on one image; returns merged detections. Results go into per-region
// slots and are concatenated in order, so the output does not depend on thread
// scheduling. `wide` (all cores in one extraction) is used when the image is a
// single region and the tile plan would leave cores idle; it is null otherwise.
static std::vector<Detection> run_sahi(const ncnn::Net& net, const ncnn::Net* wide, const ModelInfo& mi,
                                       const cv::Mat& bgr, int slice_size, const ThreadPlan& plan,
                                       std::vector<std::unique_ptr<WorkerPools>>& pools, bool use_pools,
                                       const Options& opt, size_t& n_regions, size_t& n_raw, int& failed,
                                       ThreadPlan& used) {
    std::vector<Region> regions = generate_slices(bgr.cols, bgr.rows, slice_size, slice_size, opt.overlap);
    // Optional full-image pass (SAHI's "standard prediction"): skip when one tile already covers everything
    if (opt.full && !(regions.size() == 1 && regions[0].width == bgr.cols && regions[0].height == bgr.rows)) {
        regions.push_back({0, 0, bgr.cols, bgr.rows});
    }
    n_regions = regions.size();

    std::vector<std::vector<Detection>> per_region(regions.size());
    int n_failed = 0;

    if (regions.size() == 1 && wide) {
        used = {1, wide->opt.num_threads};
        if (!detect_region(*wide, mi, bgr, regions[0], opt.conf, opt.nms_iou,
                           use_pools ? pools[0].get() : nullptr, per_region[0])) ++n_failed;
    } else {
        const int n_workers = std::max(1, std::min<int>(plan.workers, static_cast<int>(regions.size())));
        used = {n_workers, plan.threads};
        (void)n_workers;  // unused without OpenMP

        #pragma omp parallel for num_threads(n_workers) schedule(dynamic, 1) reduction(+ : n_failed)
        for (int i = 0; i < static_cast<int>(regions.size()); ++i) {
#ifdef _OPENMP
            WorkerPools* wp = use_pools ? pools[omp_get_thread_num()].get() : nullptr;
#else
            WorkerPools* wp = use_pools ? pools[0].get() : nullptr;
#endif
            if (!detect_region(net, mi, bgr, regions[i], opt.conf, opt.nms_iou, wp, per_region[i])) ++n_failed;
        }
    }
    failed = n_failed;

    std::vector<Detection> all;
    for (auto& v : per_region) all.insert(all.end(), v.begin(), v.end());
    n_raw = all.size();

    // Merge duplicates and fragments across tile borders
    return merge_detections(all, opt);
}

// Draw boxes with "<class name> <score>" labels on a green background
static void draw_detections(cv::Mat& bgr_image, const std::vector<Detection>& dets,
                            const std::vector<std::string>& names) {
    const cv::Scalar GREEN(0, 255, 0);  // BGR
    const cv::Scalar BLACK(0, 0, 0);
    const int FONT = cv::FONT_HERSHEY_SIMPLEX;
    const double FONT_SCALE = 0.5;
    const int FONT_THICKNESS = 1;
    const int PAD = 3;

    for (const auto& det : dets) {
        cv::Rect box(static_cast<int>(std::round(det.box.x)), static_cast<int>(std::round(det.box.y)),
                     static_cast<int>(std::round(det.box.width)), static_cast<int>(std::round(det.box.height)));
        cv::rectangle(bgr_image, box, GREEN, 2);

        char score[16];
        std::snprintf(score, sizeof(score), "%.2f", det.confidence);
        const std::string label = class_label(names, det.class_id) + " " + score;

        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS, &baseline);
        int bg_w = ts.width + 2 * PAD;
        int bg_h = ts.height + baseline + 2 * PAD;

        // Place the label above the box; if there's no room, put it just inside the top edge.
        int bg_x = std::clamp(box.x, 0, std::max(0, bgr_image.cols - bg_w));
        int bg_y = box.y - bg_h;
        if (bg_y < 0) bg_y = std::min(std::max(box.y, 0), std::max(0, bgr_image.rows - bg_h));

        cv::rectangle(bgr_image, cv::Rect(bg_x, bg_y, bg_w, bg_h), GREEN, cv::FILLED);
        cv::putText(bgr_image, label, cv::Point(bg_x + PAD, bg_y + PAD + ts.height),
                    FONT, FONT_SCALE, BLACK, FONT_THICKNESS, cv::LINE_AA);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const fs::path param_path(opt.param), bin_path(opt.bin);
    if (!check_model_files(param_path, bin_path)) return 1;

    // Class names, input size, box format and letterbox placement: command-line
    // options win; otherwise a metadata.yaml next to the .param; otherwise defaults.
    std::vector<std::string> class_names;
    const fs::path meta_path = param_path.parent_path() / "metadata.yaml";
    const ModelMetadata md = read_model_metadata(meta_path);
    const std::vector<std::string>& meta_names = md.names;
    const int meta_imgsz = md.imgsz;
    std::string names_source;
    if (!opt.names_path.empty()) {
        if (!load_class_names(opt.names_path, class_names)) {
            std::cerr << "Cannot read class names file: " << opt.names_path << "\n";
            return 1;
        }
        names_source = opt.names_path;
    } else if (!meta_names.empty()) {
        class_names = meta_names;
        names_source = meta_path.string();
    }
    if (opt.size == 0) opt.size = meta_imgsz >= 32 ? meta_imgsz : 640;

    std::vector<fs::path> images = collect_images(opt.inputs, opt.recursive);
    if (images.empty()) {
        std::cerr << "No input images.\n";
        return 1;
    }

    // Output naming: one image -> -o is a file; several -> -o is a directory.
    const bool single = images.size() == 1;
    fs::path out_file, out_dir;
    if (single) {
        out_file = opt.output.empty() ? fs::path("sahi_output.jpg") : fs::path(opt.output);
        if (fs::is_directory(out_file)) out_file /= images[0].stem().string() + "_sahi" + images[0].extension().string();
    } else {
        out_dir = opt.output.empty() ? fs::path("sahi_output") : fs::path(opt.output);
        std::error_code ec;
        fs::create_directories(out_dir, ec);
        if (ec) {
            std::cerr << "Cannot create output directory " << out_dir << ": " << ec.message() << "\n";
            return 1;
        }
    }
    // Several inputs can share a file name (e.g. in different subdirectories with -r):
    // add _2, _3, ... so outputs never overwrite each other.
    std::map<std::string, int> used_names;
    auto output_for = [&](const fs::path& in) -> fs::path {
        if (single) return out_file;
        std::string base = in.stem().string() + "_sahi";
        std::string ext = in.extension().string();
        int& n = used_names[base + ext];
        std::string name = (++n == 1) ? base + ext : base + "_" + std::to_string(n) + ext;
        return out_dir / name;
    };

    // ---- Load the model once. One Net is shared by all workers: each tile gets
    //      its own Extractor, which ncnn allows concurrently. ----
    const int cores = opt.cores > 0 ? opt.cores : physical_cores();
    const ThreadPlan plan = plan_threads(cores, opt);

    bool use_gpu = false;
    std::string device = "CPU";
    if (opt.gpu) {
#if NCNN_VULKAN
        if (ncnn::get_gpu_count() > 0) {
            use_gpu = true;
            device = std::string("Vulkan (") + ncnn::get_gpu_info(0).device_name() + ")";
        } else {
            std::cerr << "Warning: no Vulkan device found; using CPU\n";
        }
#else
        std::cerr << "Warning: built without Vulkan (configure with -DSAHI_VULKAN=ON); using CPU\n";
#endif
    }

    auto load_net = [&](ncnn::Net& n, int threads, bool concurrent) {
        n.opt.num_threads = threads;
        n.opt.lightmode = true;  // free intermediate blobs as soon as possible
        n.opt.use_vulkan_compute = use_gpu;
        // With several tiles in flight, idle ncnn threads must not spin-wait
        // for work (default 20 ms): they would burn cores other tiles need.
        n.opt.openmp_blocktime = concurrent ? 0 : 20;
        if (n.load_param(param_path.string().c_str()) != 0) {
            std::cerr << "Failed to load " << param_path << "\n";
            return false;
        }
        if (n.load_model(bin_path.string().c_str()) != 0) {
            std::cerr << "Failed to load " << bin_path << "\n";
            return false;
        }
        return true;
    };

    ncnn::Net net;
    if (!load_net(net, plan.threads, plan.workers > 1)) return 1;

#ifdef _OPENMP
    // Nested OpenMP only when both levels have threads to use.
    omp_set_max_active_levels(plan.workers > 1 && plan.threads > 1 ? 2 : 1);
#endif

    // Images that fit in one tile get no tile parallelism; when the tile plan
    // leaves cores idle, those run on a second copy of the model loaded with
    // all cores. Loaded only when such an image first shows up.
    std::unique_ptr<ncnn::Net> wide_net;
    bool wide_tried = false;
    auto get_wide = [&]() -> const ncnn::Net* {
        if (use_gpu || plan.threads >= cores) return nullptr;
        if (!wide_tried) {
            wide_tried = true;
            auto n = std::make_unique<ncnn::Net>();
            if (load_net(*n, cores, false)) wide_net = std::move(n);
        }
        return wide_net.get();
    };

    // One memory pool pair per worker (CPU only; Vulkan uses its own allocators)
    const bool use_pools = !use_gpu;
    std::vector<std::unique_ptr<WorkerPools>> pools;
    for (int w = 0; w < std::max(1, plan.workers); ++w) pools.push_back(std::make_unique<WorkerPools>());

    ModelInfo mi;
    mi.size = opt.size;
    std::string pad_source = "default";
    if (opt.pad != "auto") {
        mi.topleft = opt.pad == "topleft";
        pad_source = "--pad";
    } else if (md.letterbox_pad == "topleft" || md.letterbox_pad == "center" || md.letterbox_pad == "centre") {
        mi.topleft = md.letterbox_pad == "topleft";
        pad_source = "metadata.yaml";
    }
    mi.in_blob = opt.in_blob;
    mi.out_blob = opt.out_blob;
    if (mi.in_blob.empty()) {
        if (net.input_names().empty()) { std::cerr << "Model has no input blob; pass --in\n"; return 1; }
        mi.in_blob = net.input_names()[0];
    }
    if (mi.out_blob.empty()) {
        if (net.output_names().empty()) { std::cerr << "Model has no output blob; pass --out\n"; return 1; }
        mi.out_blob = net.output_names()[0];
    }
    const int slice_size = opt.slice > 0 ? opt.slice : opt.size;

    // Check the blob names and output layout once, on a blank input, so a wrong
    // --in/--out or an unsupported model fails here instead of on every tile.
    // The same output tells us the box format when nothing else does.
    std::string box_source;
    {
        ncnn::Mat probe(mi.size, mi.size, 3);
        probe.fill(0.5f);
        ncnn::Extractor ex = net.create_extractor();
        ncnn::Mat out;
        if (ex.input(mi.in_blob.c_str(), probe) != 0 || ex.extract(mi.out_blob.c_str(), out) != 0) {
            std::cerr << "Cannot run the model with input '" << mi.in_blob << "' and output '" << mi.out_blob
                      << "'. Model inputs:";
            for (const char* n : net.input_names()) std::cerr << ' ' << n;
            std::cerr << " | outputs:";
            for (const char* n : net.output_names()) std::cerr << ' ' << n;
            std::cerr << "\nPass the right names with --in / --out.\n";
            return 1;
        }
        HeadView v;
        if (!make_head_view(out, v)) return 1;
        if (opt.box != "auto") {
            mi.xyxy = opt.box == "xyxy";
            box_source = "--box";
        } else if (md.libreyolo && md.model_family.rfind("yolo9", 0) == 0) {
            mi.xyxy = true;  // LibreYOLO's YOLOv9 head exports xyxy boxes
            box_source = "metadata.yaml";
        } else {
            mi.xyxy = looks_like_xyxy(v);
            box_source = "detected";
        }
    }

    const std::string plan_desc = std::to_string(cores) + " cores: " + std::to_string(plan.workers) +
                                  " tile(s) in parallel x " + std::to_string(plan.threads) + " thread(s)";

    const double init_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    std::cout << "Model: " << param_path.string() << " + " << bin_path.filename().string() << " (" << mi.in_blob << " -> " << mi.out_blob
              << ", input " << mi.size << "x" << mi.size << ", boxes " << (mi.xyxy ? "xyxy" : "cxcywh")
              << " (" << box_source << "), pad " << (mi.topleft ? "topleft" : "center") << " (" << pad_source
              << ")) | " << device << " | " << plan_desc
#ifndef _OPENMP
              << " [built without OpenMP: tiles run sequentially]"
#endif
              << "\n"
              << "Tiles: " << slice_size << "x" << slice_size << ", overlap " << opt.overlap
              << (opt.full ? " + full image" : "") << " | merge " << opt.merge << "/" << opt.match
              << " > " << opt.match_thresh << " | " << images.size() << " image(s)";
    if (!names_source.empty()) std::cout << " | " << class_names.size() << " class names from " << names_source;
    std::cout << " | init " << static_cast<int>(init_ms) << " ms\n";

    std::ofstream csv;
    if (!opt.csv_path.empty()) {
        csv.open(opt.csv_path);
        if (!csv) {
            std::cerr << "Cannot open CSV file: " << opt.csv_path << "\n";
            return 1;
        }
        csv << "image,class_id,class_name,score,x1,y1,x2,y2\n";
    }

    // ---- Process images; decode the next one in the background meanwhile ----
    size_t ok_images = 0, failed_images = 0, total_dets = 0, beyond_names = 0;
    double total_infer_ms = 0.0;
    auto load = [](const fs::path& p) { return cv::imread(p.string(), cv::IMREAD_COLOR); };
    std::future<cv::Mat> next = std::async(std::launch::async, load, images[0]);
    std::future<bool> pending_save;
    size_t unsaved = 0;

    for (size_t idx = 0; idx < images.size(); ++idx) {
        const fs::path& path = images[idx];
        cv::Mat bgr = next.get();
        if (idx + 1 < images.size()) next = std::async(std::launch::async, load, images[idx + 1]);

        std::cout << "[" << (idx + 1) << "/" << images.size() << "] " << path.string() << ": ";
        if (bgr.empty()) {
            std::cout << "failed to load, skipped\n";
            ++failed_images;
            continue;
        }

        size_t n_regions = 0, n_raw = 0;
        int failed = 0;
        ThreadPlan used;
        const bool one_region = generate_slices(bgr.cols, bgr.rows, slice_size, slice_size, opt.overlap).size() == 1;
        const ncnn::Net* wide = one_region ? get_wide() : nullptr;  // loads on first use: not timed
        const auto ti = clock::now();
        std::vector<Detection> merged = run_sahi(net, wide, mi, bgr, slice_size, plan, pools, use_pools, opt,
                                                 n_regions, n_raw, failed, used);
        const double infer_ms = std::chrono::duration<double, std::milli>(clock::now() - ti).count();

        if (!class_names.empty()) {
            for (const auto& d : merged) {
                if (static_cast<size_t>(d.class_id) >= class_names.size()) ++beyond_names;
            }
        }

        // Draw and encode the output in the background while the next image is
        // processed; the previous save is collected first (one in flight at a time).
        if (pending_save.valid() && !pending_save.get()) ++unsaved;
        const fs::path out = output_for(path);
        pending_save = std::async(std::launch::async,
                                  [bgr, merged, out, &class_names]() mutable {
                                      draw_detections(bgr, merged, class_names);
                                      try {
                                          if (cv::imwrite(out.string(), bgr)) return true;
                                      } catch (const cv::Exception& e) {
                                          std::cerr << "imwrite error: " << e.what() << "\n";
                                      }
                                      std::cerr << "Could not save " << out.string() << "\n";
                                      return false;
                                  });

        if (csv.is_open()) {
            for (const auto& d : merged) {
                csv << csv_escape(path.string()) << ',' << d.class_id << ','
                    << csv_escape(class_label(class_names, d.class_id)) << ','
                    << d.confidence << ',' << d.box.x << ',' << d.box.y << ','
                    << (d.box.x + d.box.width) << ',' << (d.box.y + d.box.height) << '\n';
            }
        }

        std::printf("%zu regions (%d x %d thr), %zu raw -> %zu dets, %.1f ms%s\n",
                    n_regions, used.workers, used.threads, n_raw, merged.size(), infer_ms,
                    failed ? (" | " + std::to_string(failed) + " region(s) failed").c_str() : "");
        std::fflush(stdout);

        if (failed == static_cast<int>(n_regions)) ++failed_images; else ++ok_images;
        total_dets += merged.size();
        total_infer_ms += infer_ms;
    }

    if (pending_save.valid() && !pending_save.get()) ++unsaved;
    if (unsaved > 0) {
        failed_images += unsaved;
        ok_images -= std::min(ok_images, unsaved);
    }

    const double total_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    std::printf("Done: %zu ok, %zu failed, %zu detections | inference %.1f s (%.1f ms/image) | total %.1f s\n",
                ok_images, failed_images, total_dets, total_infer_ms / 1000.0,
                total_infer_ms / images.size(), total_ms / 1000.0);
    if (beyond_names > 0) {
        std::cout << "Note: " << beyond_names << " detection(s) had a class id beyond the "
                  << class_names.size() << " names in " << names_source << " (labelled \"class <id>\")\n";
    }
    std::cout << "Output: " << (single ? out_file.string() : out_dir.string())
              << (opt.csv_path.empty() ? "" : " | CSV: " + opt.csv_path) << "\n";
    return failed_images == 0 ? 0 : 1;
}
