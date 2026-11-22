// YOLOv9 + SAHI (Slicing Aided Hyper Inference) with MNN (CUDA, Vulkan or CPU).
//
// The image is cut into overlapping slices at the model's input size (and,
// with --full, the whole image is added as one more region). Every region is
// letterboxed with YOLOv9's preprocessing (RGB, 0..1, OpenCV-style bilinear
// resize, gray padding, centred unless the model says top-left) and run
// through YOLOv9; detections are mapped back to
// the image, and duplicates and fragments across slice borders are merged as
// in the SAHI library (greedy NMM with IOS).
//
// The model is given as .onnx (converted once to .mnn with MNN's converter,
// next to the .onnx) or directly as .mnn.

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

#include "detector.hpp"
#include "region_detector.hpp"
#include "onnx_meta.hpp"
#include "region_ops.hpp"

namespace fs = std::filesystem;

struct Options {
    std::string model;                // .onnx or .mnn
    std::vector<std::string> inputs;  // image files and/or directories
    std::string output;               // -o: output file (single image) or directory
    std::string csv_path;             // --csv: optional detections CSV
    std::string names_path;           // --names: class names, one per line (line 1 = class 0)
    std::string backend = "auto";     // auto or one of available_backends()
    std::string precision = "normal"; // normal, high, low
    std::string preprocess = "auto";  // auto (metadata, else yolo), yolo, rtmdet
    std::string pad = "auto";         // auto (metadata, else per preprocessing), center, topleft
    std::string box = "auto";         // auto, xyxy, cxcywh
    std::string in_name, out_name;
    bool recursive = false;
    bool full = false;                // also run on the whole (letterboxed) image
    bool fp16 = false;                // convert with fp16 weights
    bool reconvert = false;           // convert again even if the .mnn is up to date
    int size = 0;                     // model input size (0 = metadata, input shape, else 640)
    int slice = 0;                    // slice size (0 = model input size)
    int batch = 8;                    // regions per run
    int threads = 0;                  // CPU threads (0 = all cores)
    float overlap = 0.20f;
    float conf = 0.25f;
    float nms_iou = 0.45f;
    std::string merge = "nmm";        // final cross-slice merge: nmm (greedy merging) or nms
    std::string match = "ios";        // overlap metric for the final merge: ios or iou
    float match_thresh = 0.5f;
};

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

static std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string s;
    for (const auto& x : v) s += (s.empty() ? "" : sep) + x;
    return s;
}

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <model.onnx|model.mnn> <image|dir> [<image|dir> ...] [options]\n"
        << "\n"
        << "Loads the YOLOv9 model once and runs SAHI on every image given. An .onnx model is\n"
        << "converted to .mnn (next to it) on first use. Directories are scanned for\n"
        << ".jpg/.jpeg/.png/.bmp/.tif/.tiff/.webp files (sorted by name).\n"
        << "\n"
        << "  -o, --output <path>    single image: output file (default: output.jpg)\n"
        << "                         several images: output directory (default: output/)\n"
        << "                         files are written as <name>_det.<ext>\n"
        << "      --csv <path>       also write all detections to a CSV file\n"
        << "  -n, --names <file>     class names, one per line: line 1 = class 0, line 2 = class 1, ...\n"
        << "                         (default: names stored in the ONNX model, if present)\n"
        << "  -r, --recursive        scan directories recursively\n"
        << "\n"
        << "  -d, --backend <b>      " << join(available_backends(), ", ") << " or auto (default: auto =\n"
        << "                         the first available of those, in that order)\n"
        << "      --precision <p>    normal, high or low (fp16 where supported) (default: normal)\n"
        << "  -b, --batch <n>        regions per inference run (default: 8; 1 for fixed-batch models)\n"
        << "  -t, --threads <n>      CPU backend threads and preprocessing (default: all cores)\n"
        << "      --fp16             convert the .onnx with fp16 weights (smaller, faster on GPU)\n"
        << "      --reconvert        convert the .onnx again even if the .mnn is up to date\n"
        << "\n"
        << "  -s, --size <n>         model input size (default: from the model, else 640)\n"
        << "      --slice <n>        slice size (default: model input size)\n"
        << "      --overlap <r>      slice overlap ratio (default: 0.20)\n"
        << "      --full             also detect on the whole image (finds large objects)\n"
        << "      --conf <c>         confidence threshold (default: 0.25)\n"
        << "      --iou <i>          NMS IoU threshold within a slice (default: 0.45)\n"
        << "      --merge <m>        merging across slices: nmm = merge overlapping boxes into one (default),\n"
        << "                         nms = keep only the best box\n"
        << "      --match <m>        overlap metric for the merge: ios = intersection over the smaller box\n"
        << "                         (default; joins objects cut by a slice edge), iou = intersection over union\n"
        << "      --match-thresh <t> overlap above which boxes are merged (default: 0.5)\n"
        << "\n"
        << "      --preprocess <p>   yolo (RGB, 0..1, OpenCV-style resize) or rtmdet (BGR, mmdet mean/std,\n"
        << "                         Pillow-style resize) (default: from the model's metadata, else yolo)\n"
        << "      --pad <mode>       letterbox placement: center or topleft (LibreYOLO-trained checkpoints)\n"
        << "                         (default: letterbox_pad from the model's metadata, else center)\n"
        << "      --box <fmt>        box format in the model output: xyxy or cxcywh\n"
        << "                         (default: from the model's metadata, else detected from the output)\n"
        << "      --in <name>        input name (default: the model's first input)\n"
        << "      --out <name>       output name (default: the model's only output)\n";
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
        else if (a == "-d" || a == "--backend")   { if (!(v = next("--backend"))) return false; opt.backend = v; }
        else if (a == "--precision")              { if (!(v = next("--precision"))) return false; opt.precision = v; }
        else if (a == "-b" || a == "--batch")     { if (!(v = next("--batch")))   return false; opt.batch = std::atoi(v); }
        else if (a == "-t" || a == "--threads")   { if (!(v = next("--threads"))) return false; opt.threads = std::atoi(v); }
        else if (a == "--fp16")                   { opt.fp16 = true; }
        else if (a == "--reconvert")              { opt.reconvert = true; }
        else if (a == "-s" || a == "--size")      { if (!(v = next("--size")))    return false; opt.size = std::atoi(v); }
        else if (a == "--slice")                  { if (!(v = next("--slice")))   return false; opt.slice = std::atoi(v); }
        else if (a == "--overlap")                { if (!(v = next("--overlap"))) return false; opt.overlap = std::strtof(v, nullptr); }
        else if (a == "--full")                   { opt.full = true; }
        else if (a == "--conf")                   { if (!(v = next("--conf")))    return false; opt.conf = std::strtof(v, nullptr); }
        else if (a == "--iou")                    { if (!(v = next("--iou")))     return false; opt.nms_iou = std::strtof(v, nullptr); }
        else if (a == "--merge")                  { if (!(v = next("--merge")))   return false; opt.merge = v; }
        else if (a == "--match")                  { if (!(v = next("--match")))   return false; opt.match = v; }
        else if (a == "--match-thresh")           { if (!(v = next("--match-thresh"))) return false; opt.match_thresh = std::strtof(v, nullptr); }
        else if (a == "--preprocess")             { if (!(v = next("--preprocess"))) return false; opt.preprocess = v; }
        else if (a == "--pad")                    { if (!(v = next("--pad")))     return false; opt.pad = v; }
        else if (a == "--box")                    { if (!(v = next("--box")))     return false; opt.box = v; }
        else if (a == "--in")                     { if (!(v = next("--in")))      return false; opt.in_name = v; }
        else if (a == "--out")                    { if (!(v = next("--out")))     return false; opt.out_name = v; }
        else if (!a.empty() && a[0] == '-')       { std::cerr << "Unknown option: " << a << "\n"; return false; }
        else positional.push_back(a);
    }
    auto one_of = [](const std::string& v, std::initializer_list<const char*> allowed, const char* opt_name) {
        for (const char* a : allowed) if (v == a) return true;
        std::cerr << opt_name << ": unexpected value '" << v << "'\n";
        return false;
    };
    std::vector<const char*> backends = {"auto"};
    static std::vector<std::string> avail = available_backends();
    for (const auto& b : avail) backends.push_back(b.c_str());
    bool backend_ok = false;
    for (const char* b : backends) backend_ok |= opt.backend == b;
    if (!backend_ok) {
        std::cerr << "--backend: '" << opt.backend << "' is not available in this build (available: auto, "
                  << join(avail, ", ") << ")\n";
        return false;
    }
    if (!one_of(opt.precision, {"normal", "high", "low"}, "--precision") ||
        !one_of(opt.preprocess, {"auto", "rtmdet", "yolo"}, "--preprocess") ||
        !one_of(opt.pad, {"auto", "center", "topleft"}, "--pad") ||
        !one_of(opt.box, {"auto", "xyxy", "cxcywh"}, "--box") ||
        !one_of(opt.merge, {"nmm", "nms"}, "--merge") || !one_of(opt.match, {"ios", "iou"}, "--match"))
        return false;
    if (opt.batch < 1) { std::cerr << "--batch must be at least 1\n"; return false; }
    if (opt.size != 0 && opt.size < 32) { std::cerr << "--size must be at least 32\n"; return false; }
    if (opt.slice < 0) { std::cerr << "--slice must be positive\n"; return false; }
    if (opt.overlap < 0.0f || opt.overlap >= 1.0f) { std::cerr << "--overlap must be in [0, 1)\n"; return false; }
    if (positional.size() < 2) {
        if (!positional.empty()) std::cerr << "Expected the model and at least one image or directory\n";
        return false;
    }
    opt.model = positional[0];
    opt.inputs.assign(positional.begin() + 1, positional.end());
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
// Model: .onnx -> .mnn conversion
// ---------------------------------------------------------------------------

static std::string lower_ext(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return e;
}

// MNN's converter: built next to this program by CMake (YOLOV9_MNNCONVERT),
// else next to the executable, else on PATH.
static std::string find_converter(const char* argv0) {
    std::error_code ec;
#ifdef YOLOV9_MNNCONVERT
    if (fs::is_regular_file(YOLOV9_MNNCONVERT, ec)) return YOLOV9_MNNCONVERT;
#endif
    const fs::path here = fs::absolute(argv0, ec).parent_path() / "MNNConvert";
    if (fs::is_regular_file(here, ec)) return here.string();
    return "MNNConvert";
}

static std::string shell_quote(const std::string& s) {
    std::string q = "'";
    for (char c : s) {
        if (c == '\'') q += "'\\''";
        else q += c;
    }
    return q + "'";
}

// Converted model for an .onnx: <stem>.mnn (or <stem>.fp16.mnn) next to it,
// converted again when missing, older than the .onnx, or with --reconvert.
static bool ensure_mnn(const fs::path& onnx, const Options& opt, const char* argv0, fs::path& mnn) {
    mnn = onnx;
    mnn.replace_extension(opt.fp16 ? ".fp16.mnn" : ".mnn");
    std::error_code ec;
    if (!opt.reconvert && fs::is_regular_file(mnn, ec) &&
        fs::last_write_time(mnn, ec) >= fs::last_write_time(onnx, ec))
        return true;
    const std::string conv = find_converter(argv0);
    const fs::path log = fs::path(mnn.string() + ".convert.log");
    std::string cmd = shell_quote(conv) + " -f ONNX --modelFile " + shell_quote(onnx.string()) + " --MNNModel " +
                      shell_quote(mnn.string()) + " --bizCode YOLOv9";
    if (opt.fp16) cmd += " --fp16";
    cmd += " > " + shell_quote(log.string()) + " 2>&1";
    std::cout << "Converting " << onnx.string() << " -> " << mnn.string() << (opt.fp16 ? " (fp16 weights)" : "")
              << " ..." << std::endl;
    const int rc = std::system(cmd.c_str());
    if (rc != 0 || !fs::is_regular_file(mnn, ec) || fs::file_size(mnn, ec) == 0) {
        std::cerr << "Model conversion failed (" << conv << ", exit code " << rc << "). Converter output: "
                  << log.string() << "\n";
        std::ifstream lf(log);
        std::string line, last;
        int shown = 0;
        while (std::getline(lf, line)) last += line + "\n", ++shown;
        if (shown > 0) std::cerr << (last.size() > 2000 ? last.substr(last.size() - 2000) : last);
        fs::remove(mnn, ec);
        return false;
    }
    fs::remove(log, ec);
    return true;
}


// ---------------------------------------------------------------------------
// SAHI
// ---------------------------------------------------------------------------

// Generate overlapping SAHI slice coordinates
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

// SAHI regions of one image: overlapping slices, plus the whole image with
// --full (skipped when one slice already covers everything).
static std::vector<Region> sahi_regions(int img_w, int img_h, int slice_size, const Options& opt) {
    std::vector<Region> regions = generate_slices(img_w, img_h, slice_size, slice_size, opt.overlap);
    if (opt.full && !(regions.size() == 1 && regions[0].width == img_w && regions[0].height == img_h)) {
        regions.push_back({0, 0, img_w, img_h});
    }
    return regions;
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
    const int cores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int threads = opt.threads > 0 ? opt.threads : cores;
#ifdef _OPENMP
    omp_set_num_threads(threads);  // preprocessing and decoding
#endif

    std::vector<std::string> class_names;
    std::string names_source;
    if (!opt.names_path.empty()) {
        if (!load_class_names(opt.names_path, class_names)) {
            std::cerr << "Cannot read class names file: " << opt.names_path << "\n";
            return 1;
        }
        names_source = opt.names_path;
    }

    // ---- Model: .onnx (MNN: converted to .mnn on first use) or .mnn ----
    std::error_code ec;
    const fs::path model_path(opt.model);
    if (!fs::is_regular_file(model_path, ec)) {
        std::cerr << "Model file not found: " << opt.model << "\n";
        return 1;
    }
    fs::path onnx_path, mnn_path;
    const std::string ext = lower_ext(model_path);
    if (ext == ".onnx") {
        onnx_path = model_path;
        if (!ensure_mnn(onnx_path, opt, argv[0], mnn_path)) return 1;
    } else if (ext == ".mnn") {
        mnn_path = model_path;
        // Metadata from an .onnx of the same name, if there is one
        fs::path guess = model_path;
        std::string stem = guess.stem().string();
        if (stem.size() > 5 && stem.compare(stem.size() - 5, 5, ".fp16") == 0) stem.resize(stem.size() - 5);
        guess = model_path.parent_path() / (stem + ".onnx");
        if (fs::is_regular_file(guess, ec)) onnx_path = guess;
    } else {
        std::cerr << "The model must be an .onnx or .mnn file (got " << opt.model << ")\n";
        return 1;
    }

    // ONNX metadata (LibreYOLO writes names, imgsz, model_family, letterbox_pad)
    OnnxInfo onnx;
    if (!onnx_path.empty()) {
        std::string err;
        if (!read_onnx_info(onnx_path.string(), onnx, err)) {
            std::cerr << "Warning: cannot read " << onnx_path.string() << ": " << err << "\n";
        }
    }
    auto meta = [&](const char* key) {
        auto it = onnx.metadata.find(key);
        return it == onnx.metadata.end() ? std::string() : it->second;
    };
    const std::string family = meta("model_family");
    const bool libreyolo = !meta("libreyolo_version").empty();
    if (names_source.empty()) {
        std::vector<std::string> n = parse_names_json(meta("names"));
        if (!n.empty()) {
            class_names = n;
            names_source = "model metadata";
        }
    }

    // Input size: --size, metadata, the ONNX input shape, else 640
    int size = opt.size;
    if (size == 0) size = std::atoi(meta("imgsz").c_str());
    if (size < 32 && !onnx.input_dims.empty() && onnx.input_dims[0].size() == 4 && onnx.input_dims[0][2].value > 0)
        size = static_cast<int>(onnx.input_dims[0][2].value);
    if (size < 32) size = 640;

    // Preprocessing and letterbox placement
    std::string prep = opt.preprocess, prep_source = "--preprocess";
    if (prep == "auto") {
        if (family.rfind("rtmdet", 0) == 0) { prep = "rtmdet"; prep_source = "metadata"; }
        else if (family.rfind("yolo", 0) == 0) { prep = "yolo"; prep_source = "metadata"; }
        else { prep = "yolo"; prep_source = "default"; }
    }
    std::string pad = opt.pad, pad_source = "--pad";
    if (pad == "auto") {
        const std::string mp = meta("letterbox_pad");
        if (mp == "topleft" || mp == "center" || mp == "centre") { pad = mp == "topleft" ? "topleft" : "center"; pad_source = "metadata"; }
        else { pad = prep == "rtmdet" ? "topleft" : "center"; pad_source = "default"; }
    }

    EngineConfig cfg;
    cfg.model_path = mnn_path.string();
    cfg.backend = opt.backend;
    cfg.precision = opt.precision;
    cfg.threads = threads;
    cfg.batch = opt.batch;
    cfg.size = size;
    cfg.norm = prep == "rtmdet" ? InputNorm::rtmdet() : InputNorm::yolo();
    cfg.topleft = pad == "topleft";
    cfg.box = opt.box;
    cfg.box_from_metadata = libreyolo;  // LibreYOLO's YOLOv9 head exports x1,y1,x2,y2
    cfg.in_name = opt.in_name;
    cfg.out_name = opt.out_name;
    DetectorInfo info;
    std::string error;
    std::unique_ptr<RegionDetector> detector = RegionDetector::create(cfg, info, error);
    if (!detector) {
        std::cerr << "Failed to set up " << mnn_path.string() << ": " << error << "\n";
        return 1;
    }

    std::vector<fs::path> images = collect_images(opt.inputs, opt.recursive);
    if (images.empty()) {
        std::cerr << "No input images.\n";
        return 1;
    }

    // Output naming: one image -> -o is a file; several -> -o is a directory.
    const bool single = images.size() == 1;
    fs::path out_file, out_dir;
    if (single) {
        out_file = opt.output.empty() ? fs::path("output.jpg") : fs::path(opt.output);
        if (fs::is_directory(out_file)) out_file /= images[0].stem().string() + "_det" + images[0].extension().string();
    } else {
        out_dir = opt.output.empty() ? fs::path("output") : fs::path(opt.output);
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
        std::string base = in.stem().string() + "_det";
        std::string e = in.extension().string();
        int& n = used_names[base + e];
        std::string name = (++n == 1) ? base + e : base + "_" + std::to_string(n) + e;
        return out_dir / name;
    };

    const int slice_size = opt.slice > 0 ? opt.slice : size;
    const double init_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    std::cout << "Model: " << mnn_path.string() << (family.empty() ? "" : " (" + family + ")") << " | "
              << info.in_name << " -> " << info.out_name << ", input " << size << "x" << size << ", "
              << info.n_anchor << " anchors x " << (info.n_feat - 4) << " classes, boxes "
              << (info.xyxy ? "xyxy" : "cxcywh") << " (" << info.box_source << ")\n"
              << "Backend: " << engine_name() << ", " << info.backend << " | batch " << info.batch
              << (info.batch < opt.batch ? " (the model has a fixed batch size)" : "") << " | precision "
              << opt.precision << " | preprocessing " << prep << " (" << prep_source << "), pad " << pad << " ("
              << pad_source << ")\n"
              << "Slices: " << slice_size << "x" << slice_size << ", overlap " << opt.overlap
              << (opt.full ? " + full image" : "") << " | merge " << opt.merge << "/" << opt.match << " > "
              << opt.match_thresh << " | " << images.size() << " image(s)";
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

        const std::vector<Region> regions = sahi_regions(bgr.cols, bgr.rows, slice_size, opt);
        const auto ti = clock::now();
        std::vector<std::vector<Detection>> per_region;
        const int failed = detector->detect(bgr, regions, opt.conf, opt.nms_iou, per_region);
        std::vector<Detection> all;
        for (auto& v : per_region) all.insert(all.end(), v.begin(), v.end());
        // Merge duplicates and fragments across slice borders
        std::vector<Detection> merged = merge_detections(all, opt);
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

        std::printf("%zu regions (%s), %zu raw -> %zu dets, %.1f ms%s\n", regions.size(),
                    detector->last_run_desc().c_str(), all.size(), merged.size(), infer_ms,
                    failed ? (" | " + std::to_string(failed) + " region(s) failed").c_str() : "");
        std::fflush(stdout);

        if (failed == static_cast<int>(regions.size())) ++failed_images; else ++ok_images;
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
