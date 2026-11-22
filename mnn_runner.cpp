// MNN engine: CUDA, Vulkan or CPU backend.

#include "runner.hpp"

#include <algorithm>
#include <iostream>

#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

namespace {

const char* forward_name(int t) {
    switch (t) {
        case MNN_FORWARD_CPU:
        case MNN_FORWARD_CPU_EXTENSION: return "CPU";
        case MNN_FORWARD_CUDA: return "CUDA";
        case MNN_FORWARD_VULKAN: return "Vulkan";
        default: return "other";
    }
}

class MnnRunner : public Runner {
public:
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session = nullptr;
    MNN::Tensor* in = nullptr;
    MNN::Tensor* out = nullptr;
    std::unique_ptr<MNN::Tensor> host_in, host_out;
    EngineConfig cfg;

    ~MnnRunner() override {
        if (net && session) net->releaseSession(session);
    }

    // Shape the session for `batch` regions; false if the model can't take it.
    bool resize(int batch, std::string& error) {
        net->resizeTensor(in, {batch, 3, cfg.size, cfg.size});
        net->resizeSession(session);
        out = cfg.out_name.empty() ? net->getSessionOutput(session, nullptr)
                                   : net->getSessionOutput(session, cfg.out_name.c_str());
        if (!out) {
            error = "output '" + cfg.out_name + "' not found";
            return false;
        }
        const std::vector<int> s = out->shape();
        if (s.size() != 3 || s[0] != batch) return false;
        host_in.reset(new MNN::Tensor(in, MNN::Tensor::CAFFE));
        host_out.reset(new MNN::Tensor(out, MNN::Tensor::CAFFE));
        return true;
    }

    float* input() override { return host_in->host<float>(); }

    bool run(std::string& error) override {
        if (!in->copyFromHostTensor(host_in.get())) {
            error = "copy to the backend failed";
            return false;
        }
        const MNN::ErrorCode rc = net->runSession(session);
        if (rc != MNN::NO_ERROR) {
            error = "runSession failed (MNN error " + std::to_string(static_cast<int>(rc)) + ")";
            return false;
        }
        if (!out->copyToHostTensor(host_out.get())) {
            error = "copy from the backend failed";
            return false;
        }
        return true;
    }

    std::vector<int64_t> output_shape() const override {
        const std::vector<int> s = out->shape();
        return std::vector<int64_t>(s.begin(), s.end());
    }
    const float* output() const override { return host_out->host<float>(); }
};

}  // namespace

const char* engine_name() { return "MNN"; }

std::vector<std::string> available_backends() {
    std::vector<std::string> b;
#if YOLOV9_WITH_CUDA
    b.push_back("cuda");
#endif
#if YOLOV9_WITH_VULKAN
    b.push_back("vulkan");
#endif
    b.push_back("cpu");
    return b;
}

std::unique_ptr<Runner> create_runner(const EngineConfig& cfg, RunnerInfo& info, std::string& error) {
    std::unique_ptr<MnnRunner> m(new MnnRunner);
    m->cfg = cfg;
    m->net.reset(MNN::Interpreter::createFromFile(cfg.model_path.c_str()), MNN::Interpreter::destroy);
    if (!m->net) {
        error = "MNN cannot load " + cfg.model_path;
        return nullptr;
    }
    m->net->setSessionMode(MNN::Interpreter::Session_Release);

    // Backend: the requested one, or the first compiled GPU backend for "auto"
    std::vector<std::string> order;
    if (cfg.backend == "auto") order = available_backends();
    else order = {cfg.backend};
    for (const std::string& b : order) {
        MNN::ScheduleConfig sc;
        sc.type = b == "cuda" ? MNN_FORWARD_CUDA : b == "vulkan" ? MNN_FORWARD_VULKAN : MNN_FORWARD_CPU;
        sc.backupType = MNN_FORWARD_CPU;  // ops a GPU backend lacks run on the CPU
        if (sc.type == MNN_FORWARD_CPU) sc.numThread = std::max(1, cfg.threads);
        MNN::BackendConfig bc;
        bc.precision = cfg.precision == "high" ? MNN::BackendConfig::Precision_High
                     : cfg.precision == "low" ? MNN::BackendConfig::Precision_Low
                                              : MNN::BackendConfig::Precision_Normal;
        sc.backendConfig = &bc;
        MNN::Session* s = m->net->createSession(sc);
        if (!s) continue;
        int used[4] = {MNN_FORWARD_CPU, MNN_FORWARD_CPU, MNN_FORWARD_CPU, MNN_FORWARD_CPU};
        m->net->getSessionInfo(s, MNN::Interpreter::BACKENDS, used);
        if (used[0] == MNN_FORWARD_CPU_EXTENSION) used[0] = MNN_FORWARD_CPU;  // x86/ARM CPU backend
        if (used[0] != sc.type) {
            // MNN fell back to the CPU: the backend isn't available on this machine
            if (cfg.backend != "auto") {
                std::cerr << "Warning: " << b << " backend not available here; MNN uses "
                          << forward_name(used[0]) << "\n";
            } else if (sc.type != MNN_FORWARD_CPU) {
                m->net->releaseSession(s);
                continue;
            }
        }
        m->session = s;
        info.backend = forward_name(used[0]);
        if (used[0] == MNN_FORWARD_CPU) info.backend += " (" + std::to_string(std::max(1, cfg.threads)) + " threads)";
        break;
    }
    if (!m->session) {
        error = "MNN could not create a session";
        return nullptr;
    }

    // One input, one decoded output (B, anchors, 4 + classes) expected
    const auto& outs = m->net->getSessionOutputAll(m->session);
    if (cfg.out_name.empty() && outs.size() != 1) {
        error = "the model has " + std::to_string(outs.size()) + " outputs; expected one decoded output "
                "(batch, anchors, 4 + classes). Export the default ONNX (not a raw-heads layout), or "
                "pass the output name with --out.";
        return nullptr;
    }
    m->in = cfg.in_name.empty() ? m->net->getSessionInput(m->session, nullptr)
                                : m->net->getSessionInput(m->session, cfg.in_name.c_str());
    if (!m->in) {
        error = "input '" + cfg.in_name + "' not found";
        return nullptr;
    }
    info.in_name = cfg.in_name.empty() ? m->net->getSessionInputAll(m->session).begin()->first : cfg.in_name;
    info.out_name = cfg.out_name.empty() ? outs.begin()->first : cfg.out_name;

    // Batch: try the requested size; models exported without a dynamic batch stay at 1
    std::string err;
    info.batch = std::max(1, cfg.batch);
    if (!m->resize(info.batch, err)) {
        if (!err.empty()) {
            error = err;
            return nullptr;
        }
        info.batch = 1;
        if (!m->resize(1, err)) {
            const std::vector<int> s = m->out ? m->out->shape() : std::vector<int>{};
            std::string sh;
            for (int v : s) sh += (sh.empty() ? "" : ",") + std::to_string(v);
            error = err.empty() ? "output has shape [" + sh + "], expected [batch, anchors, 4 + classes]" : err;
            return nullptr;
        }
    }
    return m;
}
