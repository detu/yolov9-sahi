// Minimal protobuf reader for the parts of an ONNX ModelProto used here.
// Field numbers follow onnx.proto: ModelProto.graph = 7, .metadata_props = 14;
// GraphProto.initializer = 5, .input = 11, .output = 12; ValueInfoProto.name = 1,
// .type = 2; TypeProto.tensor_type = 1; Tensor.shape = 2; TensorShapeProto.dim = 1;
// Dimension.dim_value = 1, .dim_param = 2; TensorProto.name = 8;
// StringStringEntryProto.key = 1, .value = 2.

#include "onnx_meta.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <set>

namespace {

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;

    bool done() const { return !ok || p >= end; }
    uint64_t varint() {
        uint64_t v = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (p >= end) { ok = false; return 0; }
            const uint8_t b = *p++;
            v |= static_cast<uint64_t>(b & 0x7F) << shift;
            if (!(b & 0x80)) return v;
        }
        ok = false;
        return 0;
    }
    // Next field: returns false at the end. For length-delimited fields `sub`
    // spans the payload; other wire types are skipped.
    bool next(int& field, int& wire, Reader& sub) {
        if (done()) return false;
        const uint64_t key = varint();
        if (!ok) return false;
        field = static_cast<int>(key >> 3);
        wire = static_cast<int>(key & 7);
        switch (wire) {
            case 0: varint_value = varint(); break;
            case 1: if (end - p < 8) { ok = false; return false; } p += 8; break;
            case 5: if (end - p < 4) { ok = false; return false; } p += 4; break;
            case 2: {
                const uint64_t len = varint();
                if (!ok || len > static_cast<uint64_t>(end - p)) { ok = false; return false; }
                sub = Reader{p, p + len};
                p += len;
                break;
            }
            default: ok = false; return false;  // groups are not used by ONNX
        }
        return ok;
    }
    std::string str() const { return std::string(reinterpret_cast<const char*>(p), end - p); }
    uint64_t varint_value = 0;
};

void parse_value_info(Reader r, std::string& name, std::vector<OnnxDim>& dims) {
    int f, w;
    Reader s{nullptr, nullptr};
    while (r.next(f, w, s)) {
        if (f == 1 && w == 2) name = s.str();
        if (f != 2 || w != 2) continue;
        Reader t{nullptr, nullptr};  // TypeProto
        int f2, w2;
        while (s.next(f2, w2, t)) {
            if (f2 != 1 || w2 != 2) continue;  // tensor_type
            Reader u{nullptr, nullptr};
            int f3, w3;
            while (t.next(f3, w3, u)) {
                if (f3 != 2 || w3 != 2) continue;  // shape
                Reader d{nullptr, nullptr};
                int f4, w4;
                while (u.next(f4, w4, d)) {
                    if (f4 != 1 || w4 != 2) continue;  // dim
                    OnnxDim dim;
                    Reader v{nullptr, nullptr};
                    int f5, w5;
                    while (d.next(f5, w5, v)) {
                        if (f5 == 1 && w5 == 0) dim.value = static_cast<long long>(d.varint_value);
                        if (f5 == 2 && w5 == 2) dim.param = v.str();
                    }
                    dims.push_back(dim);
                }
            }
        }
    }
}

}  // namespace

bool read_onnx_info(const std::string& path, OnnxInfo& info, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = "cannot open " + path;
        return false;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Reader model{buf.data(), buf.data() + buf.size()};
    int field, wire;
    Reader sub{nullptr, nullptr};
    bool have_graph = false;
    std::vector<std::pair<std::string, std::vector<OnnxDim>>> inputs, outputs;
    std::set<std::string> initializers;
    while (model.next(field, wire, sub)) {
        if (field == 14 && wire == 2) {  // metadata_props
            std::string k, v;
            int f2, w2;
            Reader s{nullptr, nullptr};
            while (sub.next(f2, w2, s)) {
                if (f2 == 1 && w2 == 2) k = s.str();
                if (f2 == 2 && w2 == 2) v = s.str();
            }
            if (!k.empty()) info.metadata[k] = v;
        } else if (field == 7 && wire == 2) {  // graph
            have_graph = true;
            int f2, w2;
            Reader s{nullptr, nullptr};
            while (sub.next(f2, w2, s)) {
                if (w2 != 2) continue;
                if (f2 == 5) {  // initializer: only its name
                    int f3, w3;
                    Reader t{nullptr, nullptr};
                    while (s.next(f3, w3, t))
                        if (f3 == 8 && w3 == 2) initializers.insert(t.str());
                } else if (f2 == 11 || f2 == 12) {
                    std::string name;
                    std::vector<OnnxDim> dims;
                    parse_value_info(s, name, dims);
                    (f2 == 11 ? inputs : outputs).emplace_back(name, dims);
                }
            }
        }
    }
    if (!model.ok || !have_graph) {
        error = path + " is not a valid ONNX model";
        return false;
    }
    // Old exports also list initializers as graph inputs
    for (auto& in : inputs) {
        if (initializers.count(in.first)) continue;
        info.input_names.push_back(in.first);
        info.input_dims.push_back(in.second);
    }
    for (auto& out : outputs) {
        info.output_names.push_back(out.first);
        info.output_dims.push_back(out.second);
    }
    return true;
}

std::vector<std::string> parse_names_json(const std::string& s) {
    std::vector<std::pair<int, std::string>> items;
    size_t i = 0;
    auto skip_ws = [&]() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; };
    auto put_utf8 = [](std::string& o, unsigned cp) {
        if (cp < 0x80) o += static_cast<char>(cp);
        else if (cp < 0x800) { o += static_cast<char>(0xC0 | (cp >> 6)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { o += static_cast<char>(0xE0 | (cp >> 12)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
        else { o += static_cast<char>(0xF0 | (cp >> 18)); o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
    };
    auto read_string = [&](std::string& out) -> bool {
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        while (i < s.size() && s[i] != '"') {
            const char ch = s[i++];
            if (ch != '\\') { out += ch; continue; }
            if (i >= s.size()) return false;
            const char e = s[i++];
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    if (i + 4 > s.size()) return false;
                    unsigned cp = static_cast<unsigned>(std::strtoul(s.substr(i, 4).c_str(), nullptr, 16));
                    i += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        const unsigned lo = static_cast<unsigned>(std::strtoul(s.substr(i + 2, 4).c_str(), nullptr, 16));
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            i += 6;
                        }
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: out += e;  // \" \\ \/
            }
        }
        if (i >= s.size()) return false;
        ++i;
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
