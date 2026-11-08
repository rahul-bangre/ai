#include "qwen/weights.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

constexpr char kMagic[4] = {'Q', 'W', 'N', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kDtypeF32 = 0;

uint32_t u32(std::ifstream& in) {
    uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("failed reading u32 from weight file");
    }
    return value;
}

float f32(std::ifstream& in) {
    float value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("failed reading f32 from weight file");
    }
    return value;
}

std::string name(std::ifstream& in) {
    const uint32_t name_len = u32(in);
    std::string out(name_len, '\0');
    in.read(out.data(), name_len);
    if (!in) {
        throw std::runtime_error("failed reading tensor name");
    }
    return out;
}

Tensor tensor(std::ifstream& in) {
    const uint32_t ndim = u32(in);
    if (ndim == 0 || ndim > 4) {
        throw std::runtime_error("unsupported tensor rank");
    }
    std::vector<int> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        shape[i] = static_cast<int>(u32(in));
    }
    const uint32_t dtype = u32(in);
    if (dtype != kDtypeF32) {
        throw std::runtime_error("only float32 weights are supported");
    }
    Tensor out(shape);
    in.read(
        reinterpret_cast<char*>(out.data.data()),
        static_cast<std::streamsize>(out.size() * sizeof(float))
    );
    if (!in) {
        throw std::runtime_error("failed reading tensor payload");
    }
    return out;
}

const Tensor& get(
    const std::unordered_map<std::string, Tensor>& tensors,
    const std::string& key
) {
    const auto it = tensors.find(key);
    if (it == tensors.end()) {
        throw std::runtime_error("missing tensor: " + key);
    }
    return it->second;
}

LinearWeights linear(
    const std::unordered_map<std::string, Tensor>& tensors,
    const std::string& prefix,
    bool has_bias
) {
    LinearWeights out;
    out.weight = get(tensors, prefix + ".weight");
    if (has_bias) {
        out.bias = get(tensors, prefix + ".bias");
    }
    return out;
}

}

ModelWeights load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("could not open weights file: " + path);
    }

    char magic[4];
    in.read(magic, 4);
    if (!in || magic[0] != kMagic[0] || magic[1] != kMagic[1] ||
        magic[2] != kMagic[2] || magic[3] != kMagic[3]) {
        throw std::runtime_error("bad weight file magic");
    }
    if (u32(in) != kVersion) {
        throw std::runtime_error("unsupported weight file version");
    }

    ModelWeights model;
    model.config.vocab_size = static_cast<int32_t>(u32(in));
    model.config.hidden_size = static_cast<int32_t>(u32(in));
    model.config.intermediate_size = static_cast<int32_t>(u32(in));
    model.config.num_hidden_layers = static_cast<int32_t>(u32(in));
    model.config.num_attention_heads = static_cast<int32_t>(u32(in));
    model.config.num_key_value_heads = static_cast<int32_t>(u32(in));
    model.config.max_position_embeddings = static_cast<int32_t>(u32(in));
    model.config.bos_token_id = static_cast<int32_t>(u32(in));
    model.config.eos_token_id = static_cast<int32_t>(u32(in));
    model.config.rms_norm_eps = f32(in);
    model.config.rope_theta = f32(in);
    model.config.tie_word_embeddings = static_cast<int32_t>(u32(in));

    const uint32_t tensor_count = u32(in);
    std::unordered_map<std::string, Tensor> tensors;
    tensors.reserve(tensor_count);
    for (uint32_t i = 0; i < tensor_count; ++i) {
        std::string key = name(in);
        Tensor value = tensor(in);
        tensors.emplace(std::move(key), std::move(value));
    }

    model.embed_tokens = get(tensors, "model.embed_tokens.weight");
    model.final_norm = get(tensors, "model.norm.weight");
    model.layers.resize(static_cast<size_t>(model.config.num_hidden_layers));

    for (int layer_i = 0; layer_i < model.config.num_hidden_layers; ++layer_i) {
        const std::string prefix =
            "model.layers." + std::to_string(layer_i);
        LayerWeights& layer = model.layers[static_cast<size_t>(layer_i)];
        layer.input_layernorm = get(tensors, prefix + ".input_layernorm.weight");
        layer.post_attention_layernorm =
            get(tensors, prefix + ".post_attention_layernorm.weight");
        layer.self_attn.q_proj = linear(tensors, prefix + ".self_attn.q_proj", true);
        layer.self_attn.k_proj = linear(tensors, prefix + ".self_attn.k_proj", true);
        layer.self_attn.v_proj = linear(tensors, prefix + ".self_attn.v_proj", true);
        layer.self_attn.o_proj = linear(tensors, prefix + ".self_attn.o_proj", false);
        layer.mlp.gate_proj = linear(tensors, prefix + ".mlp.gate_proj", false);
        layer.mlp.up_proj = linear(tensors, prefix + ".mlp.up_proj", false);
        layer.mlp.down_proj = linear(tensors, prefix + ".mlp.down_proj", false);
    }

    return model;
}
