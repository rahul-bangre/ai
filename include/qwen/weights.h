#pragma once

#include "qwen/config.h"
#include "tensor.h"

#include <string>
#include <vector>

struct LinearWeights {
    Tensor weight;
    Tensor bias;
};

struct AttentionWeights {
    LinearWeights q_proj;
    LinearWeights k_proj;
    LinearWeights v_proj;
    LinearWeights o_proj;
};

struct MlpWeights {
    LinearWeights gate_proj;
    LinearWeights up_proj;
    LinearWeights down_proj;
};

struct LayerWeights {
    Tensor input_layernorm;
    AttentionWeights self_attn;
    Tensor post_attention_layernorm;
    MlpWeights mlp;
};

struct ModelWeights {
    ModelConfig config;
    Tensor embed_tokens;
    std::vector<LayerWeights> layers;
    Tensor final_norm;
};

ModelWeights load(const std::string& path);
