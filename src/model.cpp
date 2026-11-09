#include "qwen/model.h"

#include "ops.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

Tensor attn(
    const Tensor& x,
    const AttentionWeights& w,
    int num_heads,
    int num_kv_heads,
    float rope_theta
) {
    const int seq = x.shape[0];
    const int hidden = x.shape[1];
    if (hidden % num_heads != 0) {
        throw std::runtime_error("hidden not divisible by num_heads");
    }
    const int head_dim = hidden / num_heads;
    const int kv_dim = num_kv_heads * head_dim;
    const int group = num_heads / num_kv_heads;
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("num_heads must divide by num_kv_heads");
    }

    Tensor q = linear(x, w.q_proj.weight, w.q_proj.bias);
    Tensor k = linear(x, w.k_proj.weight, w.k_proj.bias);
    Tensor v = linear(x, w.v_proj.weight, w.v_proj.bias);
    if (q.shape[1] != hidden || k.shape[1] != kv_dim || v.shape[1] != kv_dim) {
        throw std::runtime_error("attn projection width mismatch");
    }

    rope(q, k, num_heads, num_kv_heads, rope_theta);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor ctx({seq, hidden});

    for (int t = 0; t < seq; ++t) {
        for (int h = 0; h < num_heads; ++h) {
            const int kv_h = h / group;
            const float* q_vec = q.data.data()
                + (static_cast<size_t>(t) * num_heads + h) * head_dim;

            float max_score = -std::numeric_limits<float>::infinity();
            std::vector<float> scores(static_cast<size_t>(t) + 1);
            for (int s = 0; s <= t; ++s) {
                const float* k_vec = k.data.data()
                    + (static_cast<size_t>(s) * num_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[static_cast<size_t>(s)] = dot * scale;
                if (scores[static_cast<size_t>(s)] > max_score) {
                    max_score = scores[static_cast<size_t>(s)];
                }
            }

            float sum_exp = 0.0f;
            for (int s = 0; s <= t; ++s) {
                scores[static_cast<size_t>(s)] =
                    std::exp(scores[static_cast<size_t>(s)] - max_score);
                sum_exp += scores[static_cast<size_t>(s)];
            }

            float* out_vec = ctx.data.data()
                + (static_cast<size_t>(t) * num_heads + h) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] = 0.0f;
            }
            for (int s = 0; s <= t; ++s) {
                const float weight = scores[static_cast<size_t>(s)] / sum_exp;
                const float* v_vec = v.data.data()
                    + (static_cast<size_t>(s) * num_kv_heads + kv_h) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    out_vec[d] += weight * v_vec[d];
                }
            }
        }
    }

    return linear(ctx, w.o_proj.weight, w.o_proj.bias);
}

Tensor mlp(const Tensor& x, const MlpWeights& w) {
    Tensor gate = silu(linear(x, w.gate_proj.weight, w.gate_proj.bias));
    Tensor up = linear(x, w.up_proj.weight, w.up_proj.bias);
    if (gate.shape != up.shape) {
        throw std::runtime_error("mlp gate/up shape mismatch");
    }
    for (size_t i = 0; i < gate.size(); ++i) {
        gate.data[i] *= up.data[i];
    }
    return linear(gate, w.down_proj.weight, w.down_proj.bias);
}

}

std::vector<float> forward(
    const ModelWeights& weights,
    const std::vector<int>& token_ids
) {
    if (token_ids.empty()) {
        throw std::runtime_error("forward needs at least one token");
    }

    const ModelConfig& config = weights.config;
    Tensor x = embed(weights.embed_tokens, token_ids);

    for (int layer_i = 0; layer_i < config.num_hidden_layers; ++layer_i) {
        const LayerWeights& layer = weights.layers[static_cast<size_t>(layer_i)];
        Tensor normed = rmsnorm(x, layer.input_layernorm, config.rms_norm_eps);
        Tensor attn_out = attn(
            normed,
            layer.self_attn,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.rope_theta
        );
        add(x, attn_out);

        normed = rmsnorm(x, layer.post_attention_layernorm, config.rms_norm_eps);
        Tensor mlp_out = mlp(normed, layer.mlp);
        add(x, mlp_out);
    }

    x = rmsnorm(x, weights.final_norm, config.rms_norm_eps);

    const int hidden = config.hidden_size;
    const int vocab = config.vocab_size;
    const int last = static_cast<int>(token_ids.size()) - 1;
    const float* x_last = x.data.data() + static_cast<size_t>(last) * hidden;

    std::vector<float> logits(static_cast<size_t>(vocab));
    for (int v = 0; v < vocab; ++v) {
        const float* embed_row =
            weights.embed_tokens.data.data() + static_cast<size_t>(v) * hidden;
        float sum = 0.0f;
        for (int h = 0; h < hidden; ++h) {
            sum += x_last[h] * embed_row[h];
        }
        logits[static_cast<size_t>(v)] = sum;
    }
    return logits;
}
