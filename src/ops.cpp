#include "ops.h"

#include <cmath>
#include <stdexcept>

Tensor embed(const Tensor& table, const std::vector<int>& token_ids) {
    if (table.shape.size() != 2) {
        throw std::runtime_error("embed table must be rank 2");
    }
    const int vocab = table.shape[0];
    const int hidden = table.shape[1];
    Tensor out({static_cast<int>(token_ids.size()), hidden});
    for (size_t t = 0; t < token_ids.size(); ++t) {
        const int id = token_ids[t];
        if (id < 0 || id >= vocab) {
            throw std::runtime_error("token id out of range");
        }
        const float* src = table.data.data() + static_cast<size_t>(id) * hidden;
        float* dst = out.data.data() + t * hidden;
        for (int i = 0; i < hidden; ++i) {
            dst[i] = src[i];
        }
    }
    return out;
}

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    if (x.shape.size() != 2 || weight.shape.size() != 2) {
        throw std::runtime_error("linear expects rank-2 x and weight");
    }
    const int rows = x.shape[0];
    const int in_f = x.shape[1];
    const int out_f = weight.shape[0];
    if (weight.shape[1] != in_f) {
        throw std::runtime_error("linear weight shape mismatch");
    }
    if (!bias.data.empty() &&
        (bias.shape.size() != 1 || bias.shape[0] != out_f)) {
        throw std::runtime_error("linear bias shape mismatch");
    }

    Tensor out({rows, out_f});
    for (int r = 0; r < rows; ++r) {
        const float* x_row = x.data.data() + static_cast<size_t>(r) * in_f;
        float* out_row = out.data.data() + static_cast<size_t>(r) * out_f;
        for (int o = 0; o < out_f; ++o) {
            const float* w_row = weight.data.data() + static_cast<size_t>(o) * in_f;
            float sum = 0.0f;
            for (int i = 0; i < in_f; ++i) {
                sum += x_row[i] * w_row[i];
            }
            if (!bias.data.empty()) {
                sum += bias.data[static_cast<size_t>(o)];
            }
            out_row[o] = sum;
        }
    }
    return out;
}

Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps) {
    if (x.shape.size() != 2 || weight.shape.size() != 1) {
        throw std::runtime_error("rmsnorm shape mismatch");
    }
    const int rows = x.shape[0];
    const int hidden = x.shape[1];
    if (weight.shape[0] != hidden) {
        throw std::runtime_error("rmsnorm weight size mismatch");
    }

    Tensor out({rows, hidden});
    for (int r = 0; r < rows; ++r) {
        const float* x_row = x.data.data() + static_cast<size_t>(r) * hidden;
        float* out_row = out.data.data() + static_cast<size_t>(r) * hidden;
        float sq_mean = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            sq_mean += x_row[i] * x_row[i];
        }
        sq_mean /= static_cast<float>(hidden);
        const float scale = 1.0f / std::sqrt(sq_mean + eps);
        for (int i = 0; i < hidden; ++i) {
            out_row[i] = x_row[i] * scale * weight.data[static_cast<size_t>(i)];
        }
    }
    return out;
}

Tensor silu(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.size(); ++i) {
        const float v = x.data[i];
        out.data[i] = v / (1.0f + std::exp(-v));
    }
    return out;
}

void rope(
    Tensor& q,
    Tensor& k,
    int num_q_heads,
    int num_kv_heads,
    float theta
) {
    if (q.shape.size() != 2 || k.shape.size() != 2) {
        throw std::runtime_error("rope expects rank-2 q and k");
    }
    const int seq = q.shape[0];
    const int q_dim = q.shape[1];
    const int k_dim = k.shape[1];
    if (q_dim % num_q_heads != 0 || k_dim % num_kv_heads != 0) {
        throw std::runtime_error("rope head packing mismatch");
    }
    const int head_dim = q_dim / num_q_heads;
    if (k_dim / num_kv_heads != head_dim) {
        throw std::runtime_error("rope q/k head_dim mismatch");
    }
    if (head_dim % 2 != 0) {
        throw std::runtime_error("rope head_dim must be even");
    }
    const int half = head_dim / 2;

    auto rotate = [&](Tensor& x, int num_heads) {
        for (int t = 0; t < seq; ++t) {
            for (int h = 0; h < num_heads; ++h) {
                float* row = x.data.data()
                    + (static_cast<size_t>(t) * num_heads + h) * head_dim;
                for (int i = 0; i < half; ++i) {
                    const float freq = 1.0f / std::pow(
                        theta,
                        static_cast<float>(2 * i) / static_cast<float>(head_dim)
                    );
                    const float angle = static_cast<float>(t) * freq;
                    const float c = std::cos(angle);
                    const float s = std::sin(angle);
                    const float x1 = row[i];
                    const float x2 = row[half + i];
                    row[i] = x1 * c - x2 * s;
                    row[half + i] = x2 * c + x1 * s;
                }
            }
        }
    };

    rotate(q, num_q_heads);
    rotate(k, num_kv_heads);
}

void add(Tensor& x, const Tensor& y) {
    if (x.shape != y.shape) {
        throw std::runtime_error("add shape mismatch");
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x.data[i] += y.data[i];
    }
}
