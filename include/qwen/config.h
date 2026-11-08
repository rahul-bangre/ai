#pragma once

#include <cstdint>

struct ModelConfig {
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t max_position_embeddings = 0;
    int32_t bos_token_id = 0;
    int32_t eos_token_id = 0;
    float rms_norm_eps = 0.0f;
    float rope_theta = 0.0f;
    int32_t tie_word_embeddings = 0;
};
