#pragma once

#include "tensor.h"

#include <vector>

Tensor embed(const Tensor& table, const std::vector<int>& token_ids);
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);
Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps);
Tensor silu(const Tensor& x);
void rope(Tensor& q, Tensor& k, int num_q_heads, int num_kv_heads, float theta);
void add(Tensor& x, const Tensor& y);
