#pragma once

#include "qwen/weights.h"

#include <vector>

std::vector<float> forward(
    const ModelWeights& weights,
    const std::vector<int>& token_ids
);
