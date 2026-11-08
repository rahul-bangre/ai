#pragma once

#include <cstddef>
#include <vector>

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;

    Tensor() = default;
    explicit Tensor(std::vector<int> shape);

    size_t size() const { return data.size(); }
};
