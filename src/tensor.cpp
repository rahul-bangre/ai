#include "tensor.h"

#include <stdexcept>

Tensor::Tensor(std::vector<int> shape) : shape(std::move(shape)) {
    size_t n = 1;
    for (int d : this->shape) {
        if (d <= 0) {
            throw std::runtime_error("tensor dims must be positive");
        }
        n *= static_cast<size_t>(d);
    }
    data.assign(n, 0.0f);
}
