#pragma once

#include "ttnn/experimental/lazy/lazy_operation.hpp"

namespace ttnn::experimental::lazy {

size_t count(const std::shared_ptr<LazyOperationInputs>& inputs) {
    size_t count = 0;
    inputs->for_each([&](const std::shared_ptr<LazyTensor>& input) { count++; });
    return count;
}

std::shared_ptr<LazyTensor> get(const std::shared_ptr<LazyOperationInputs>& inputs, size_t idx) {
    std::shared_ptr<LazyTensor> tensor;
    inputs->for_each([&](const std::shared_ptr<LazyTensor>& input) {
        if (idx == 0) {
            tensor = input;
        }
    });
    return tensor;
}

}  // namespace ttnn::experimental::lazy