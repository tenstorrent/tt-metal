// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/lazy/lazy_tensor.hpp"
#include <any>
#include <functional>

namespace ttnn::experimental::lazy {

// Interface for lazy operation inputs
struct LazyOperationInputs {
    virtual void for_each(const std::function<void(const std::shared_ptr<LazyTensor>&)>& fn) const {}
    virtual std::any inputs() const { return std::any(); }
    virtual ~LazyOperationInputs() = default;

    size_t size() const {
        size_t count = 0;
        for_each([&](const std::shared_ptr<LazyTensor>&) { count++; });
        return count;
    }

    std::shared_ptr<LazyTensor> at(size_t idx) const {
        std::shared_ptr<LazyTensor> result;
        size_t current_idx = 0;
        for_each([&](const std::shared_ptr<LazyTensor>& input) {
            if (idx == current_idx) {
                result = input;
            }
            current_idx++;
        });
        return result;
    }
};

}  // namespace ttnn::experimental::lazy
