// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/experimental/jit/lazy_tensor.hpp"
#include <vector>

namespace ttnn::experimental::jit {

struct LazyOperation {
    LazyOperation() = default;
    virtual std::vector<tt::tt_metal::Tensor> invoke(const std::vector<tt::tt_metal::Tensor>& input_tensors) = 0;
    virtual std::string_view name() const = 0;
    // TODO: Do we need some attributes for serialization purposes?
    virtual ~LazyOperation() = default;
};

}  // namespace ttnn::experimental::jit
