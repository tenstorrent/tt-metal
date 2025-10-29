// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/experimental/jit/lazy_tensor.hpp"
#include <vector>

namespace ttnn::experimental::jit {

struct LazyOperation {
    LazyOperation() = default;
    virtual std::vector<Tensor> invoke(std::vector<LazyTensor> input_tensors) = 0;
    virtual ~LazyOperation() = default;
};

}  // namespace ttnn::experimental::jit
