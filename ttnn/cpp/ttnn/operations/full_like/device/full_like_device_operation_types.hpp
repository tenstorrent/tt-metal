// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace ttnn::prim {
struct operation_attributes_t {
    const std::variant<float, int> fill_value;
    const DataType dtype;
    const Layout layout;
    const MemoryConfig memory_config;
};

struct tensor_args_t {
    const Tensor& input;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;
}  // namespace ttnn::prim
