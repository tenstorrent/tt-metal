// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <variant>
#include "ttnn/tensor/types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::full {

struct operation_attributes_t {
    const ttnn::SmallVector<uint32_t> shape;
    const std::variant<float, int> fill_value;
    const DataType dtype;
    const Layout layout;
    const MemoryConfig memory_config;
};

struct tensor_args_t {
    const Tensor& any;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::full
