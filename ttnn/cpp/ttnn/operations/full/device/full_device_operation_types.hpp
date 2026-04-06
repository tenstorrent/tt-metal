// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace ttnn::operations::full {
struct operation_attributes_t {
    const ttnn::SmallVector<uint32_t> shape;
    const std::variant<float, int> fill_value;
    ttnn::MeshDevice* mesh_device;
    const DataType dtype;
    const Layout layout;
    const MemoryConfig memory_config;
};

struct tensor_args_t {};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;
}  // namespace ttnn::operations::full
