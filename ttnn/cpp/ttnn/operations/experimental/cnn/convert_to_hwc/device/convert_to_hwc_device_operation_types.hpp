// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::cnn {

struct operation_attributes_t {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
};

struct tensor_args_t {
    const Tensor& input;
};

}  // namespace ttnn::operations::experimental::cnn
