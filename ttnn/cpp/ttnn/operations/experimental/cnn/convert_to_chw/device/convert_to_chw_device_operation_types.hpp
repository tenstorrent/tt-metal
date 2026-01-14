// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::cnn::to_chw {

struct operation_attributes_t {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
};

struct tensor_args_t {
    Tensor input;
};

}  // namespace ttnn::operations::experimental::cnn::to_chw
