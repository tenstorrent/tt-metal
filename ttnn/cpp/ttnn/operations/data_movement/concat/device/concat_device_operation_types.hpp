// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::concat {

struct operation_attributes_t {
    uint32_t dim;
    unsigned int groups;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct tensor_args_t {
    std::vector<Tensor> input_tensors;
};

}  // namespace ttnn::operations::data_movement::concat
