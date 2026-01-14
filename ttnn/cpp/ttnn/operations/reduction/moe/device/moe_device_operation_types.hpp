// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction::moe {

struct operation_attributes_t {
    uint16_t k{};
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct tensor_args_t {
    Tensor input;
    Tensor expert_mask;
    Tensor topk_mask;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::operations::reduction::moe
