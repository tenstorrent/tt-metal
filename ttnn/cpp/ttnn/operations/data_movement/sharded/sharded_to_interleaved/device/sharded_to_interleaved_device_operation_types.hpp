// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::prim {

struct ShardedToInterleavedParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{};
    uint32_t num_slices = 1;
    uint32_t slice_index = 0;
};

struct ShardedToInterleavedInputs {
    Tensor input_tensor;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::prim
