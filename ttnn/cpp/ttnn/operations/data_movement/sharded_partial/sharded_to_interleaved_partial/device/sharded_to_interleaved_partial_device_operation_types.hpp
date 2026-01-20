// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::prim {

struct ShardedToInterleavedPartialParams {
    uint32_t num_slices{};
    uint32_t slice_index{};
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{};
};

struct ShardedToInterleavedPartialInputs {
    Tensor input_tensor;
    Tensor cache_tensor;
};

}  // namespace ttnn::prim
