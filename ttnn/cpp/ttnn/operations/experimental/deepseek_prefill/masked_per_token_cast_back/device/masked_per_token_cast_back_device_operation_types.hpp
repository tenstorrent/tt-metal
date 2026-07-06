// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::masked_per_token_cast_back {

struct MaskedPerTokenCastBackParams {
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::MemoryConfig output_memory_config;
    uint32_t experts_per_chip;
};

struct MaskedPerTokenCastBackInputs {
    const Tensor& input_e4m3;
    const Tensor& input_scale;
    const Tensor& expert_region_offsets;
    const Tensor& expert_token_counts;
    const Tensor& global_expert_idx_table;
};

}  // namespace ttnn::experimental::prim::masked_per_token_cast_back
