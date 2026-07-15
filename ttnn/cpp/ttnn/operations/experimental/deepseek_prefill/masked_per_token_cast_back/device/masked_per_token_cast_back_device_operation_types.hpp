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
    // When true, `input_scale` is actually the dispatch `metadata` tensor: per-token fp32 scales live in
    // its row tail (columns [metadata_len - H/128, metadata_len)), bit-stored as int32. The reader skips
    // the leading routing columns via scale_col_offset = scale_last_dim - H/128. When false (default),
    // `input_scale` is a plain FLOAT32 (M, H/128) scale tensor (col offset 0).
    bool scales_from_metadata = false;
};

struct MaskedPerTokenCastBackInputs {
    const Tensor& input_e4m3;
    const Tensor& input_scale;
    const Tensor& expert_region_offsets;
    const Tensor& expert_token_counts;
    const Tensor& global_expert_idx_table;
};

}  // namespace ttnn::experimental::prim::masked_per_token_cast_back
