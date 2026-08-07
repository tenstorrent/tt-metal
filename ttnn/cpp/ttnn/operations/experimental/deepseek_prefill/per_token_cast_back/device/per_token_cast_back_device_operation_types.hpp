// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_back {

struct PerTokenCastBackParams {
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::MemoryConfig output_memory_config;

    bool token_count_aware = false;
    // Number of local experts hosted on this chip (token-count-aware path only).
    uint32_t experts_per_chip = 0;
    // When true, `input_scale` is the dispatch metadata tensor: per-token fp32 scales are read from its
    // row tail (bit-stored as int32) instead of a plain FLOAT32 (M, H/128) scale tensor.
    bool scales_from_metadata = false;
};

struct PerTokenCastBackInputs {
    const Tensor& input_e4m3;
    // Plain path: FLOAT32 (M, H/128) scale tensor. Token-count-aware metadata path: the int32/uint32
    // dispatch metadata tensor whose row tail holds the per-token fp32 scales.
    const Tensor& input_scale;
    // Token-count-aware path only: per-expert region offsets / token counts / global-expert-idx table (UINT32).
    std::optional<Tensor> expert_region_offsets;
    std::optional<Tensor> expert_token_counts;
    std::optional<Tensor> global_expert_idx_table;
};

}  // namespace ttnn::experimental::prim::per_token_cast_back
