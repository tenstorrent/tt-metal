// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_to_fp8 {

struct PerTokenCastToFp8Params {
    tt::tt_metal::MemoryConfig output_memory_config;
    bool round_scale_to_power_of_two;
    uint32_t scale_block_w;      // elements per scale; divides the 128-element I/O chunk (32, 64 or 128)
    bool round_to_nearest_even;  // RNE to e4m3 in SFPU before the (truncating) packer
    // Expert-buffer mode (TILE input): quantize only the valid prefix of a flat MoE expert buffer, sized
    // on device from region offsets / token counts / global expert table (like per_token_cast_back).
    bool token_count_aware = false;
    uint32_t experts_per_chip = 0;
    // Write each row as [H e4m3 | H/scale_block_w fp32 scales] in the e4m3 output (scale output unused).
    bool pack_scales_in_row = false;
    // pack_scales_in_row only: pad each packed row to a multiple of this many bytes (0 = no padding). 64 keeps
    // every row segment of a later width-concat (e.g. an all-gather of per-chip slices) DRAM-read aligned.
    uint32_t packed_row_align = 0;
};

struct PerTokenCastToFp8Inputs {
    const Tensor& input_tensor;
    std::optional<Tensor> expert_region_offsets;
    std::optional<Tensor> expert_token_counts;
    std::optional<Tensor> global_expert_idx_table;
};

}  // namespace ttnn::experimental::prim::per_token_cast_to_fp8
