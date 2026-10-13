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
    // When true, the compute kernel narrows the fp32 scale to bf16 on-device and runs the broadcast
    // multiply in bf16 (HiFi2); when false (default), the scale stays fp32 (HiFi4).
    bool narrow_scales_to_bf16 = false;

    bool token_count_aware = false;
    // Number of local experts hosted on this chip (token-count-aware path only).
    uint32_t experts_per_chip = 0;
    // When true, `input_scale` is the dispatch metadata tensor: per-token fp32 scales are read from its
    // row tail (bit-stored as int32) instead of a plain FLOAT32 (M, H/128) scale tensor.
    bool scales_from_metadata = false;
    // Elements per scale (32, 64 or 128). The I/O chunk stays 128 wide; a chunk row holds 128/scale_block_w scales.
    uint32_t scale_block_w = 128;
    // Plain path only: each e4m3 input row is packed as [H e4m3 bytes | H/scale_block_w fp32 scales]
    // (the scaled FP8 combine output). input_scale is the same tensor; the output is H wide.
    bool scales_in_row_tail = false;
    // Token-count-aware, 1x32 row-tail scales only: write TILE output (output_dtype BFLOAT8_B / BFLOAT16) so the
    // consumer skips its tilize; nothing is untilized here.
    bool tile_output = false;
    // tile_output only: each row is row_segments equal segments [seg e4m3 | seg/32 fp32 scales | pad to 64 B] (a
    // width-concat of per-chip packed slices, e.g. the all-gathered latent); 1 = one [H e4m3 | H/32 scales] row.
    uint32_t row_segments = 1;
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
