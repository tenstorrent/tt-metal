// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The sparse SDPA family's compute pieces around the recipe core (sparse_sdpa_compute.cpp,
// sparse_sdpa_msa_compute.cpp): a segment is one token's query rows (one row per head) against the keys its index list
// selects, so the recipe state is per token, the K chunks arrive from a gather, and the output leaves row-major.
// Include after the recipe headers (recipe/recipe_streaming.hpp, recipe/recipe_ring.hpp).
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace sparse_recipe {
using namespace recipe_ring_cb;

// The state the recipe step assumes on entry (sdpa_recipe.cpp's kernel_main): srcA in the scores' format, srcB in the
// reduce scaler's, the packer in its default mode at the normalized output's format. The tilizes, the dequant and the
// untilize leave other configurations.
ALWI void restore_state() {
    reconfig_full_operand_srca(qk);
    reconfig_full_operand_srcb(identity_scale);
    pack_reconfig_data_format(out);
    PACK((llk_pack_init<ckernel::PackMode::Default, false, false, false>(out, 1)));
}

// One K chunk of a token's state through the recipe core (the hot path, at the kernel's optimization level). The K CB
// holds the chunk in the recipe's K^T layout; cb_v is the V CB, or the K CB for V read in place (V = K's leading
// columns).
template <uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles, uint32_t vd_tiles, uint32_t scale_fp32, uint32_t cb_v>
__attribute__((noinline)) void run_chunk(RecipeAccumulatorState& state, bool final, bool release_q) {
    // FP32 recipes process one Q tile row per group; paired BF16 recipes two (an odd chunk ends with one row).
    constexpr uint32_t subblock_h = recipe::fp32 ? 1 : 2;
    sdpa_recipe_segment<
        q_tiles,
        k_tiles,
        d_tiles,
        vd_tiles,
        scale_fp32,
        subblock_h,
        recipe::qk_w,
        subblock_h,
        recipe::pv_w,
        q,
        k,
        cb_v,
        qk,
        identity_scale,
        exp_max_diff,
        col_identity,
        recip_scratch,
        out,
        true>(state, 1, final, release_q);
}

// A token's Q rows (q_tiles tile rows of the reader's row-major pages, padding rows zero) into the recipe Q CB.
template <uint32_t cb_q_rm, uint32_t d_tiles, uint32_t q_tiles>
ALWI void tilize_q() {
    compute_kernel_lib::tilize<d_tiles, cb_q_rm, q>(q_tiles, q_tiles * tt::constants::TILE_HEIGHT);
}

// The normalized rows in the output CB (q_tiles rows) to the writer: the first `rows` untilize into cb_out_rm, the
// padding rows are dropped.
template <uint32_t q_tiles, uint32_t vd_tiles, uint32_t cb_out_rm>
ALWI void emit_rows(uint32_t rows) {
    compute_kernel_lib::untilize<
        vd_tiles,
        out,
        cb_out_rm,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(rows);
    if (rows < q_tiles) {
        CircularBuffer(out).wait_front((q_tiles - rows) * vd_tiles);
        CircularBuffer(out).pop_front((q_tiles - rows) * vd_tiles);
    }
    restore_state();
}
}  // namespace sparse_recipe
