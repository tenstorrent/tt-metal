// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/ema.h"

/*
 * -------------------------------------------------------------------------------------------------
 * The following snippet implements EMA using SFPI. This snippet assumes the following:
 * 1. Each tile has 1 sample (t dimension) each of 1024 elements (32x32) in the input
 * 2. Each tile has 4 faces
 * 3. The dst regs at prev_dst_index hold the previous output for each sample in the tile.
 *
 * With this, it does the following:
 * 1. Compute EMA for each sample in the tile
 * 2. Write the output to the output buffer
 * 3. Store the output in the previous dst regs for the next tile
 *
 * @note: To use this SFPI based implementation, the input tensor shape and reader/writer
 *        kernels should be configured such that consecutive samples for a channel are in
 *        consecutive tiles. Thus, the input is of shape [1, T, B, C].
 *
 *        Since this is 25% slower than the SFPU based implementation for similar shapes,
 *        we retain this snippet for reference, but do not use it below.
 */

/*
#ifdef TRISC_MATH
inline void ema_sfpi_tile(
    uint32_t inp_dst_index,
    uint32_t prv_dst_index,
    uint32_t out_dst_index) {
    constexpr uint32_t n_vector_in_tile = 32;

    const uint32_t inp_base_idx = inp_dst_index * n_vector_in_tile;
    const uint32_t prv_base_idx = prv_dst_index * n_vector_in_tile;
    const uint32_t out_base_idx = out_dst_index * n_vector_in_tile;

    constexpr size_t vectors_per_face = 8;
    for (size_t i = 0; i < vectors_per_face; i++) {
        vFloat inp = dst_reg[inp_base_idx + i];
        vFloat prv = dst_reg[prv_base_idx + i];
        vFloat result = first_sample ? inp * beta : inp * beta + prv * alpha;
        dst_reg[out_base_idx + i] = result;
        dst_reg[prv_base_idx + i] = result;
    }
}
#endif

inline void ema_sfpi_tile(
    uint32_t inp_dst_index,
    uint32_t prv_dst_index,
    uint32_t out_dst_index,
    float alpha,
    float beta,
    bool first_sample) {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(
        ema_sfpi_face, inp_dst_index, prv_dst_index, out_dst_index,
        VectorMode::RC, alpha, beta, first_sample));
}
// ------------------------------------------------------------------------------------------------
*/

namespace NAMESPACE {
void MAIN {
    // Compile time args
    // -----------------
    constexpr auto total_batches_per_core = get_compile_time_arg_val(0);
    constexpr auto tiles_per_channel = get_compile_time_arg_val(1);
    constexpr auto alpha_bits = get_compile_time_arg_val(2);
    constexpr auto beta_bits = get_compile_time_arg_val(3);

    // CB indices
    // ----------
    constexpr auto src_cb = tt::CBIndex::c_0;
    constexpr auto dst_cb = tt::CBIndex::c_1;
    constexpr auto trp_cb = tt::CBIndex::c_2;

    // DST indices
    // -----------
    constexpr auto inp_dst_index = 0;
    constexpr auto output_dst_index = inp_dst_index + 1;

    //-------------------------------------------------------------------------
    // Main loop - compute ema for each batch
    ema_init(alpha_bits, beta_bits);
    transpose_wh_init(src_cb, dst_cb);

    for (uint32_t batch_id = 0; batch_id < total_batches_per_core; ++batch_id) {
        // For each batch, clear the previous output
        ema_clear_previous_output();
        for (uint32_t tile_id = 0; tile_id < tiles_per_channel; ++tile_id) {
            // Read input, transpose and compute ema
            cb_wait_front(src_cb, 1);
            tile_regs_acquire();
            transpose_wh_tile(src_cb, 0, inp_dst_index);
            ema_tile(inp_dst_index);
            tile_regs_commit();
            cb_pop_front(src_cb, 1);

            cb_reserve_back(trp_cb, 1);
            tile_regs_wait();
            pack_tile(output_dst_index, trp_cb);
            tile_regs_release();
            cb_push_back(trp_cb, 1);

            // Transpose back and write to output
            cb_wait_front(trp_cb, 1);
            tile_regs_acquire();
            transpose_wh_tile(trp_cb, 0, output_dst_index);
            tile_regs_commit();
            cb_pop_front(trp_cb, 1);

            cb_reserve_back(dst_cb, 1);
            tile_regs_wait();
            pack_tile(output_dst_index, dst_cb);
            tile_regs_release();
            cb_push_back(dst_cb, 1);
        }
    }
}
}  // namespace NAMESPACE
