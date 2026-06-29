// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Fused Wan2.2 distributed LayerNorm compute kernel (Welford) — Phase 2 bringup.
 *
 * Scope (Phase 2): is_tp_1 (no all-gather / fabric), whole-row norm (no
 * per_head), no RoPE, bf16 input, resident layout (no streaming / block-major).
 * Optional gamma (weight) and beta (bias), broadcast [1,H] or per-token [N,H].
 *
 * Per tile-row of the core's slice:
 *   1) PRE (Welford): transpose each of the num_tile_cols input tiles into DST
 *      and fold it into the running per-token mean/M2 (welford_update). After
 *      all tiles, welford_finalize_to_row converts M2 -> variance and writes the
 *      per-token mean to row 0 of one DST tile and variance to row 0 of the next.
 *   2) Transpose mean/var back to col 0 (so they broadcast over feature columns),
 *      compute 1/std = rsqrt(var + eps), and stage mean (stats_local_cb) and
 *      1/std (reduce_result_cb) as col-0 broadcast tiles.
 *   3) POST: x' = (x - mean) [rotated_input_cb] ; x'' = x' * (1/std)
 *      [norm_result_cb] ; optional * weight ; optional + bias ; -> output_cb.
 *
 * This shares the program factory's CBs / reader / writer with the RMSNorm
 * kernel. CBs idle on the is_tp_1 + no-RoPE path are repurposed: stats_local_cb
 * holds the per-token mean, reduce_result_cb holds 1/std, and rotated_input_cb
 * (RoPE-only normally) holds (x - mean). The cross-shard Welford merge (TP>1)
 * and wide-shard / fp32-input handling are later phases. See
 * WELFORD_LAYERNORM_DESIGN.md.
 */

#include <cstdint>
#include <array>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/welford.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    // === Compile-time args (shared list with the RMSNorm kernel; LN reads a subset) ===
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mean_cb = get_compile_time_arg_val(1);  // stats_local_cb (idle on is_tp_1) -> per-token mean
    constexpr uint32_t weight_cb = get_compile_time_arg_val(3);
    constexpr uint32_t invstd_cb = get_compile_time_arg_val(7);        // reduce_result_cb -> 1/std
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(8);  // norm / weight result (fp32)
    constexpr uint32_t output_cb = get_compile_time_arg_val(10);
    constexpr uint32_t xmm_cb = get_compile_time_arg_val(14);  // rotated_input_cb (idle, no RoPE) -> (x - mean)
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(15);
    constexpr uint32_t block_size = get_compile_time_arg_val(16);
    constexpr uint32_t has_weight = get_compile_time_arg_val(20);
    constexpr uint32_t bias_cb = get_compile_time_arg_val(28);
    constexpr uint32_t has_bias = get_compile_time_arg_val(29);
    constexpr uint32_t per_token_weight = get_compile_time_arg_val(32);
    constexpr uint32_t per_token_bias = get_compile_time_arg_val(33);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(34);

    // Welford reduces over the full row: num_tile_cols * TILE_WIDTH features.
    constexpr uint32_t tile_width = 32u;
    constexpr uint32_t reduce_width = num_tile_cols * tile_width;
    // No reciprocal LUT in Phase 2: reciprocal_size==0 -> runtime 1/(idx+1) division.
    constexpr std::array<uint32_t, 0> no_lut{};

    // DST tile assignment for the Welford pass.
    constexpr uint32_t welford_in_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // has_bias implies has_weight (enforced in validate). Keep the normalized
    // result in fp32 intermediate_cb when weight/bias follow; otherwise straight
    // to output_cb.
    constexpr uint32_t norm_result_cb = (has_weight != 0) ? intermediate_cb : output_cb;
    constexpr uint32_t weight_result_cb = (has_bias != 0) ? intermediate_cb : output_cb;

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    binary_op_init_common(input_cb, input_cb, input_cb);

    for (uint32_t row = 0; row < num_tile_rows; row++) {
        // Input for this row stays resident: used by the Welford pass (transpose)
        // and re-read by POST (x - mean). Popped at end of row.
        cb_wait_front(input_cb, num_tile_cols);

        // -------- PHASE 1: PRE — Welford per-token (mean, var) over the row --------
        {
            DeviceZoneScopedN("LN_PRE_WELFORD");
            // bf16 input -> transpose_wh_tile goes through SrcA and does NOT touch
            // the SFPU replay buffer, so no welford_init<PreserveStats> recovery is
            // needed after each transpose (that is only required on the fp32
            // UnpackToDest path). Phase 2 requires bf16 input (enforced in validate).
            transpose_wh_init_short(input_cb);
            tile_regs_acquire();
            welford_init();
            uint32_t start_n = 0;
            for (uint32_t col = 0; col < num_tile_cols; col++) {
                transpose_wh_tile(input_cb, col, welford_in_dst);
                welford_update<0>(welford_in_dst, start_n, no_lut);
                start_n += tile_width;
            }
            // M2 -> variance (population, scale = 1/reduce_width); mean -> row 0 of
            // mean_dst, var -> row 0 of var_dst.
            welford_finalize_to_row<0>(mean_dst, reduce_width - 1, no_lut);
            tile_regs_commit();

            // Stash the row-0 mean/var so we can transpose them back to col 0.
            cb_reserve_back(mean_cb, 1);
            cb_reserve_back(invstd_cb, 1);
            tile_regs_wait();
            pack_reconfig_data_format(mean_cb);
            pack_tile(mean_dst, mean_cb);
            pack_reconfig_data_format(invstd_cb);
            pack_tile(var_dst, invstd_cb);
            tile_regs_release();
            cb_push_back(mean_cb, 1);
            cb_push_back(invstd_cb, 1);
        }

        // -------- Transpose mean/var to col 0; var -> 1/std = rsqrt(var + eps) --------
        {
            DeviceZoneScopedN("LN_STAT_FINALIZE");
            cb_wait_front(mean_cb, 1);
            cb_wait_front(invstd_cb, 1);
            reconfig_data_format_srca(mean_cb);
            transpose_wh_init_short(mean_cb);
            tile_regs_acquire();
            transpose_wh_tile(mean_cb, 0, mean_dst);   // mean row 0 -> col 0
            transpose_wh_tile(invstd_cb, 0, var_dst);  // var  row 0 -> col 0
            // 1/std = rsqrt(var + eps), in place on var_dst (col 0).
            binop_with_scalar_tile_init();
            add_unary_tile(var_dst, eps_bits);
            rsqrt_tile_init();
            rsqrt_tile(var_dst);
            tile_regs_commit();

            // Overwrite mean_cb / invstd_cb (now col-0 broadcast tiles) in place.
            cb_pop_front(mean_cb, 1);
            cb_pop_front(invstd_cb, 1);
            cb_reserve_back(mean_cb, 1);
            cb_reserve_back(invstd_cb, 1);
            tile_regs_wait();
            pack_reconfig_data_format(mean_cb);
            pack_tile(mean_dst, mean_cb);
            pack_reconfig_data_format(invstd_cb);
            pack_tile(var_dst, invstd_cb);
            tile_regs_release();
            cb_push_back(mean_cb, 1);
            cb_push_back(invstd_cb, 1);
            cb_wait_front(mean_cb, 1);
            cb_wait_front(invstd_cb, 1);
        }

        // -------- PHASE 2: POST — (x - mean) * (1/std) [* weight] [+ bias] --------
        // Sub-phase A: x - mean (broadcast mean over feature cols) -> xmm_cb.
        {
            DeviceZoneScopedN("LN_SUB_MEAN");
            reconfig_data_format(input_cb, mean_cb);
            pack_reconfig_data_format(xmm_cb);
            sub_bcast_cols_init_short(input_cb, mean_cb);
            for (uint32_t col = 0; col < num_tile_cols; col += block_size) {
                const uint32_t n = (col + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col);
                cb_reserve_back(xmm_cb, n);
                tile_regs_acquire();
                for (uint32_t i = 0; i < n; i++) {
                    sub_tiles_bcast_cols(input_cb, mean_cb, col + i, 0, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < n; i++) {
                    pack_tile(i, xmm_cb);
                }
                tile_regs_release();
                cb_push_back(xmm_cb, n);
            }
        }

        // Sub-phase B: (x - mean) * (1/std) -> norm_result_cb.
        {
            DeviceZoneScopedN("LN_MUL_INVSTD");
            reconfig_data_format(xmm_cb, invstd_cb);
            pack_reconfig_data_format(norm_result_cb);
            mul_bcast_cols_init_short(xmm_cb, invstd_cb);
            for (uint32_t col = 0; col < num_tile_cols; col += block_size) {
                const uint32_t n = (col + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col);
                cb_wait_front(xmm_cb, block_size);
                cb_reserve_back(norm_result_cb, n);
                tile_regs_acquire();
                for (uint32_t i = 0; i < n; i++) {
                    mul_tiles_bcast_cols(xmm_cb, invstd_cb, i, 0, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < n; i++) {
                    pack_tile(i, norm_result_cb);
                }
                tile_regs_release();
                cb_push_back(norm_result_cb, n);
                cb_pop_front(xmm_cb, block_size);
            }
        }

        // Sub-phase C: * weight (gamma).
        if constexpr (has_weight != 0) {
            DeviceZoneScopedN("LN_WEIGHT");
            reconfig_data_format(norm_result_cb, weight_cb);
            pack_reconfig_data_format(weight_result_cb);
            if constexpr (per_token_weight != 0) {
                mul_tiles_init(norm_result_cb, weight_cb);
            } else {
                mul_bcast_rows_init_short(norm_result_cb, weight_cb);
            }
            for (uint32_t col = 0; col < num_tile_cols; col += block_size) {
                const uint32_t n = (col + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col);
                cb_wait_front(weight_cb, col + n);
                cb_wait_front(norm_result_cb, block_size);
                tile_regs_acquire();
                for (uint32_t i = 0; i < n; i++) {
                    if constexpr (per_token_weight != 0) {
                        mul_tiles(norm_result_cb, weight_cb, i, col + i, i);
                    } else {
                        mul_tiles_bcast_rows(norm_result_cb, weight_cb, i, col + i, i);
                    }
                }
                tile_regs_commit();
                cb_pop_front(norm_result_cb, block_size);
                cb_reserve_back(weight_result_cb, n);
                tile_regs_wait();
                for (uint32_t i = 0; i < n; i++) {
                    pack_tile(i, weight_result_cb);
                }
                tile_regs_release();
                cb_push_back(weight_result_cb, n);
            }
        }

        // Sub-phase D: + bias (beta).
        if constexpr (has_bias != 0) {
            DeviceZoneScopedN("LN_BIAS");
            reconfig_data_format(weight_result_cb, bias_cb);
            pack_reconfig_data_format(output_cb);
            if constexpr (per_token_bias != 0) {
                add_tiles_init(weight_result_cb, bias_cb);
            } else {
                add_bcast_rows_init_short(weight_result_cb, bias_cb);
            }
            for (uint32_t col = 0; col < num_tile_cols; col += block_size) {
                const uint32_t n = (col + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col);
                cb_wait_front(bias_cb, col + n);
                cb_wait_front(weight_result_cb, block_size);
                tile_regs_acquire();
                for (uint32_t i = 0; i < n; i++) {
                    if constexpr (per_token_bias != 0) {
                        add_tiles(weight_result_cb, bias_cb, i, col + i, i);
                    } else {
                        add_tiles_bcast_rows(weight_result_cb, bias_cb, i, col + i, i);
                    }
                }
                tile_regs_commit();
                cb_pop_front(weight_result_cb, block_size);
                cb_reserve_back(output_cb, n);
                tile_regs_wait();
                for (uint32_t i = 0; i < n; i++) {
                    pack_tile(i, output_cb);
                }
                tile_regs_release();
                cb_push_back(output_cb, n);
            }
        }

        cb_pop_front(mean_cb, 1);
        cb_pop_front(invstd_cb, 1);
        cb_pop_front(input_cb, num_tile_cols);
    }
}
