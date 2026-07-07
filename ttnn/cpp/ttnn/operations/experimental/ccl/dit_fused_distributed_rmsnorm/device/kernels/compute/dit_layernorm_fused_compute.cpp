// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Fused Wan2.2 distributed LayerNorm compute kernel (Welford) — Phases 2-3.
 *
 * Scope: whole-row norm (no per_head), no RoPE, bf16 input, resident layout.
 * Optional gamma (weight) and beta (bias), broadcast [1,H] or per-token [N,H].
 * TP=1 (is_tp_1): reduce locally, no fabric. TP>1: gather per-shard Welford
 * (mean, var) partials over the fabric ring and merge them (parallel Welford).
 *
 * Per tile-row:
 *   1) PRE (Welford): transpose each of the num_tile_cols input tiles into DST and
 *      fold it into the running per-token mean/M2; welford_finalize_to_row converts
 *      M2 -> per-shard variance, writing mean to row 0 of one DST tile and variance
 *      to row 0 of the next (over the local shard of feat_local = num_tile_cols*32).
 *   2a) is_tp_1: that IS the full mean/var -> transpose to col 0, 1/std = rsqrt(var+eps).
 *   2b) TP>1: push (mean, var) row-0 sticks; the worker writer ring-gathers all
 *       ring_size shards' partials into stats_transposed_gathered_cb (interleaved
 *       [mean_d, var_d]); the equal-count Welford combine (below) merges them into the
 *       global (mean, 1/std); transpose both to col 0.
 *   3) POST: x' = (x - mean) [rotated_input_cb] ; x'' = x' * (1/std) ; [* weight] ;
 *      [+ bias] -> output_cb.
 *
 * Shares the program factory's CBs / reader / writer with the RMSNorm kernel.
 * On is_tp_1/no-RoPE: stats_local_cb holds the per-token mean (col 0),
 * reduce_result_cb holds 1/std (col 0), rotated_input_cb (RoPE-only normally)
 * holds (x - mean). On TP>1: stats_transposed_local_cb carries the local
 * (mean, var) partial, stats_transposed_gathered_cb the gathered ring partials,
 * stats_gathered_cb the merged (mean, 1/std).
 */

#include <cstdint>
#include <array>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/welford.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

void kernel_main() {
    // === Compile-time args (shared list with the RMSNorm kernel; LN reads a subset) ===
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t combine_cb = get_compile_time_arg_val(2);  // stats_gathered_cb -> merged [mean, 1/std]
    constexpr uint32_t mean_cb = get_compile_time_arg_val(1);     // stats_local_cb (idle) -> per-token mean (col 0)
    constexpr uint32_t weight_cb = get_compile_time_arg_val(3);
    constexpr uint32_t invstd_cb = get_compile_time_arg_val(7);        // reduce_result_cb -> 1/std (col 0)
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(8);  // norm / weight result (fp32)
    constexpr uint32_t output_cb = get_compile_time_arg_val(10);
    constexpr uint32_t xmm_cb = get_compile_time_arg_val(14);  // rotated_input_cb (idle, no RoPE) -> (x - mean)
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(15);
    constexpr uint32_t block_size = get_compile_time_arg_val(16);
    constexpr uint32_t ring_size = get_compile_time_arg_val(17);  // stats_tiles_cols (== TP shards)
    constexpr uint32_t has_weight = get_compile_time_arg_val(19);
    constexpr uint32_t is_tp_1 = get_compile_time_arg_val(22);
    constexpr uint32_t stats_local_cb = get_compile_time_arg_val(23);     // local partial (mean, var) row 0
    constexpr uint32_t stats_gathered_cb = get_compile_time_arg_val(24);  // ring partials [mean_d, var_d] row 0
    constexpr uint32_t bias_cb = get_compile_time_arg_val(27);
    constexpr uint32_t has_bias = get_compile_time_arg_val(28);
    constexpr uint32_t per_token_weight = get_compile_time_arg_val(31);
    constexpr uint32_t per_token_bias = get_compile_time_arg_val(32);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(33);
    // Wide-shard layout: streaming_low_l1 streams the input (Welford PRE consumes
    // the reader's 1st pass block-by-block; POST consumes the 2nd pass).
    // block_major_post fuses (x-mean)*1/std*w+b per block so intermediate/output
    // CBs stay O(block_size). For wide LayerNorm the factory sets both together.
    constexpr uint32_t streaming_low_l1 = get_compile_time_arg_val(34);
    constexpr uint32_t block_major_post = get_compile_time_arg_val(36);
    // Reciprocal LUT (CT 39/40): when use_recip_lut, recip_lut_cb holds reduce_width fp32
    // reciprocals [1/1..1/reduce_width] (reader filled it once from DRAM). The Welford LLK
    // does an array load instead of a soft-float 1/(N+1) per sample. Absent -> runtime div.
    constexpr uint32_t recip_lut_cb = get_compile_time_arg_val(38);
    constexpr uint32_t use_recip_lut = get_compile_time_arg_val(39);
    // Zeroed welford-state CB (2 tiles: mean=0, M2=0). Captured once below while the SFPU is clean,
    // reloaded per row to reset the welford accumulator; see the cold-start capture.
    constexpr uint32_t welford_zero_cb = get_compile_time_arg_val(40);
    // Per-batch adaLN (CT 41/42/43): weight/bias is [batch,1,H] — broadcast over seq (so the
    // bcast_rows path applies) but distinct per batch. All batches' rows are resident in
    // weight_cb / bias_cb (batch * num_tile_cols tiles); we offset the tile index by
    // wbatch * num_tile_cols, wbatch = (tile_row_start + row) / rows_per_batch_tiles. For a
    // true-broadcast weight per_batch_* is 0 and the offset collapses to 0 (unchanged).
    constexpr uint32_t per_batch_weight = get_compile_time_arg_val(41);
    constexpr uint32_t per_batch_bias = get_compile_time_arg_val(42);
    constexpr uint32_t rows_per_batch_tiles = get_compile_time_arg_val(43);

    // Welford reduces over the local shard: num_tile_cols * TILE_WIDTH features.
    constexpr uint32_t tile_width = 32u;
    constexpr uint32_t reduce_width = num_tile_cols * tile_width;  // = feat_local
    // Reciprocal-LUT fallback: empty array -> the LLK computes 1/(N+1) by float division.
    constexpr std::array<uint32_t, 0> no_lut{};
    // LUT pointer: typed std::array<uint32_t, reduce_width>* always (NOT gated on
    // use_recip_lut) so the discarded if-constexpr branch below still type-checks — this
    // kernel is a plain function, so both branches are semantically validated. It's only
    // dereferenced (and the CB only filled) when use_recip_lut. Welford reads it via the
    // SFPU as the per-sample 1/(N+1) lookup; see dit_rmsnorm_fused_reader.cpp.
    const std::array<uint32_t, reduce_width>* p_recip = nullptr;
    if constexpr (use_recip_lut != 0) {
        cb_wait_front(recip_lut_cb, 1);
        p_recip = norm::kernel_util::compute::memory::get_pointer_to_cb_data<std::array<uint32_t, reduce_width>>(
            recip_lut_cb, 0);
    }
    // Compile-time dispatch to the LUT or division flavor of each Welford call.
    auto wf_update = [&](uint32_t dst, uint32_t start) {
        if constexpr (use_recip_lut != 0) {
            welford_update<reduce_width>(dst, start, *p_recip);
        } else {
            welford_update<0>(dst, start, no_lut);
        }
    };
    auto wf_finalize = [&](uint32_t mean_d, uint32_t scale) {
        if constexpr (use_recip_lut != 0) {
            welford_finalize_to_row<reduce_width>(mean_d, scale, *p_recip);
        } else {
            welford_finalize_to_row<0>(mean_d, scale, no_lut);
        }
    };

    constexpr uint32_t welford_in_dst = 0;
    constexpr uint32_t mean_dst = 1;
    constexpr uint32_t var_dst = 2;

    // has_bias implies has_weight (enforced in validate).
    constexpr uint32_t norm_result_cb = (has_weight != 0) ? intermediate_cb : output_cb;
    constexpr uint32_t weight_result_cb = (has_bias != 0) ? intermediate_cb : output_cb;

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    // Worker's first GLOBAL tile-row (RT arg 1): maps each local row to its batch for per-batch
    // adaLN. Only consumed when per_batch_weight/bias; harmless otherwise.
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);

    binary_op_init_common(input_cb, input_cb, input_cb);

    // Cold-start capture of a zeroed welford state into welford_zero_cb (mean=0 tile, M2=0 tile),
    // done ONCE here while the SFPU condition code is clean (before any row's combine). Each row's
    // PRE reloads this via copy_tile + welford_restore_state to reset the accumulator on the
    // unpredicated L1->DST path — the standard layernorm_large_tensor_welford fuse_pre_add trick.
    {
        tile_regs_acquire();
        welford_init();
        welford_save_state(mean_dst);  // LREG4/5 (cleared) -> mean_dst / var_dst
        tile_regs_commit();
        cb_reserve_back(welford_zero_cb, 2);
        tile_regs_wait();
        pack_reconfig_data_format(welford_zero_cb);
        pack_tile(mean_dst, welford_zero_cb);
        pack_tile(var_dst, welford_zero_cb);
        tile_regs_release();
        cb_push_back(welford_zero_cb, 2);
    }
    cb_wait_front(welford_zero_cb, 2);  // resident for the whole kernel (never popped)

    for (uint32_t row = 0; row < num_tile_rows; row++) {
        // Per-batch adaLN (per_batch_weight/bias): the reader streams THIS row's batch slice to
        // the front of weight_cb / bias_cb (face-row broadcast), so the compute consumes at col
        // and pops per row — identical to per-token, using mul_bcast_rows. No batch offset here.

        // Resident layout: whole row stays in L1 (Welford PRE + POST re-read).
        // Streaming layout (wide shards): each pass waits/pops per block instead.
        if constexpr (streaming_low_l1 == 0) {
            cb_wait_front(input_cb, num_tile_cols);
        }

        // -------- PHASE 1: PRE — local per-token Welford (mean, var) over the shard --------
        {
            tile_regs_acquire();
            // welford_init() programs the SFPU replay buffer + address mods (needed every row).
            // Its SFPLOADI accumulator clear is UNRELIABLE here: a prior row's combine (SFPU rsqrt
            // with data-dependent predication) can leave the condition code predicated, so the
            // clear skips some token lanes -> stale ~1e36 mean -> M2 overflow -> inf var -> token
            // collapse on warm rows. So reset the accumulator via the UNPREDICATED
            // L1->DST path instead: copy the cold-captured zero state and welford_restore_state
            // (mirrors layernorm_large_tensor_welford's fuse_pre_add reload).
            welford_init();
            reconfig_data_format_srca(welford_zero_cb);
            copy_tile_init(welford_zero_cb);
            copy_tile(welford_zero_cb, 0, mean_dst);
            copy_tile(welford_zero_cb, 1, var_dst);
            welford_restore_state(mean_dst);
            // Reconfigure the unpacker back to the bf16 input for the transpose-fed welford.
            reconfig_data_format_srca(welford_zero_cb, input_cb);
            transpose_wh_init_short(input_cb);
            uint32_t start_n = 0;
            if constexpr (streaming_low_l1 != 0) {
                // Streamed 1st pass: wait + Welford + pop each block; LREG4/5 accumulate
                // across blocks. The reader re-pushes the row for the POST 2nd pass.
                for (uint32_t col = 0; col < num_tile_cols; col += block_size) {
                    cb_wait_front(input_cb, block_size);
                    for (uint32_t i = 0; i < block_size; i++) {
                        transpose_wh_tile(input_cb, i, welford_in_dst);
                        wf_update(welford_in_dst, start_n);
                        start_n += tile_width;
                    }
                    cb_pop_front(input_cb, block_size);
                }
            } else {
                for (uint32_t col = 0; col < num_tile_cols; col++) {
                    transpose_wh_tile(input_cb, col, welford_in_dst);
                    wf_update(welford_in_dst, start_n);
                    start_n += tile_width;
                }
            }
            wf_finalize(mean_dst, reduce_width - 1);  // mean->dst1 row0, var->dst2 row0
            tile_regs_commit();

            if constexpr (is_tp_1 != 0) {
                // Local stat IS the full mean/var: stash row 0 for the transpose below.
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
            } else {
                // Push the local (mean, var) partial (row 0) for the worker writer to
                // ring-gather. stats_local_cb holds [mean, var] for this shard.
                cb_reserve_back(stats_local_cb, 2);
                tile_regs_wait();
                pack_reconfig_data_format(stats_local_cb);
                pack_tile(mean_dst, stats_local_cb);
                pack_tile(var_dst, stats_local_cb);
                tile_regs_release();
                cb_push_back(stats_local_cb, 2);
            }
        }

        // -------- Produce per-token mean (col 0) + 1/std (col 0) into mean_cb / invstd_cb --------
        if constexpr (is_tp_1 != 0) {
            // Transpose row 0 -> col 0; 1/std = rsqrt(var + eps).
            cb_wait_front(mean_cb, 1);
            cb_wait_front(invstd_cb, 1);
            reconfig_data_format_srca(mean_cb);
            transpose_wh_init_short(mean_cb);
            tile_regs_acquire();
            transpose_wh_tile(mean_cb, 0, mean_dst);
            transpose_wh_tile(invstd_cb, 0, var_dst);
            binop_with_scalar_tile_init();
            add_unary_tile(var_dst, eps_bits);
            // legacy rsqrt to match the composite dit_layernorm baseline (it uses
            // rsqrt_tile<true>); the non-legacy default diverges on low-variance rows.
            rsqrt_tile_init<true>();
            rsqrt_tile<true>(var_dst);
            tile_regs_commit();

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
        } else {
            // Equal-count Welford (Chan) combine. All ring_size shards have identical count
            // n_i == reduce_width, so the exact parallel-Welford combine collapses in closed
            // form to:
            //   mean_g = mean(mean_i)
            //   var_g  = mean(var_i) + (1/K) Σ (mean_i - mean_g)^2
            // The between-shard term uses the STABLE deviation form Σ(mean_i - mean_g)^2
            // (squares only the small deviations), not the cancellation-prone
            // mean(mean_i^2) - mean_g^2 -- numerically identical to Welford's pairwise merge.
            // Gathered CB interleaves [mean_0, var_0, ...] (row 0): mean_i at 2i, var_i at 2i+1;
            // output [mean_g, 1/std] (row 0) -> combine_cb for the downstream transpose. legacy
            // rsqrt matches the composite dit_layernorm baseline.
            constexpr uint32_t DM = 0;   // Σ mean_i  -> mean_g
            constexpr uint32_t DV = 1;   // Σ var_i   -> 1/std
            constexpr uint32_t DMM = 2;  // Σ (mean_i - mean_g)^2
            constexpr uint32_t DT = 3;   // scratch
            constexpr uint32_t two_ring = 2u * ring_size;
            constexpr uint32_t recip_k_bits = __builtin_bit_cast(uint32_t, 1.0f / static_cast<float>(ring_size));
            cb_wait_front(stats_gathered_cb, two_ring);
            reconfig_data_format(stats_gathered_cb, stats_gathered_cb);
            pack_reconfig_data_format(combine_cb);
            tile_regs_acquire();
            // Σ mean_i (even tile indices) via FPU pairwise add + dst-accumulate.
            binary_tiles_init<true, EltwiseBinaryType::ELWADD>(stats_gathered_cb, stats_gathered_cb, false);
            add_tiles(stats_gathered_cb, stats_gathered_cb, 0, 2, DM);
            binary_tiles_init<false, EltwiseBinaryType::ELWADD>(stats_gathered_cb, stats_gathered_cb, true);
            for (uint32_t k = 4; k < two_ring; k += 4) {
                add_tiles(stats_gathered_cb, stats_gathered_cb, k, k + 2, DM);
            }
            // Σ var_i (odd tile indices).
            binary_tiles_init<true, EltwiseBinaryType::ELWADD>(stats_gathered_cb, stats_gathered_cb, false);
            add_tiles(stats_gathered_cb, stats_gathered_cb, 1, 3, DV);
            binary_tiles_init<false, EltwiseBinaryType::ELWADD>(stats_gathered_cb, stats_gathered_cb, true);
            for (uint32_t k = 5; k < two_ring; k += 4) {
                add_tiles(stats_gathered_cb, stats_gathered_cb, k, k + 2, DV);
            }
            // mean_g = (1/K) Σ mean_i  (needed before the deviation form below).
            binop_with_scalar_tile_init();
            mul_unary_tile(DM, recip_k_bits);  // DM = mean_g
            // Between-shard term via the STABLE deviation form: Σ (mean_i - mean_g)^2. Squares
            // only the small deviations, so no catastrophic cancellation (unlike Σmean^2-mean_g^2).
            fill_tile_init();
            fill_tile(DMM, 0.f);
            for (uint32_t i = 0; i < ring_size; i++) {
                copy_tile_to_dst_init_short(stats_gathered_cb);
                copy_tile(stats_gathered_cb, 2u * i, DT);  // DT = mean_i
                sub_binary_tile_init();
                sub_binary_tile(DT, DM, DT);  // DT = mean_i - mean_g
                square_tile_init();
                square_tile(DT);  // DT = (mean_i - mean_g)^2
                add_binary_tile_init();
                add_binary_tile(DMM, DT, DMM);  // Σ (mean_i - mean_g)^2
            }
            // var_g = (1/K) Σ var_i + (1/K) Σ (mean_i - mean_g)^2 = (Σvar + Σdev^2) / K
            add_binary_tile_init();
            add_binary_tile(DV, DMM, DV);  // DV = Σvar + Σdev^2
            binop_with_scalar_tile_init();
            mul_unary_tile(DV, recip_k_bits);  // DV = var_g
            add_unary_tile(DV, eps_bits);      // var_g + eps
            rsqrt_tile_init<true>();
            rsqrt_tile<true>(DV);  // DV = 1/std (legacy, matches baseline)
            tile_regs_commit();
            cb_reserve_back(combine_cb, 2);
            tile_regs_wait();
            pack_tile(DM, combine_cb);  // mean_g -> tile 0
            pack_tile(DV, combine_cb);  // 1/std  -> tile 1
            tile_regs_release();
            cb_pop_front(stats_gathered_cb, two_ring);
            cb_push_back(combine_cb, 2);

            // Transpose merged mean / 1/std row 0 -> col 0.
            cb_wait_front(combine_cb, 2);
            reconfig_data_format_srca(combine_cb);
            transpose_wh_init_short(combine_cb);
            tile_regs_acquire();
            transpose_wh_tile(combine_cb, 0, mean_dst);
            transpose_wh_tile(combine_cb, 1, var_dst);
            tile_regs_commit();
            cb_pop_front(combine_cb, 2);
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

        cb_wait_front(mean_cb, 1);
        cb_wait_front(invstd_cb, 1);

        // -------- PHASE 2: POST — (x - mean) * (1/std) [* weight] [+ bias] --------
        if constexpr (block_major_post != 0) {
            // ===== Block-major POST (wide shards) =====
            // Per block: re-read the streamed input (2nd pass) and fuse
            // (x-mean) -> *1/std -> [*weight] -> [+bias] -> output, so xmm /
            // intermediate / output CBs stay O(block_size). mean_cb / invstd_cb
            // (col 0) and weight_cb / bias_cb (whole-row) stay resident.
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                // (x - mean) -> xmm_cb
                cb_wait_front(input_cb, block_size);
                reconfig_data_format(input_cb, mean_cb);
                pack_reconfig_data_format(xmm_cb);
                sub_bcast_cols_init_short(input_cb, mean_cb);
                cb_reserve_back(xmm_cb, block_size);
                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size; i++) {
                    sub_tiles_bcast_cols(input_cb, mean_cb, i, 0, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size; i++) {
                    pack_tile(i, xmm_cb);
                }
                tile_regs_release();
                cb_push_back(xmm_cb, block_size);
                cb_pop_front(input_cb, block_size);

                // (x - mean) * (1/std) -> norm_result_cb
                cb_wait_front(xmm_cb, block_size);
                reconfig_data_format(xmm_cb, invstd_cb);
                pack_reconfig_data_format(norm_result_cb);
                mul_bcast_cols_init_short(xmm_cb, invstd_cb);
                cb_reserve_back(norm_result_cb, block_size);
                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size; i++) {
                    mul_tiles_bcast_cols(xmm_cb, invstd_cb, i, 0, i);
                }
                tile_regs_commit();
                cb_pop_front(xmm_cb, block_size);
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size; i++) {
                    pack_tile(i, norm_result_cb);
                }
                tile_regs_release();
                cb_push_back(norm_result_cb, block_size);

                // * weight
                if constexpr (has_weight != 0) {
                    cb_wait_front(weight_cb, col_tile + block_size);
                    cb_wait_front(norm_result_cb, block_size);
                    reconfig_data_format(norm_result_cb, weight_cb);
                    pack_reconfig_data_format(weight_result_cb);
                    if constexpr (per_token_weight != 0) {
                        mul_tiles_init(norm_result_cb, weight_cb);
                    } else {
                        mul_bcast_rows_init_short(norm_result_cb, weight_cb);
                    }
                    cb_reserve_back(weight_result_cb, block_size);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < block_size; i++) {
                        if constexpr (per_token_weight != 0) {
                            mul_tiles(norm_result_cb, weight_cb, i, col_tile + i, i);
                        } else {
                            mul_tiles_bcast_rows(norm_result_cb, weight_cb, i, col_tile + i, i);
                        }
                    }
                    tile_regs_commit();
                    cb_pop_front(norm_result_cb, block_size);
                    tile_regs_wait();
                    for (uint32_t i = 0; i < block_size; i++) {
                        pack_tile(i, weight_result_cb);
                    }
                    tile_regs_release();
                    cb_push_back(weight_result_cb, block_size);
                }

                // + bias
                if constexpr (has_bias != 0) {
                    cb_wait_front(bias_cb, col_tile + block_size);
                    cb_wait_front(weight_result_cb, block_size);
                    reconfig_data_format(weight_result_cb, bias_cb);
                    pack_reconfig_data_format(output_cb);
                    if constexpr (per_token_bias != 0) {
                        add_tiles_init(weight_result_cb, bias_cb);
                    } else {
                        add_bcast_rows_init_short(weight_result_cb, bias_cb);
                    }
                    cb_reserve_back(output_cb, block_size);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < block_size; i++) {
                        if constexpr (per_token_bias != 0) {
                            add_tiles(weight_result_cb, bias_cb, i, col_tile + i, i);
                        } else {
                            add_tiles_bcast_rows(weight_result_cb, bias_cb, i, col_tile + i, i);
                        }
                    }
                    tile_regs_commit();
                    cb_pop_front(weight_result_cb, block_size);
                    tile_regs_wait();
                    for (uint32_t i = 0; i < block_size; i++) {
                        pack_tile(i, output_cb);
                    }
                    tile_regs_release();
                    cb_push_back(output_cb, block_size);
                }
            }
            // Per-row affine (per-token OR per-batch adaLN): the reader pushes row r's slice; pop
            // it so row r+1's slice is at the front next iteration (block-major runs only for
            // divisible num_tile_cols, so the whole row's num_tile_cols was consumed above).
            // Broadcast weight/bias stays resident (never popped).
            if constexpr (per_token_weight != 0 || per_batch_weight != 0) {
                cb_pop_front(weight_cb, num_tile_cols);
            }
            if constexpr (per_token_bias != 0 || per_batch_bias != 0) {
                cb_pop_front(bias_cb, num_tile_cols);
            }
        } else {
            // ===== Resident POST (shard fits L1) =====
            // The intermediate CBs (xmm / norm_result / weight_result / output) are pushed
            // and consumed in whole block_size blocks — the last block is padded when
            // num_tile_cols % block_size != 0 (only the n valid tiles are computed; the pad
            // slot is stale and dropped by the writer, which also drains in block_size units).
            // This keeps the resident POST + the worker writer divisibility-agnostic. The
            // whole-row weight_cb / bias_cb instead hold exactly num_tile_cols and use a
            // cumulative (col + n) wait. CBs are sized div_up(num_tile_cols, block_size).
            // Sub-phase A: x - mean (broadcast mean over feature cols) -> xmm_cb.
            {
                reconfig_data_format(input_cb, mean_cb);
                pack_reconfig_data_format(xmm_cb);
                sub_bcast_cols_init_short(input_cb, mean_cb);
                for (uint32_t col = 0; col < num_tile_cols; col += block_size) {
                    const uint32_t n = (col + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col);
                    cb_reserve_back(xmm_cb, block_size);
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
                    cb_push_back(xmm_cb, block_size);
                }
            }

            // Sub-phase B: (x - mean) * (1/std) -> norm_result_cb.
            {
                reconfig_data_format(xmm_cb, invstd_cb);
                pack_reconfig_data_format(norm_result_cb);
                mul_bcast_cols_init_short(xmm_cb, invstd_cb);
                for (uint32_t col = 0; col < num_tile_cols; col += block_size) {
                    const uint32_t n = (col + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col);
                    cb_wait_front(xmm_cb, block_size);
                    cb_reserve_back(norm_result_cb, block_size);
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
                    cb_push_back(norm_result_cb, block_size);
                    cb_pop_front(xmm_cb, block_size);
                }
            }

            // Sub-phase C: * weight (gamma).
            if constexpr (has_weight != 0) {
                reconfig_data_format(norm_result_cb, weight_cb);
                pack_reconfig_data_format(weight_result_cb);
                if constexpr (per_token_weight != 0) {
                    mul_tiles_init(norm_result_cb, weight_cb);
                } else {
                    mul_bcast_rows_init_short(norm_result_cb, weight_cb);
                }
                for (uint32_t col = 0; col < num_tile_cols; col += block_size) {
                    const uint32_t n = (col + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col);
                    // whole-row weight: cumulative wait; per-batch offsets into this batch's
                    // resident num_tile_cols slice (w_off==0 for broadcast/per-token).
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
                    cb_reserve_back(weight_result_cb, block_size);
                    tile_regs_wait();
                    for (uint32_t i = 0; i < n; i++) {
                        pack_tile(i, weight_result_cb);
                    }
                    tile_regs_release();
                    cb_push_back(weight_result_cb, block_size);
                }
                // Per-row affine (per-token OR per-batch adaLN): weight is pushed fresh per row by
                // the reader (row r's slice at the front); pop this row's num_tile_cols so row
                // r+1's slice is at the front next iteration. Broadcast weight stays resident.
                if constexpr (per_token_weight != 0 || per_batch_weight != 0) {
                    cb_pop_front(weight_cb, num_tile_cols);
                }
            }

            // Sub-phase D: + bias (beta).
            if constexpr (has_bias != 0) {
                reconfig_data_format(weight_result_cb, bias_cb);
                pack_reconfig_data_format(output_cb);
                if constexpr (per_token_bias != 0) {
                    add_tiles_init(weight_result_cb, bias_cb);
                } else {
                    add_bcast_rows_init_short(weight_result_cb, bias_cb);
                }
                for (uint32_t col = 0; col < num_tile_cols; col += block_size) {
                    const uint32_t n = (col + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col);
                    // whole-row bias: cumulative wait; per-batch offsets into this batch's slice.
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
                    cb_reserve_back(output_cb, block_size);
                    tile_regs_wait();
                    for (uint32_t i = 0; i < n; i++) {
                        pack_tile(i, output_cb);
                    }
                    tile_regs_release();
                    cb_push_back(output_cb, block_size);
                }
                // Per-row affine bias (per-token OR per-batch adaLN): pop this row's slice.
                if constexpr (per_token_bias != 0 || per_batch_bias != 0) {
                    cb_pop_front(bias_cb, num_tile_cols);
                }
            }
        }  // end resident POST (else of block_major_post)

        cb_pop_front(mean_cb, 1);
        cb_pop_front(invstd_cb, 1);
        if constexpr (block_major_post == 0) {
            // Resident path holds the row through POST; block-major popped per block.
            cb_pop_front(input_cb, num_tile_cols);
        }
    }
}
