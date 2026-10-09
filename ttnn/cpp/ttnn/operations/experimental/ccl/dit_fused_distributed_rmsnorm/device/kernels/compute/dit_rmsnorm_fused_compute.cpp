// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Fused Wan2.2 distributed RMSNorm compute kernel — CHUNKED.
 *
 * Processes the core's tile-row slice in fixed-size windows ("chunks") of
 * chunk_size_rows. For each chunk we:
 *   1) Wait for the chunk's input tiles in input_cb (reader streams them).
 *   2) PRE: compute sum(x**2) per row, push one stat tile per row to
 *      stats_local_cb (forwarder will ring-gather across TP).
 *   3) Wait for stats_gathered_cb to fill (chunk_size_rows × stats_tiles_cols
 *      tiles per chunk, where stats_tiles_cols == ring_size).
 *   4) POST: reduce gathered stats across TP, eps+rsqrt, multiply input by
 *      1/rms (input still L1-resident in input_cb), optional gamma row-bcast,
 *      optional RoPE; push outputs to output_cb.
 *   5) Pop the chunk's input + gathered stats from their CBs.
 *
 * The chunk window is the L1-residency window: input lives in L1 from step 1
 * through step 4, then is released in step 5. Sizing chunk_size_rows lets the
 * program factory trade off L1 footprint vs AG amortization.
 *
 * For is_tp_1 (ring_size==1 or per_head_norm) there is no all-gather and no
 * forwarder: compute pushes its stats straight into stats_gathered_cb, and the
 * per-row reduce degenerates to a local reduce (stats_tiles_cols==1 for TP=1).
 */

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/layernorm.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    // === Compile-time args ===
    constexpr std::uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr std::uint32_t stats_local_cb = get_compile_time_arg_val(1);
    constexpr std::uint32_t stats_gathered_cb = get_compile_time_arg_val(2);
    constexpr std::uint32_t weight_cb = get_compile_time_arg_val(3);
    constexpr std::uint32_t reduce_scalar_sum_cb = get_compile_time_arg_val(4);
    constexpr std::uint32_t reduce_scalar_avg_cb = get_compile_time_arg_val(5);
    constexpr std::uint32_t epsilon_cb = get_compile_time_arg_val(6);
    constexpr std::uint32_t reduce_result_cb = get_compile_time_arg_val(7);
    constexpr std::uint32_t intermediate_cb = get_compile_time_arg_val(8);
    constexpr std::uint32_t pre_intermediate_cb = get_compile_time_arg_val(9);
    constexpr std::uint32_t output_cb = get_compile_time_arg_val(10);
    constexpr std::uint32_t transformation_mat_cb = get_compile_time_arg_val(11);
    constexpr std::uint32_t rope_cos_cb = get_compile_time_arg_val(12);
    constexpr std::uint32_t rope_sin_cb = get_compile_time_arg_val(13);
    constexpr std::uint32_t rotated_input_cb = get_compile_time_arg_val(14);
    constexpr std::uint32_t num_tile_cols = get_compile_time_arg_val(15);
    constexpr std::uint32_t block_size = get_compile_time_arg_val(16);
    constexpr std::uint32_t stats_tiles_cols = get_compile_time_arg_val(17);
    constexpr std::uint32_t chunk_size_rows = 1u;  // chunk size is always 1 (no CT arg)
    constexpr std::uint32_t has_weight = get_compile_time_arg_val(18);
    constexpr std::uint32_t fuse_rope = get_compile_time_arg_val(19);
    constexpr std::uint32_t head_dim_tiles = get_compile_time_arg_val(20);
    // When is_tp_1 is true, the compute kernel skips stats_local_cb entirely
    // and pushes per-row stats directly into stats_gathered_cb. This makes
    // TP=1 (single-device) operation self-contained — no forwarder needed.
    constexpr std::uint32_t is_tp_1 = get_compile_time_arg_val(21);
    // Packed-page all-gather (the forwarder AG path). When enabled the pre phase
    // transposes the per-row stat tile (real data in col 0 → row 0) so the worker
    // can extract two contiguous 64-byte spans per tile and the forwarder packs
    // `window_size` of them into a single fabric packet. The post phase then
    // transposes the row-0 gathered tiles back to col 0 before the
    // existing reduce<AVG,REDUCE_ROW> chain runs.
    constexpr std::uint32_t stats_transposed_local_cb = get_compile_time_arg_val(22);
    constexpr std::uint32_t stats_transposed_gathered_cb = get_compile_time_arg_val(23);
    constexpr std::uint32_t packed_ag_enabled = get_compile_time_arg_val(24);
    // Per-head RoPE: reader pushes num_tile_cols (num_heads * head_dim_tiles)
    // cos/sin tiles per row (one per col). The post phase reads them by
    // absolute index (col_tile + i) — no wrap-cycling. Broadcast RoPE
    // (per_head_rope=0) pushes only head_dim_tiles and wraps.
    constexpr std::uint32_t per_head_rope = get_compile_time_arg_val(25);
    // Optional row-broadcast bias: tilized [1, H] tensor added after weight
    // multiply (sub-phase 2.5). Mirrors weight handling.
    constexpr std::uint32_t bias_cb = get_compile_time_arg_val(26);
    constexpr std::uint32_t has_bias = get_compile_time_arg_val(27);
    // Per-head normalization (FLUX.2): reduce over head_dim per head instead
    // of the full row. Forces is_tp_1 path (skip AG entirely); pre phase
    // produces num_heads_per_device stat tiles per row, post phase computes
    // an rsqrt per head and applies it to that head's columns.
    constexpr std::uint32_t per_head_norm = get_compile_time_arg_val(28);
    constexpr std::uint32_t num_heads_per_device = get_compile_time_arg_val(29);
    // Per-token weight / bias: [N, H] (vs broadcast [1, H]). Reader pushes
    // per-row tiles. Compute uses mul_tiles / add_tiles (no _bcast_rows) and
    // pops the row's weight/bias tiles at end of each row in the chunk.
    constexpr std::uint32_t per_token_weight = get_compile_time_arg_val(30);
    constexpr std::uint32_t per_token_bias = get_compile_time_arg_val(31);
    // Bit-packed fp32 epsilon for the fused +eps SFPU scalar-add in the reduce
    // post-op (replaces the prior bf16 epsilon_cb add_tiles path; fp32 is at
    // least as precise).
    constexpr std::uint32_t eps_bits = get_compile_time_arg_val(32);
    // Streaming low-L1: input_cb is block-sized, so PRE reads input block by
    // block (popping each) and POST sub-phase 1 (x*1/rms) re-reads a second
    // streamed pass from the reader (also popping each block). Only the whole-
    // row reduce path is supported (per_head_norm stays resident). The math /
    // accumulation order is identical to the resident path, so the result is
    // bit-exact; only the input_cb residency window changes.
    constexpr std::uint32_t streaming_low_l1 = get_compile_time_arg_val(33);
    // Block-major POST: fuse the matmul rotate + RoPE finalize into one per-block
    // loop so rotated_input_cb is block-local (host shrinks it when the whole-row
    // resident POST would overflow L1 at wide per-head shards). Adds per-block
    // reconfigs (slower) but fits L1; only set for the OOMing config.
    constexpr std::uint32_t fuse_mm_rope = get_compile_time_arg_val(34);
    // Full block-major POST: fuse ALL post sub-phases (x*1/rms, weight, bias,
    // matmul-rotate, RoPE) into ONE per-block loop so intermediate_cb /
    // rotated_input_cb / output_cb are block-local (O(block_size)). Engaged by the
    // host only when even the input-streamed layout overflows L1 (wide low-TP
    // shards: TP=1 WAN/FLUX/LTX-video, TP=2 FLUX). Implies streaming_low_l1 and
    // per_head_norm==0. Bit-exact with the resident path (same math + order, fp32
    // intermediates), trading per-block reconfigs for a bounded L1 footprint.
    constexpr std::uint32_t block_major_post = get_compile_time_arg_val(35);
    // Normalization variant: 0 = RMSNorm (PRE sum-of-squares, POST x*rsqrt(E[x^2]+eps)),
    // 1 = Welford LayerNorm (PRE per-shard Welford (mean, M2), POST merge +
    // (x-mean)*rsqrt(var+eps)). Wired in Phase 1; the LayerNorm code paths land in
    // later phases. Until then only RMS is exercised, so this stays inert.
    constexpr std::uint32_t norm_type = get_compile_time_arg_val(36);
    static_assert(norm_type == 0u, "Welford LayerNorm (norm_type=1) is not yet implemented in the compute kernel");

    constexpr std::uint32_t stats_dest_cb = (is_tp_1 != 0) ? stats_gathered_cb : stats_local_cb;
    // Per-row post reduce reads ring_size tiles. With packed AG enabled the
    // ring_size tiles live in stats_transposed_gathered_cb (post-transpose);
    // otherwise (is_tp_1) the local reduce uses stats_gathered_cb directly.
    constexpr std::uint32_t stats_reduce_src_cb =
        (packed_ag_enabled != 0) ? stats_transposed_gathered_cb : stats_gathered_cb;

    // Resident whole-row packed-AG PRE: turn S = sum_k x_k*x_k (DST-accumulated, packed once) into the
    // ROW-0 stick layout with ONE matmul C = ones * S^T, where ones = the SUM reduce-scalar tile (1.0 in
    // row 0 of every face), so C[0][i] = sum_j S[i][j]. That replaces reduce<SUM,REDUCE_ROW> (col 0) plus
    // the col-0 -> row-0 transpose: one unpack/math/pack round trip instead of two on the AG-start path.
    // S is unpacked to tf32 in SrcA exactly as the reduce did, so the stat precision is unchanged.
    constexpr bool mm_row_stat = (packed_ag_enabled != 0) && (streaming_low_l1 == 0) && (per_head_norm == 0);

    const std::uint32_t num_tile_rows = get_arg_val<std::uint32_t>(0);

    // Two-wave column split: the writer lands one pair read per device in the gathered CB, whose pages are
    // 64 B. Partial t (device t / 2, column half t % 2) is the fp32 tile whose row 0 starts at page
    // (t / 2) * gather_pair_pages + t % 2; the rest of that tile view is other slots' data, unused.
    constexpr std::uint32_t col_split = get_compile_time_arg_val(43);
    constexpr std::uint32_t gathered_cb_pages = get_compile_time_arg_val(44);
    constexpr std::uint32_t gather_pair_pages = get_compile_time_arg_val(45);
    constexpr std::uint32_t gathered_wait_pages = (col_split != 0) ? gathered_cb_pages : stats_tiles_cols;
    auto gathered_tile = [](std::uint32_t t) -> std::uint32_t {
        return (col_split != 0) ? (t / 2u) * gather_pair_pages + (t % 2u) : t;
    };

    compute_kernel_hw_startup<SrcOrder::Reverse>(intermediate_cb, transformation_mat_cb, rotated_input_cb);
    matmul_init(intermediate_cb, transformation_mat_cb);
    compute_kernel_hw_startup(input_cb, input_cb, input_cb);

    CircularBuffer cb_input(input_cb);
    CircularBuffer cb_stats_local(stats_local_cb);
    CircularBuffer cb_stats_gathered(stats_gathered_cb);
    CircularBuffer cb_weight(weight_cb);
    CircularBuffer cb_bias(bias_cb);
    CircularBuffer cb_reduce_scalar_sum(reduce_scalar_sum_cb);
    CircularBuffer cb_reduce_scalar_avg(reduce_scalar_avg_cb);
    CircularBuffer cb_epsilon(epsilon_cb);
    CircularBuffer cb_reduce_result(reduce_result_cb);
    CircularBuffer cb_intermediate(intermediate_cb);
    CircularBuffer cb_pre_intermediate(pre_intermediate_cb);
    CircularBuffer cb_output(output_cb);
    CircularBuffer cb_transformation_mat(transformation_mat_cb);
    CircularBuffer cb_rope_cos(rope_cos_cb);
    CircularBuffer cb_rope_sin(rope_sin_cb);
    CircularBuffer cb_rotated_input(rotated_input_cb);
    CircularBuffer cb_stats_transposed_local(stats_transposed_local_cb);
    CircularBuffer cb_stats_transposed_gathered(stats_transposed_gathered_cb);
    // Aliases the packed-AG or the plain gathered CB, per stats_reduce_src_cb above.
    CircularBuffer cb_stats_reduce_src(stats_reduce_src_cb);

    // One-time waits for reader-produced singletons.
    cb_reduce_scalar_sum.wait_front(1);
    cb_reduce_scalar_avg.wait_front(1);
    cb_epsilon.wait_front(1);
    if constexpr (fuse_rope) {
        cb_transformation_mat.wait_front(1);
    }

    constexpr std::uint32_t mul_rms_result_cb = (fuse_rope || has_weight) ? intermediate_cb : output_cb;
    // has_bias implies has_weight (enforced in validate). Weight output stays
    // in intermediate when bias or rope follows.
    constexpr std::uint32_t mul_weight_result_cb = (fuse_rope || has_bias) ? intermediate_cb : output_cb;
    constexpr std::uint32_t add_bias_result_cb = fuse_rope ? intermediate_cb : output_cb;

    CircularBuffer cb_mul_rms_result(mul_rms_result_cb);
    CircularBuffer cb_mul_weight_result(mul_weight_result_cb);
    CircularBuffer cb_add_bias_result(add_bias_result_cb);

    // Pre-scale by gamma under the AG wait: x * gamma doesn't depend on the gathered
    // stats, so for the plain broadcast-weight path compute it into intermediate_cb
    // (fp32, whole padded row) right after the local stat is pushed, while the
    // forwarder ring-gathers. POST then collapses to ONE pass:
    // out = (x * gamma) * bcast_col(1/rms), replacing sub-phases 1 (x * 1/rms) and 2 (* gamma).
    constexpr bool prescale_weight = (has_weight != 0) && (per_token_weight == 0) && (has_bias == 0) &&
                                     (fuse_rope == 0) && (per_head_norm == 0) && (streaming_low_l1 == 0) &&
                                     (block_major_post == 0) && (packed_ag_enabled != 0);

    // Process the core's tile rows one at a time (chunk size is always 1).
    // per_head_norm produces num_heads_per_device stats per row (one per
    // head) instead of stats_tiles_cols (== ring_size) for the AG path.
    constexpr std::uint32_t per_row_stats_count = (per_head_norm != 0) ? num_heads_per_device : stats_tiles_cols;
    for (std::uint32_t row_processed = 0; row_processed < num_tile_rows; ++row_processed) {
        const std::uint32_t chunk_input_tiles = num_tile_cols;
        const std::uint32_t chunk_stats_tiles = per_row_stats_count;

        // -------- PHASE 1: PRE — sum(x**2) per row --------
        // Cumulative input wait: instead of one wait_front for the whole
        // chunk, wait per col-block. Lets the reader push block N+1 while
        // compute is processing block N. Counter resets per chunk because we
        // cb_input.pop_front(chunk_input_tiles) at the end.
        // Per-head norm: inner reduce spans head_dim_tiles columns per head;
        // we run num_heads_per_device reduces per row. Whole-row norm: single
        // reduce over num_tile_cols per row.
        constexpr std::uint32_t pre_groups_per_row = (per_head_norm != 0) ? num_heads_per_device : 1u;
        constexpr std::uint32_t pre_group_width = (per_head_norm != 0) ? head_dim_tiles : num_tile_cols;
        {
            // PERF NOTE (RMS_PRE = ~29% of the compute floor, ALL shapes): this phase
            // is sum(x^2) per row = mul_tiles(x,x) (num_tile_cols FPU muls) then
            // reduce<SUM,REDUCE_ROW>. Profiling flagged it as the top universal cost,
            // but there is no cheaper path with current LLKs: the square is inherent
            // (mul_tiles is the minimal square) and can't fold into the reduce matmul
            // (which is linear, input * ones-scalar). A real speedup needs a NEW LLK
            // that squares in the unpacker/math before the row-reduce (a "reduce of
            // squares" primitive) -- kernel-dev scope, deferred. Init-hoisting was
            // tried and does NOT help (the per-row reduce clobbers the math config, so
            // mul_init must re-run each row; it's FPU-throughput-bound, not
            // init-bound).
            if constexpr (streaming_low_l1) {
                // Streamed whole-row PRE (one group). Input
                // arrives block by block from the reader's first pass; each
                // block is popped after its x**2 is accumulated into
                // pre_intermediate_cb[0]. The accumulation order (and l1_acc
                // sequencing) is identical to the resident path, so the stat
                // tile is bit-exact — only the input residency window differs.
                {  // single row per iteration (chunk size is 1)
                    reconfig_data_format(input_cb, input_cb);
                    pack_reconfig_data_format(pre_intermediate_cb);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    mul_init(input_cb, input_cb);

                    cb_pre_intermediate.reserve_back(1);
                    for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                        const std::uint32_t tiles_in_block =
                            ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                        cb_input.wait_front(tiles_in_block);
                        tile_regs_acquire();
                        for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                            mul_tiles(input_cb, input_cb, i, i, i);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                            pack_tile<true>(i, pre_intermediate_cb, 0);
                            if (col_tile == 0 && i == 0) {
                                PACK((llk_pack_reconfig_l1_acc(1)));
                            }
                        }
                        tile_regs_release();
                        cb_input.pop_front(tiles_in_block);
                    }
                    cb_pre_intermediate.push_back(1);
                    PACK((llk_pack_reconfig_l1_acc(0)));

                    compute_kernel_lib::reduce<
                        PoolType::SUM,
                        ReduceDim::REDUCE_ROW,
                        pre_intermediate_cb,
                        reduce_scalar_sum_cb,
                        stats_dest_cb>(compute_kernel_lib::ReduceInputBlockShape::single());
                }
            } else {
                std::uint32_t input_tiles_waited = 0;
                {  // single row per iteration (chunk size is 1)
                    constexpr std::uint32_t row_base = 0u;

                    for (std::uint32_t g = 0; g < pre_groups_per_row; g++) {
                        const std::uint32_t group_base = row_base + g * pre_group_width;

                        reconfig_data_format(input_cb, input_cb);
                        pack_reconfig_data_format(pre_intermediate_cb);
                        PACK((llk_pack_reconfig_l1_acc(0)));
                        mul_init(input_cb, input_cb);

                        cb_pre_intermediate.reserve_back(1);

                        // Accumulate sum(x**2) for the whole group in DST tile 0: ELWMUL always
                        // accumulates onto Dst (Dst += SrcA*SrcB; DST is zeroed on release), so
                        // every mul_tiles into dst 0 adds its square to the running fp32 sum. One
                        // pack per group instead of one L1-accumulating fp32 pack per input tile
                        // (the packer was the PRE bottleneck, ~125 ns/tile). DST is held across
                        // the cumulative input waits; nothing upstream depends on compute here.
                        tile_regs_acquire();
                        for (std::uint32_t col_tile = 0; col_tile < pre_group_width; col_tile += block_size) {
                            const std::uint32_t tiles_in_block = ((pre_group_width - col_tile) >= block_size)
                                                                     ? block_size
                                                                     : (pre_group_width - col_tile);
                            // Cumulative wait covers the absolute tile range we need:
                            // r*num_tile_cols + g*pre_group_width + col_tile + tiles_in_block.
                            // Reader pushes block_size at a time across the whole row, so
                            // a wait for fewer-than-block_size tiles is satisfied by the
                            // next reader push regardless.
                            const std::uint32_t need = group_base + col_tile + tiles_in_block;
                            if (need > input_tiles_waited) {
                                cb_input.wait_front(need);
                                input_tiles_waited = need;
                            }

                            for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                const std::uint32_t abs_idx = group_base + col_tile + i;
                                mul_tiles(input_cb, input_cb, abs_idx, abs_idx, 0);
                            }
                        }
                        tile_regs_commit();

                        tile_regs_wait();
                        pack_tile(0, pre_intermediate_cb);
                        tile_regs_release();
                        cb_pre_intermediate.push_back(1);

                        if constexpr (mm_row_stat) {
                            // Row sums straight into row 0 (see mm_row_stat): matmul transposes in1 (S).
                            reconfig_data_format(pre_intermediate_cb, reduce_scalar_sum_cb);
                            pack_reconfig_data_format(stats_transposed_local_cb);
                            matmul_init(reduce_scalar_sum_cb, pre_intermediate_cb, /*transpose=*/1);
                            cb_pre_intermediate.wait_front(1);
                            cb_stats_transposed_local.reserve_back(1);
                            tile_regs_acquire();
                            matmul_tiles(reduce_scalar_sum_cb, pre_intermediate_cb, 0, 0, 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_tile(0, stats_transposed_local_cb);
                            tile_regs_release();
                            cb_stats_transposed_local.push_back(1);
                            cb_pre_intermediate.pop_front(1);
                        } else {
                            // Row/head reduce → 1 stat tile. SUM (col 0 = sum). Post phase
                            // divides by H_full or head_dim via the AVG scalar.
                            compute_kernel_lib::reduce<
                                PoolType::SUM,
                                ReduceDim::REDUCE_ROW,
                                pre_intermediate_cb,
                                reduce_scalar_sum_cb,
                                stats_dest_cb>(compute_kernel_lib::ReduceInputBlockShape::single());
                        }
                    }
                }
            }
        }  // RMS_PRE

        // Transpose each row's stat tile from COL 0 -> ROW 0 so the writer packs
        // two contiguous 64 B face-rows (tile byte offsets {0, 1024}) instead of
        // 32 strided fp32 col-0 loads. transpose_wh maps col 0 (face_00 col0 +
        // face_10 col0) -> row 0 (face_00 row0 + face_01 row0). Packed-AG (all-gather)
        // path only; is_tp_1 keeps col 0 and reduces locally (no forwarder involved).
        if constexpr (packed_ag_enabled != 0 && !mm_row_stat) {
            transpose_init(stats_local_cb);
            pack_reconfig_data_format(stats_transposed_local_cb);
            {  // single row per iteration (chunk size is 1)
                cb_stats_local.wait_front(1);
                cb_stats_transposed_local.reserve_back(1);
                tile_regs_acquire();
                transpose_tile(stats_local_cb, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, stats_transposed_local_cb);
                tile_regs_release();
                cb_stats_transposed_local.push_back(1);
                cb_stats_local.pop_front(1);
            }
        }

        // -------- x * gamma → intermediate_cb (overlaps the AG) --------
        if constexpr (prescale_weight) {
            reconfig_data_format(input_cb, weight_cb);
            pack_reconfig_data_format(intermediate_cb);
            mul_bcast_rows_init(input_cb, weight_cb);
            for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const std::uint32_t tiles_in_block =
                    (col_tile + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col_tile);
                cb_weight.wait_front(col_tile + tiles_in_block);
                cb_intermediate.reserve_back(block_size);
                tile_regs_acquire();
                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                    mul_tiles_bcast_rows(input_cb, weight_cb, col_tile + i, col_tile + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                    pack_tile(i, intermediate_cb);
                }
                tile_regs_release();
                cb_intermediate.push_back(block_size);
            }
        }

        // -------- WAIT FOR FORWARDER TO COMPLETE AG FOR THIS CHUNK --------
        {
            // Packed-AG path: the worker writer lands the ring gather in row-0 of the
            // transposed gathered CB; is_tp_1 fills the plain col-0 gathered CB locally.
            cb_stats_reduce_src.wait_front(chunk_stats_tiles);
        }

        // -------- PHASE 3: POST — finalize normalization --------
        {
            {  // single row per iteration (chunk size is 1)
                constexpr std::uint32_t row_base = 0u;
                // Single shared cos/sin tile cursor: in broadcast RoPE the cos
                // and sin sequences cycle identically over head_dim_tiles, and
                // the fused finalize multiplies both with the same index.
                std::uint32_t rope_cos_tile_in_head = 0;

                // Per-head norm: do reduce + eps+rsqrt + sub-phase 1 per head, so
                // each head's rsqrt only stays in reduce_result_cb long enough to
                // be consumed by that head's mul. Whole-row norm: one iteration.
                constexpr std::uint32_t post_groups_per_row = (per_head_norm != 0) ? num_heads_per_device : 1u;
                constexpr std::uint32_t post_group_width = (per_head_norm != 0) ? head_dim_tiles : num_tile_cols;
                {
                    for (std::uint32_t g = 0; g < post_groups_per_row; g++) {
                        const std::uint32_t group_col_base = g * post_group_width;
                        const std::uint32_t group_abs_base = row_base + group_col_base;

                        // Per-head / is_tp_1 path: the stat is already in COL 0 of
                        // stats_gathered_cb (compute pushed it locally — no AG), so
                        // reduce<AVG,REDUCE_ROW> over the row of ring_size tiles runs on
                        // stats_gathered_cb unchanged. For the per-head path each head's
                        // stat is a single tile.
                        // Fused reduce + eps + rsqrt: the post_reduce_op runs on DST
                        // (after reduce math, before pack) so we drop a separate
                        // tile_regs cycle, a reduce_result_cb round-trip, and the two
                        // eps reconfigs per row. fp32 scalar eps >= the prior bf16
                        // epsilon_cb add in precision, and rsqrt sees the un-truncated
                        // fp32 mean still in DST.
                        auto eps_rsqrt = [](std::uint32_t dst_idx) {
                            binop_with_scalar_tile_init();
                            add_unary_tile(dst_idx, eps_bits);
                            rsqrt_tile_init();
                            rsqrt_tile(dst_idx);
                        };
                        {
                            if constexpr (per_head_norm != 0) {
                                compute_kernel_lib::reduce<
                                    PoolType::AVG,
                                    ReduceDim::REDUCE_ROW,
                                    stats_gathered_cb,
                                    reduce_scalar_avg_cb,
                                    reduce_result_cb>(
                                    compute_kernel_lib::ReduceInputBlockShape::single(),
                                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                                    compute_kernel_lib::NoAccumulation{},
                                    eps_rsqrt);
                            } else if constexpr (stats_tiles_cols > 1) {
                                // Sum the ring_size gathered partial-sum tiles with an FPU
                                // eltwise add (dst-accumulate) instead of the matmul-based
                                // reduce<AVG,REDUCE_ROW>. Each gathered tile holds this device's
                                // per-token partial sum-of-squares in col 0, so adding the tiles
                                // element-wise gives the full per-token sum in col 0; then
                                // *1/H_full + eps + rsqrt on dst (mirrors the eps_rsqrt post-op).
                                // Removing the matmul eliminates the matmul->pack transition that
                                // wedges the packer at multi-row-chunk x ring_size>1.
                                // stats_tiles_cols == ring_size == TP is always even (2/4/8).
                                // NB: this branch is `if constexpr (stats_tiles_cols > 1)`, but the
                                // kernel is a plain function (constexpr CT args, not a template), so the
                                // discarded branch is still semantically checked — hence the `== 1` exempt
                                // so the TP=1 (stats_tiles_cols==1) build, which takes the else-branch
                                // matmul reduce below, compiles. The even-ness guard still protects the
                                // pairwise add_tiles(...,k,k+1,...) loop from an OOB read at odd ring_size>1.
                                static_assert(
                                    stats_tiles_cols == 1 || stats_tiles_cols % 2 == 0,
                                    "eltwise stats-sum needs even ring_size");
                                constexpr std::uint32_t recip_h_full_bits = __builtin_bit_cast(
                                    std::uint32_t, 1.0f / static_cast<float>(num_tile_cols * 32u * stats_tiles_cols));
                                if constexpr (packed_ag_enabled != 0) {
                                    // All-gather path: the worker writer lands each device's stats in
                                    // ROW 0 of stats_transposed_gathered_cb (two contiguous 64 B face-rows).
                                    // FPU-add the ring_size row-0 tiles, then transpose the summed
                                    // tile IN DST (row 0 -> col 0) with transpose_dest — no CB
                                    // round-trip — then *1/H + eps + rsqrt on col 0. One transpose
                                    // total (deferred past the sum) vs one per gathered tile.
                                    cb_stats_transposed_gathered.wait_front(gathered_wait_pages);
                                    DeviceZoneScopedN("C_COMB");
                                    reconfig_data_format(stats_transposed_gathered_cb, stats_transposed_gathered_cb);
                                    pack_reconfig_data_format(reduce_result_cb);
                                    tile_regs_acquire();
                                    binary_tiles_init<true, EltwiseBinaryType::ELWADD>(
                                        stats_transposed_gathered_cb, stats_transposed_gathered_cb, false);
                                    add_tiles(
                                        stats_transposed_gathered_cb,
                                        stats_transposed_gathered_cb,
                                        gathered_tile(0),
                                        gathered_tile(1),
                                        0);
                                    binary_tiles_init<false, EltwiseBinaryType::ELWADD>(
                                        stats_transposed_gathered_cb, stats_transposed_gathered_cb, true);
                                    for (std::uint32_t k = 2; k < stats_tiles_cols; k += 2) {
                                        add_tiles(
                                            stats_transposed_gathered_cb,
                                            stats_transposed_gathered_cb,
                                            gathered_tile(k),
                                            gathered_tile(k + 1),
                                            0);
                                    }
#if defined(ARCH_BLACKHOLE)
                                    // 1/rms = rsqrt(sum * 1/H + eps) on ROW 0 only, before the transpose: the 32
                                    // per-token sums are tile row 0 (faces 0/1). One BH SFPLOAD covers 4 rows x 8
                                    // cols of one column parity, so ITERATIONS=2 per face = rows 0-3, all columns;
                                    // VectorMode::R = faces 0 and 1. 4 SFPU iterations of one fused MAD + rsqrt
                                    // instead of 3 full-tile (96-iteration) passes on the post-AG critical path.
                                    add_rsqrt_tile_init();
                                    add_rsqrt_tile<false, VectorMode::R, 2, false, recip_h_full_bits>(0, eps_bits);
                                    // row-0 1/rms -> col-0, in place (fp32 DST).
                                    transpose_dest_init<true>(stats_transposed_gathered_cb);
                                    transpose_dest<true>(0);
#else
                                    // row-0 sum -> col-0, in place (fp32 DST).
                                    transpose_dest_init<true>(stats_transposed_gathered_cb);
                                    transpose_dest<true>(0);
                                    binop_with_scalar_tile_init();
                                    mul_unary_tile(0, recip_h_full_bits);
                                    add_unary_tile(0, eps_bits);
                                    rsqrt_tile_init();
                                    rsqrt_tile(0);
#endif
                                    tile_regs_commit();
                                    tile_regs_wait();
                                    cb_reduce_result.reserve_back(1);
                                    pack_tile(0, reduce_result_cb);
                                    cb_reduce_result.push_back(1);
                                    tile_regs_release();
                                    cb_stats_transposed_gathered.pop_front(gathered_wait_pages);
                                } else {
                                    // Non-packed path: gathered tiles are in COL 0.
                                    cb_stats_gathered.wait_front(stats_tiles_cols);
                                    reconfig_data_format(stats_gathered_cb, stats_gathered_cb);
                                    pack_reconfig_data_format(reduce_result_cb);
                                    tile_regs_acquire();
                                    // dst[0] = t0 + t1, then dst[0] += t_k + t_{k+1} for the rest.
                                    binary_tiles_init<true, EltwiseBinaryType::ELWADD>(
                                        stats_gathered_cb, stats_gathered_cb, false);
                                    add_tiles(stats_gathered_cb, stats_gathered_cb, 0, 1, 0);
                                    binary_tiles_init<false, EltwiseBinaryType::ELWADD>(
                                        stats_gathered_cb, stats_gathered_cb, true);
                                    for (std::uint32_t k = 2; k < stats_tiles_cols; k += 2) {
                                        add_tiles(stats_gathered_cb, stats_gathered_cb, k, k + 1, 0);
                                    }
                                    // mean = sum / H_full, then + eps, then rsqrt.
                                    binop_with_scalar_tile_init();
                                    mul_unary_tile(0, recip_h_full_bits);
                                    add_unary_tile(0, eps_bits);
                                    rsqrt_tile_init();
                                    rsqrt_tile(0);
                                    tile_regs_commit();
                                    tile_regs_wait();
                                    cb_reduce_result.reserve_back(1);
                                    pack_tile(0, reduce_result_cb);
                                    cb_reduce_result.push_back(1);
                                    tile_regs_release();
                                    cb_stats_gathered.pop_front(stats_tiles_cols);
                                }
                            } else {
                                // TP=1: a single gathered tile; the matmul reduce is fine
                                // (no multi-chunk x ring hang at ring_size==1).
                                compute_kernel_lib::reduce<
                                    PoolType::AVG,
                                    ReduceDim::REDUCE_ROW,
                                    stats_gathered_cb,
                                    reduce_scalar_avg_cb,
                                    reduce_result_cb>(
                                    compute_kernel_lib::ReduceInputBlockShape::row(stats_tiles_cols),
                                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                                    compute_kernel_lib::NoAccumulation{},
                                    eps_rsqrt);
                            }

                            // The prescale POST waits after its unpack reconfig/init (below), so the
                            // unpacker sets up POST while math/pack still run the stat combine.
                            if constexpr (!prescale_weight) {
                                cb_reduce_result.wait_front(1);
                            }
                        }  // P_NRED

                        // block_major_post fuses mul-rms into the single per-block POST loop
                        // below, so the standalone P_NMUL sub-phase is skipped (and
                        // reduce_result_cb stays resident for that loop to consume).
                        if constexpr (prescale_weight) {
                            // ----- Single POST pass: (x * gamma) * (1/rms) → output_cb -----
                            reconfig_data_format(intermediate_cb, reduce_result_cb);
                            pack_reconfig_data_format(output_cb);
                            mul_bcast_cols_init(intermediate_cb, reduce_result_cb);
                            cb_reduce_result.wait_front(1);
                            DeviceZoneScopedN("C_POST");
                            for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                                const std::uint32_t tiles_in_block =
                                    (col_tile + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col_tile);
                                cb_intermediate.wait_front(block_size);
                                cb_output.reserve_back(block_size);
                                tile_regs_acquire();
                                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                    mul_tiles_bcast_cols(intermediate_cb, reduce_result_cb, i, 0, i);
                                }
                                tile_regs_commit();
                                tile_regs_wait();
                                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                    pack_tile(i, output_cb);
                                }
                                tile_regs_release();
                                cb_intermediate.pop_front(block_size);
                                cb_output.push_back(block_size);
                            }
                            cb_reduce_result.pop_front(1);
                        } else if constexpr (!block_major_post) {
                            // ----- Sub-phase 1: x * (1/rms) → mul_rms_result_cb -----
                            reconfig_data_format(input_cb, reduce_result_cb);
                            pack_reconfig_data_format(mul_rms_result_cb);
                            mul_bcast_cols_init(input_cb, reduce_result_cb);
                            if constexpr (streaming_low_l1) {
                                // Streamed POST re-read: reader's 2nd pass pushes the
                                // row's input block by block. Each block is waited at a
                                // block-relative index, multiplied by 1/rms, packed, then
                                // popped — matching the reader's 2nd-pass pushes.
                                // streaming_low_l1 implies per_head_norm==0, so there is a
                                // single group with post_group_width == num_tile_cols, and
                                // num_tile_cols % block_size == 0 (host TT_FATAL invariant).
                                for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                                    cb_input.wait_front(block_size);
                                    cb_mul_rms_result.reserve_back(block_size);
                                    tile_regs_acquire();
                                    for (std::uint32_t i = 0; i < block_size; i++) {
                                        mul_tiles_bcast_cols(input_cb, reduce_result_cb, i, 0, i);
                                    }
                                    tile_regs_commit();
                                    tile_regs_wait();
                                    for (std::uint32_t i = 0; i < block_size; i++) {
                                        pack_tile(i, mul_rms_result_cb);
                                    }
                                    tile_regs_release();
                                    cb_mul_rms_result.push_back(block_size);
                                    cb_input.pop_front(block_size);
                                }
                            } else {
                                for (std::uint32_t col_tile = 0; col_tile < post_group_width; col_tile += block_size) {
                                    // Per_head_norm pushes head_dim_tiles per head (no padding)
                                    // so multiple heads don't blow past intermediate_cb. The
                                    // whole-row path keeps the block_size-padded push so
                                    // downstream sub-phases (still block_size-driven) consume
                                    // matching counts even when num_tile_cols < block_size.
                                    const std::uint32_t tiles_in_block =
                                        (per_head_norm != 0) ? (((post_group_width - col_tile) >= block_size)
                                                                    ? block_size
                                                                    : (post_group_width - col_tile))
                                                             : block_size;
                                    cb_mul_rms_result.reserve_back(tiles_in_block);
                                    tile_regs_acquire();
                                    for (std::uint32_t i = 0; i < block_size && col_tile + i < post_group_width; i++) {
                                        const std::uint32_t abs_idx = group_abs_base + col_tile + i;
                                        mul_tiles_bcast_cols(input_cb, reduce_result_cb, abs_idx, 0, i);
                                    }
                                    tile_regs_commit();
                                    tile_regs_wait();
                                    for (std::uint32_t i = 0; i < block_size && col_tile + i < post_group_width; i++) {
                                        pack_tile(i, mul_rms_result_cb);
                                    }
                                    tile_regs_release();
                                    cb_mul_rms_result.push_back(tiles_in_block);
                                }
                            }

                            cb_reduce_result.pop_front(1);
                        }  // !block_major_post

                        if constexpr (block_major_post && per_head_norm != 0) {
                            // ===== Head-major POST (per_head_norm wide shards) =====
                            // Process THIS head's post_group_width (head_dim_tiles) cols
                            // fully: x*(1/rms_head) -> [weight] -> [bias] -> [matmul-rotate
                            // + RoPE] -> output, with head-local intermediate/rotated/output
                            // CBs. Input stays RESIDENT (read at absolute index, popped at
                            // end-of-chunk). reduce_result holds this head's 1/rms (front),
                            // popped after this head. Broadcast cos/sin are held resident
                            // across heads (cyclic index); popped once after the head loop.
                            for (std::uint32_t col_tile = 0; col_tile < post_group_width; col_tile += block_size) {
                                const std::uint32_t tiles_in_block = ((post_group_width - col_tile) >= block_size)
                                                                         ? block_size
                                                                         : (post_group_width - col_tile);
                                // ---- x * (1/rms_head): resident input cols -> mul_rms_result_cb ----
                                reconfig_data_format(input_cb, reduce_result_cb);
                                pack_reconfig_data_format(mul_rms_result_cb);
                                mul_bcast_cols_init(input_cb, reduce_result_cb);
                                cb_mul_rms_result.reserve_back(block_size);
                                tile_regs_acquire();
                                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                    mul_tiles_bcast_cols(
                                        input_cb, reduce_result_cb, group_abs_base + col_tile + i, 0, i);
                                }
                                tile_regs_commit();
                                tile_regs_wait();
                                for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                    pack_tile(i, mul_rms_result_cb);
                                }
                                tile_regs_release();
                                cb_mul_rms_result.push_back(block_size);

                                // ---- * weight (resident, absolute col index) ----
                                if constexpr (has_weight) {
                                    cb_weight.wait_front(group_abs_base + col_tile + tiles_in_block);
                                    cb_mul_rms_result.wait_front(block_size);
                                    reconfig_data_format(mul_rms_result_cb, weight_cb);
                                    pack_reconfig_data_format(mul_weight_result_cb);
                                    if constexpr (per_token_weight != 0) {
                                        mul_init(mul_rms_result_cb, weight_cb);
                                    } else {
                                        mul_bcast_rows_init(mul_rms_result_cb, weight_cb);
                                    }
                                    cb_mul_weight_result.reserve_back(block_size);
                                    tile_regs_acquire();
                                    for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                        if constexpr (per_token_weight != 0) {
                                            mul_tiles(
                                                mul_rms_result_cb, weight_cb, i, group_abs_base + col_tile + i, i);
                                        } else {
                                            mul_tiles_bcast_rows(
                                                mul_rms_result_cb, weight_cb, i, group_abs_base + col_tile + i, i);
                                        }
                                    }
                                    tile_regs_commit();
                                    cb_mul_rms_result.pop_front(block_size);
                                    tile_regs_wait();
                                    for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                        pack_tile(i, mul_weight_result_cb);
                                    }
                                    tile_regs_release();
                                    cb_mul_weight_result.push_back(block_size);
                                }
                                // ---- + bias ----
                                if constexpr (has_bias) {
                                    cb_bias.wait_front(group_abs_base + col_tile + tiles_in_block);
                                    cb_mul_weight_result.wait_front(block_size);
                                    reconfig_data_format(mul_weight_result_cb, bias_cb);
                                    pack_reconfig_data_format(add_bias_result_cb);
                                    if constexpr (per_token_bias != 0) {
                                        add_init(mul_weight_result_cb, bias_cb);
                                    } else {
                                        add_bcast_rows_init(mul_weight_result_cb, bias_cb);
                                    }
                                    cb_add_bias_result.reserve_back(block_size);
                                    tile_regs_acquire();
                                    for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                        if constexpr (per_token_bias != 0) {
                                            add_tiles(
                                                mul_weight_result_cb, bias_cb, i, group_abs_base + col_tile + i, i);
                                        } else {
                                            add_tiles_bcast_rows(
                                                mul_weight_result_cb, bias_cb, i, group_abs_base + col_tile + i, i);
                                        }
                                    }
                                    tile_regs_commit();
                                    cb_mul_weight_result.pop_front(block_size);
                                    tile_regs_wait();
                                    for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                        pack_tile(i, add_bias_result_cb);
                                    }
                                    tile_regs_release();
                                    cb_add_bias_result.push_back(block_size);
                                }
                                // ---- matmul-rotate + RoPE finalize, else block already output ----
                                if constexpr (fuse_rope) {
                                    reconfig_data_format(transformation_mat_cb, intermediate_cb);
                                    pack_reconfig_data_format(rotated_input_cb);
                                    matmul_block_init(intermediate_cb, transformation_mat_cb, 0, 1, block_size, 1);
                                    cb_intermediate.wait_front(block_size);
                                    cb_rotated_input.reserve_back(block_size);
                                    tile_regs_acquire();
                                    matmul_block(intermediate_cb, transformation_mat_cb, 0, 0, 0, 0, 1, block_size, 1);
                                    tile_regs_commit();
                                    tile_regs_wait();
                                    for (std::uint32_t i = 0; i < block_size; i++) {
                                        pack_tile(i, rotated_input_cb);
                                    }
                                    tile_regs_release();
                                    cb_rotated_input.push_back(block_size);
                                    // cos/sin: per-head RoPE uses this head's absolute cols (popped
                                    // per head); broadcast holds head_dim_tiles resident, indexed
                                    // by the column within the head (popped once after the head loop).
                                    if constexpr (per_head_rope != 0) {
                                        cb_rope_cos.wait_front(group_abs_base + col_tile + tiles_in_block);
                                        cb_rope_sin.wait_front(group_abs_base + col_tile + tiles_in_block);
                                    } else {
                                        cb_rope_cos.wait_front(head_dim_tiles);
                                        cb_rope_sin.wait_front(head_dim_tiles);
                                    }
                                    cb_intermediate.wait_front(block_size);
                                    cb_rotated_input.wait_front(block_size);
                                    cb_output.reserve_back(block_size);
                                    reconfig_data_format(intermediate_cb, rope_cos_cb);
                                    pack_reconfig_data_format(output_cb);
                                    tile_regs_acquire();
                                    binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(
                                        intermediate_cb, rope_cos_cb, false);
                                    for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                        const std::uint32_t ridx = (per_head_rope != 0)
                                                                       ? (group_abs_base + col_tile + i)
                                                                       : ((col_tile + i) % head_dim_tiles);
                                        mul_tiles(intermediate_cb, rope_cos_cb, i, ridx, i);
                                    }
                                    binary_tiles_init<false, EltwiseBinaryType::ELWMUL>(
                                        rotated_input_cb, rope_sin_cb, true);
                                    for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                        const std::uint32_t ridx = (per_head_rope != 0)
                                                                       ? (group_abs_base + col_tile + i)
                                                                       : ((col_tile + i) % head_dim_tiles);
                                        mul_tiles(rotated_input_cb, rope_sin_cb, i, ridx, i);
                                    }
                                    tile_regs_commit();
                                    tile_regs_wait();
                                    for (std::uint32_t i = 0; i < tiles_in_block; i++) {
                                        pack_tile(i, output_cb);
                                    }
                                    tile_regs_release();
                                    cb_output.push_back(block_size);
                                    cb_intermediate.pop_front(block_size);
                                    cb_rotated_input.pop_front(block_size);
                                    if constexpr (per_head_rope != 0) {
                                        cb_rope_cos.pop_front(tiles_in_block);
                                        cb_rope_sin.pop_front(tiles_in_block);
                                    }
                                }
                                // !fuse_rope: the last affine sub-phase already wrote output_cb.
                            }
                            cb_reduce_result.pop_front(1);  // this head's 1/rms
                        }
                    }
                }  // P_NORM

                if constexpr (block_major_post && per_head_norm != 0 && fuse_rope && per_head_rope == 0) {
                    // Broadcast cos/sin were held resident across all heads; drain once.
                    cb_rope_cos.pop_front(head_dim_tiles);
                    cb_rope_sin.pop_front(head_dim_tiles);
                }

                if constexpr (block_major_post && per_head_norm == 0) {
                    // ===== Full block-major POST (wide low-TP shards) =====
                    // streaming_low_l1 + per_head_norm==0 guaranteed: one resident row,
                    // a single group (post_group_width == num_tile_cols), and
                    // num_tile_cols % block_size == 0 (host TT_FATAL invariants). Per
                    // block: x*(1/rms) -> [*weight] -> [+bias] -> [matmul-rotate + RoPE]
                    // / output. Every intermediate CB is block-local (host sized them
                    // O(block_size)). The aliases route the LAST affine sub-phase to
                    // output_cb when !fuse_rope, so the no-rope case needs no extra copy.
                    // reduce_result_cb (1/rms) is still at the front (P_NMUL skipped).
                    std::uint32_t rope_cursor = 0;  // broadcast cos/sin cyclic index (mod head_dim_tiles)
                    for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                        // ---- x * (1/rms): input 2nd-pass block (streamed) -> mul_rms_result_cb ----
                        cb_input.wait_front(block_size);
                        reconfig_data_format(input_cb, reduce_result_cb);
                        pack_reconfig_data_format(mul_rms_result_cb);
                        mul_bcast_cols_init(input_cb, reduce_result_cb);
                        cb_mul_rms_result.reserve_back(block_size);
                        tile_regs_acquire();
                        for (std::uint32_t i = 0; i < block_size; i++) {
                            mul_tiles_bcast_cols(input_cb, reduce_result_cb, i, 0, i);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (std::uint32_t i = 0; i < block_size; i++) {
                            pack_tile(i, mul_rms_result_cb);
                        }
                        tile_regs_release();
                        cb_mul_rms_result.push_back(block_size);
                        cb_input.pop_front(block_size);

                        // ---- * weight ----
                        if constexpr (has_weight) {
                            cb_weight.wait_front(col_tile + block_size);
                            cb_mul_rms_result.wait_front(block_size);
                            reconfig_data_format(mul_rms_result_cb, weight_cb);
                            pack_reconfig_data_format(mul_weight_result_cb);
                            if constexpr (per_token_weight != 0) {
                                mul_init(mul_rms_result_cb, weight_cb);
                            } else {
                                mul_bcast_rows_init(mul_rms_result_cb, weight_cb);
                            }
                            cb_mul_weight_result.reserve_back(block_size);
                            tile_regs_acquire();
                            for (std::uint32_t i = 0; i < block_size; i++) {
                                if constexpr (per_token_weight != 0) {
                                    mul_tiles(mul_rms_result_cb, weight_cb, i, col_tile + i, i);
                                } else {
                                    mul_tiles_bcast_rows(mul_rms_result_cb, weight_cb, i, col_tile + i, i);
                                }
                            }
                            tile_regs_commit();
                            cb_mul_rms_result.pop_front(block_size);
                            tile_regs_wait();
                            for (std::uint32_t i = 0; i < block_size; i++) {
                                pack_tile(i, mul_weight_result_cb);
                            }
                            tile_regs_release();
                            cb_mul_weight_result.push_back(block_size);
                        }

                        // ---- + bias ----
                        if constexpr (has_bias) {
                            cb_bias.wait_front(col_tile + block_size);
                            cb_mul_weight_result.wait_front(block_size);
                            reconfig_data_format(mul_weight_result_cb, bias_cb);
                            pack_reconfig_data_format(add_bias_result_cb);
                            if constexpr (per_token_bias != 0) {
                                add_init(mul_weight_result_cb, bias_cb);
                            } else {
                                add_bcast_rows_init(mul_weight_result_cb, bias_cb);
                            }
                            cb_add_bias_result.reserve_back(block_size);
                            tile_regs_acquire();
                            for (std::uint32_t i = 0; i < block_size; i++) {
                                if constexpr (per_token_bias != 0) {
                                    add_tiles(mul_weight_result_cb, bias_cb, i, col_tile + i, i);
                                } else {
                                    add_tiles_bcast_rows(mul_weight_result_cb, bias_cb, i, col_tile + i, i);
                                }
                            }
                            tile_regs_commit();
                            cb_mul_weight_result.pop_front(block_size);
                            tile_regs_wait();
                            for (std::uint32_t i = 0; i < block_size; i++) {
                                pack_tile(i, add_bias_result_cb);
                            }
                            tile_regs_release();
                            cb_add_bias_result.push_back(block_size);
                        }

                        // ---- matmul-rotate + RoPE finalize (fuse_rope), else block already output ----
                        if constexpr (fuse_rope) {
                            // affine'd block is at intermediate_cb front (add_bias_result_cb ==
                            // intermediate_cb when fuse_rope). Rotate it, then RoPE-finalize.
                            reconfig_data_format(transformation_mat_cb, intermediate_cb);
                            pack_reconfig_data_format(rotated_input_cb);
                            matmul_block_init(
                                intermediate_cb,
                                transformation_mat_cb,
                                /*transpose=*/0,
                                /*ct_dim=*/1,
                                /*rt_dim=*/block_size,
                                /*kt_dim=*/1);
                            cb_intermediate.wait_front(block_size);  // NOT popped; RoPE re-reads it
                            cb_rotated_input.reserve_back(block_size);
                            tile_regs_acquire();
                            matmul_block(
                                intermediate_cb,
                                transformation_mat_cb,
                                /*in0_idx=*/0,
                                /*in1_idx=*/0,
                                /*idst=*/0,
                                /*transpose=*/0,
                                /*ct_dim=*/1,
                                /*rt_dim=*/block_size,
                                /*kt_dim=*/1);
                            tile_regs_commit();
                            tile_regs_wait();
                            for (std::uint32_t i = 0; i < block_size; i++) {
                                pack_tile(i, rotated_input_cb);
                            }
                            tile_regs_release();
                            cb_rotated_input.push_back(block_size);

                            // out = x*cos + rotate(x)*sin (FPU dst-accumulate, single rounding at pack).
                            // cos/sin are RESIDENT whole-row under block-major (the reader pushed
                            // the whole row before the deferred POST input pass). Per-head: this
                            // block's tiles sit at absolute col_tile..; broadcast: head_dim_tiles
                            // held resident, indexed cyclically. Drained once after the loop.
                            if constexpr (per_head_rope != 0) {
                                cb_rope_cos.wait_front(col_tile + block_size);
                                cb_rope_sin.wait_front(col_tile + block_size);
                            } else {
                                cb_rope_cos.wait_front(head_dim_tiles);
                                cb_rope_sin.wait_front(head_dim_tiles);
                            }
                            cb_intermediate.wait_front(block_size);
                            cb_rotated_input.wait_front(block_size);
                            cb_output.reserve_back(block_size);
                            reconfig_data_format(intermediate_cb, rope_cos_cb);
                            pack_reconfig_data_format(output_cb);
                            const std::uint32_t rope_base = rope_cursor;
                            tile_regs_acquire();
                            binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(intermediate_cb, rope_cos_cb, false);
                            for (std::uint32_t i = 0; i < block_size; i++) {
                                const std::uint32_t ridx =
                                    (per_head_rope != 0) ? (col_tile + i) : ((rope_base + i) % head_dim_tiles);
                                mul_tiles(intermediate_cb, rope_cos_cb, i, ridx, i);
                            }
                            binary_tiles_init<false, EltwiseBinaryType::ELWMUL>(rotated_input_cb, rope_sin_cb, true);
                            for (std::uint32_t i = 0; i < block_size; i++) {
                                const std::uint32_t ridx =
                                    (per_head_rope != 0) ? (col_tile + i) : ((rope_base + i) % head_dim_tiles);
                                mul_tiles(rotated_input_cb, rope_sin_cb, i, ridx, i);
                            }
                            if constexpr (per_head_rope == 0) {
                                rope_cursor = (rope_base + block_size) % head_dim_tiles;
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            for (std::uint32_t i = 0; i < block_size; i++) {
                                pack_tile(i, output_cb);
                            }
                            tile_regs_release();
                            cb_output.push_back(block_size);
                            cb_intermediate.pop_front(block_size);
                            cb_rotated_input.pop_front(block_size);
                            // cos/sin are resident whole-row; drained once after the loop (below).
                        }
                        // !fuse_rope: the last affine sub-phase already wrote output_cb (aliases).
                    }
                    cb_reduce_result.pop_front(1);
                    if constexpr (fuse_rope) {
                        // cos/sin held resident across the whole row; drain once. Per-head holds
                        // num_tile_cols tiles (one per col); broadcast holds head_dim_tiles.
                        const std::uint32_t rope_row_tiles = (per_head_rope != 0) ? num_tile_cols : head_dim_tiles;
                        cb_rope_cos.pop_front(rope_row_tiles);
                        cb_rope_sin.pop_front(rope_row_tiles);
                    }
                }

                if constexpr (has_weight && !block_major_post && !prescale_weight) {
                    // ----- Sub-phase 2: (x * 1/rms) * weight → mul_weight_result_cb -----
                    // Broadcast weight (default): weight_cb holds num_tile_cols
                    // row-broadcast tiles pushed once per worker; we use
                    // mul_tiles_bcast_rows so the same weight column applies to
                    // every token row.
                    // Per-token weight: weight_cb holds chunk_size_rows * num_tile_cols
                    // tiles, with this row's slice at the front. Use mul_tiles
                    // directly (no broadcast) and pop the row's slice at end of
                    // row.
                    reconfig_data_format(mul_rms_result_cb, weight_cb);
                    pack_reconfig_data_format(mul_weight_result_cb);
                    if constexpr (per_token_weight != 0) {
                        mul_init(mul_rms_result_cb, weight_cb);
                    } else {
                        mul_bcast_rows_init(mul_rms_result_cb, weight_cb);
                    }

                    for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                        const std::uint32_t tiles_in_block =
                            (col_tile + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col_tile);
                        cb_weight.wait_front(col_tile + tiles_in_block);
                        cb_mul_rms_result.wait_front(block_size);
                        tile_regs_acquire();
                        for (std::uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                            if constexpr (per_token_weight != 0) {
                                mul_tiles(mul_rms_result_cb, weight_cb, i, col_tile + i, i);
                            } else {
                                mul_tiles_bcast_rows(mul_rms_result_cb, weight_cb, i, col_tile + i, i);
                            }
                        }
                        tile_regs_commit();
                        cb_mul_rms_result.pop_front(block_size);
                        cb_mul_weight_result.reserve_back(block_size);
                        tile_regs_wait();
                        for (std::uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                            pack_tile(i, mul_weight_result_cb);
                        }
                        tile_regs_release();
                        cb_mul_weight_result.push_back(block_size);
                    }
                }

                if constexpr (has_bias && !block_major_post) {
                    // ----- Sub-phase 2.5: + bias → add_bias_result_cb -----
                    // Broadcast bias uses add_tiles_bcast_rows; per-token bias
                    // uses add_tiles. Same per-row pop pattern as weight when
                    // per_token_bias.
                    reconfig_data_format(mul_weight_result_cb, bias_cb);
                    pack_reconfig_data_format(add_bias_result_cb);
                    if constexpr (per_token_bias != 0) {
                        add_init(mul_weight_result_cb, bias_cb);
                    } else {
                        add_bcast_rows_init(mul_weight_result_cb, bias_cb);
                    }
                    for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                        const std::uint32_t tiles_in_block =
                            (col_tile + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col_tile);
                        cb_bias.wait_front(col_tile + tiles_in_block);
                        cb_mul_weight_result.wait_front(block_size);
                        tile_regs_acquire();
                        for (std::uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                            if constexpr (per_token_bias != 0) {
                                add_tiles(mul_weight_result_cb, bias_cb, i, col_tile + i, i);
                            } else {
                                add_tiles_bcast_rows(mul_weight_result_cb, bias_cb, i, col_tile + i, i);
                            }
                        }
                        tile_regs_commit();
                        cb_mul_weight_result.pop_front(block_size);
                        cb_add_bias_result.reserve_back(block_size);
                        tile_regs_wait();
                        for (std::uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                            pack_tile(i, add_bias_result_cb);
                        }
                        tile_regs_release();
                        cb_add_bias_result.push_back(block_size);
                    }
                }

                if constexpr (fuse_rope && !block_major_post) {
                    if constexpr (fuse_mm_rope) {
                        // ===== Block-major POST: fuse matmul-rotate + RoPE finalize per block =====
                        // rotated_input_cb is block-local (host shrank it for wide per-head
                        // shards that would otherwise overflow L1), so rotate ONE block then
                        // immediately RoPE-finalize it before the next. Costs per-block
                        // matmul<->rope reconfigs (the resident sub-phase-major path below is
                        // faster). fuse_mm_rope is only set when per_head_rope, so cos/sin are
                        // streamed: block-relative index, popped per block.
                        for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                            const std::uint32_t tiles_in_block =
                                (col_tile + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col_tile);
                            // --- rotate this block: intermediate[front] * trans -> rotated[block] ---
                            reconfig_data_format(transformation_mat_cb, intermediate_cb);
                            pack_reconfig_data_format(rotated_input_cb);
                            matmul_block_init(
                                intermediate_cb,
                                transformation_mat_cb,
                                /*transpose=*/0,
                                /*ct_dim=*/1,
                                /*rt_dim=*/block_size,
                                /*kt_dim=*/1);
                            cb_intermediate.wait_front(block_size);  // front block; popped after RoPE
                            cb_rotated_input.reserve_back(block_size);
                            tile_regs_acquire();
                            matmul_block(
                                intermediate_cb,
                                transformation_mat_cb,
                                /*in0_idx=*/0,
                                /*in1_idx=*/0,
                                /*idst=*/0,
                                /*transpose=*/0,
                                /*ct_dim=*/1,
                                /*rt_dim=*/block_size,
                                /*kt_dim=*/1);
                            tile_regs_commit();
                            tile_regs_wait();
                            for (std::uint32_t i = 0; i < block_size; i++) {
                                pack_tile(i, rotated_input_cb);
                            }
                            tile_regs_release();
                            cb_rotated_input.push_back(block_size);
                            // --- RoPE finalize: x*cos + rotate(x)*sin -> output (FPU dst-accumulate) ---
                            cb_rope_cos.wait_front(tiles_in_block);
                            cb_rope_sin.wait_front(tiles_in_block);
                            cb_rotated_input.wait_front(block_size);
                            cb_output.reserve_back(block_size);
                            reconfig_data_format(intermediate_cb, rope_cos_cb);
                            pack_reconfig_data_format(output_cb);
                            tile_regs_acquire();
                            binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(intermediate_cb, rope_cos_cb, false);
                            for (std::uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                mul_tiles(intermediate_cb, rope_cos_cb, i, i, i);  // block-relative cos
                            }
                            binary_tiles_init<false, EltwiseBinaryType::ELWMUL>(rotated_input_cb, rope_sin_cb, true);
                            for (std::uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                mul_tiles(rotated_input_cb, rope_sin_cb, i, i, i);  // + rotate*sin (acc)
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            for (std::uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                pack_tile(i, output_cb);
                            }
                            tile_regs_release();
                            cb_output.push_back(block_size);
                            cb_intermediate.pop_front(block_size);
                            cb_rotated_input.pop_front(block_size);
                            cb_rope_cos.pop_front(tiles_in_block);
                            cb_rope_sin.pop_front(tiles_in_block);
                        }
                    } else {
                        {
                            // ----- Sub-phase 3a: matmul(intermediate, trans_mat) → rotated -----
                            reconfig_data_format(transformation_mat_cb, intermediate_cb);
                            pack_reconfig_data_format(rotated_input_cb);
                            // Block matmul: rotate a whole block of tiles in ONE call —
                            // rt_dim=block_size tiles of intermediate, each multiplied by the
                            // single 32x32 transformation tile (kt_dim=ct_dim=1). One
                            // unpack+math dispatch per block instead of block_size per-tile
                            // matmul_tiles calls. intermediate_cb is padded to a block_size
                            // multiple (P_WEIGHT pushes full block_size/block), so rt_dim is
                            // always block_size; the RoPE finalize only consumes the valid
                            // tiles, so any padding rows packed here are never used.
                            matmul_block_init(
                                intermediate_cb,
                                transformation_mat_cb,
                                /*transpose=*/0,
                                /*ct_dim=*/1,
                                /*rt_dim=*/block_size,
                                /*kt_dim=*/1);
                            for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                                // Don't pop intermediate — the RoPE finalize re-reads it.
                                cb_intermediate.wait_front(col_tile + block_size);
                                cb_rotated_input.reserve_back(block_size);
                                tile_regs_acquire();
                                matmul_block(
                                    intermediate_cb,
                                    transformation_mat_cb,
                                    /*in0_idx=*/col_tile,
                                    /*in1_idx=*/0,
                                    /*idst=*/0,
                                    /*transpose=*/0,
                                    /*ct_dim=*/1,
                                    /*rt_dim=*/block_size,
                                    /*kt_dim=*/1);
                                tile_regs_commit();
                                tile_regs_wait();
                                for (std::uint32_t i = 0; i < block_size; i++) {
                                    pack_tile(i, rotated_input_cb);
                                }
                                tile_regs_release();
                                cb_rotated_input.push_back(block_size);
                            }
                        }  // P_MM

                        {
                            // ----- Fused RoPE finalize: out = x*cos + rotate(x)*sin -----
                            // FPU dst-accumulate. The first mul writes x*cos into dst;
                            // the second mul is initialized with acc_to_dest=true so the
                            // FPU computes rotate(x)*sin + dst -> dst (the add is free,
                            // done by the multiply, in fp32 dest). A SINGLE final rounding
                            // happens at pack -> precision-preserving (same as the old
                            // fp32-intermediate add). 1 dst reg / output tile (block_size
                            // tiles per acquire), 1 pack/tile. Replaces P_COS/P_SIN/P_ADD.
                            //
                            // Broadcast RoPE (per_head_rope==0) reuses head_dim_tiles cos/
                            // sin tiles cyclically; cos and sin share the same index, so we
                            // recompute the cursor from rope_base for both mul passes.
                            reconfig_data_format(intermediate_cb, rope_cos_cb);
                            pack_reconfig_data_format(output_cb);
                            for (std::uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                                const std::uint32_t tiles_in_block =
                                    (col_tile + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col_tile);
                                // Per-head RoPE: cos/sin are STREAMED — the reader pushes this
                                // block's tiles (block_size groups) and we pop them at block end,
                                // so only a few blocks are ever resident (O(block_size), not the
                                // full per-device width num_heads*head_dim — which overflows L1 at
                                // TP=2 feat-2048). Index is block-relative (front of CB).
                                // Broadcast RoPE: a small head_dim_tiles cos/sin buffer is held
                                // across the whole row and indexed cyclically; popped at end of row.
                                if constexpr (per_head_rope != 0) {
                                    cb_rope_cos.wait_front(tiles_in_block);
                                    cb_rope_sin.wait_front(tiles_in_block);
                                } else {
                                    cb_rope_cos.wait_front(head_dim_tiles);
                                    cb_rope_sin.wait_front(head_dim_tiles);
                                }
                                cb_intermediate.wait_front(block_size);
                                cb_rotated_input.wait_front(block_size);
                                cb_output.reserve_back(block_size);
                                const std::uint32_t rope_base = rope_cos_tile_in_head;
                                tile_regs_acquire();
                                // x*cos -> dst (overwrite: acc_to_dest=false)
                                binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(intermediate_cb, rope_cos_cb, false);
                                for (std::uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                    const std::uint32_t rope_idx =
                                        (per_head_rope != 0) ? i : ((rope_base + i) % head_dim_tiles);
                                    mul_tiles(intermediate_cb, rope_cos_cb, i, rope_idx, i);
                                }
                                // rotate(x)*sin + dst -> dst (FPU accumulate: acc_to_dest=true).
                                // full_init=false: operand formats match the x*cos mul above
                                // (intermediate/rotated both fp32, cos/sin both bf16), so the
                                // unpacker AB config is already valid — only the math init
                                // re-runs to flip acc_to_dest. Skips a redundant unpack init.
                                binary_tiles_init<false, EltwiseBinaryType::ELWMUL>(
                                    rotated_input_cb, rope_sin_cb, true);
                                std::uint32_t valid = 0;
                                for (std::uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                    const std::uint32_t rope_idx =
                                        (per_head_rope != 0) ? i : ((rope_base + i) % head_dim_tiles);
                                    mul_tiles(rotated_input_cb, rope_sin_cb, i, rope_idx, i);
                                    valid++;
                                }
                                if constexpr (per_head_rope == 0) {
                                    rope_cos_tile_in_head = (rope_base + valid) % head_dim_tiles;
                                }
                                tile_regs_commit();
                                tile_regs_wait();
                                for (std::uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                                    pack_tile(i, output_cb);
                                }
                                tile_regs_release();
                                cb_output.push_back(block_size);
                                cb_intermediate.pop_front(block_size);
                                cb_rotated_input.pop_front(block_size);
                                // Per-head: drain this block's streamed cos/sin now (matches the
                                // reader's per-block push). Broadcast keeps them for cyclic reuse.
                                if constexpr (per_head_rope != 0) {
                                    cb_rope_cos.pop_front(tiles_in_block);
                                    cb_rope_sin.pop_front(tiles_in_block);
                                }
                            }
                        }  // P_ROPE
                    }  // else: resident sub-phase-major P_MM + P_ROPE
                }

                if constexpr (fuse_rope && per_head_rope == 0 && !block_major_post) {
                    // Broadcast RoPE holds its head_dim_tiles cos/sin across the row;
                    // pop once here. Per-head streamed cos/sin were popped per block above
                    // (sum over blocks == num_tile_cols, the reader's per-row push count).
                    cb_rope_cos.pop_front(head_dim_tiles);
                    cb_rope_sin.pop_front(head_dim_tiles);
                }

                // Per-token weight/bias: pop the row's slice now so the next
                // row's slice is at the front. (Broadcast path keeps the same
                // tiles across all rows and pops at end of kernel.)
                if constexpr (per_token_weight != 0) {
                    cb_weight.pop_front(num_tile_cols);
                }
                if constexpr (per_token_bias != 0) {
                    cb_bias.pop_front(num_tile_cols);
                }
            }
        }  // RMS_POST

        // -------- RELEASE THIS CHUNK --------
        // Streaming popped input block-by-block in PRE (num_tile_cols) +
        // P_NMUL (num_tile_cols) = 2*num_tile_cols, matching the reader's two
        // passes. Only the resident path holds the chunk and drains it here.
        if constexpr (!streaming_low_l1) {
            cb_input.pop_front(chunk_input_tiles);
        }
        // NOTE: stats_gathered_cb is NOT popped here — the reduce<AVG> with
        // default WaitAndPopPerTile policy already drains chunk_stats_tiles
        // (num_heads × 1 for per_head, else ring_size × 1). A manual pop here would
        // be a DOUBLE-POP, advancing rd_ptr past wr_ptr and causing subsequent rows
        // to read stale L1 contents.
    }

    cb_reduce_scalar_sum.pop_front(1);
    cb_reduce_scalar_avg.pop_front(1);
    cb_epsilon.pop_front(1);
    // Broadcast weight/bias: pop the worker's single row slice at end of
    // kernel. Per-token already popped per-row inside the chunk loop.
    if constexpr (has_weight && per_token_weight == 0) {
        cb_weight.pop_front(num_tile_cols);
    }
    if constexpr (has_bias && per_token_bias == 0) {
        cb_bias.pop_front(num_tile_cols);
    }
    if constexpr (fuse_rope) {
        cb_transformation_mat.pop_front(1);
    }
}
