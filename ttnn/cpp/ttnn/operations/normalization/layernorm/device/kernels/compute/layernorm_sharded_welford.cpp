// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/welford.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "ttnn/operations/normalization/kernel_util/compute/combine_welford.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

/**
 * @brief This kernel computes layernorm for sharded tensors using
 *        Welford's algorithm for the mean and variance calculations
 *
 * @details Computes layernorm(x) = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 *
 * There are two flavors of sharded layernorm. The details
 * here are for row-major tensors, but the logic for column-major
 * is the same (core/tensor rows are replaced by core/tensor columns):
 * 1. Single-stage reduce:
 *   - Each core gets a width slice (of size `block_wt` tiles) of
 *     one or more rows of the tensor (the number of rows each core
 *     is assigned is `num_tiles_per_allgather_worker`)
 *   - Each core computes its partial mean and variance of its slices
 *     for all rows it is assigned (using Welford's algorithm) and pushes
 *     the interleaved mean and variance results to `cb_ex_partial`.
 *     This produces 1 mean tile and 1 variance tile per tile row
 *   - The reader kernels populate `cb_ex_external` with the core's
 *     partial result + the other partial results from cores in the
 *     same core row for each row the core is assigned
 *   - Each core combines all partial results for its row(s) in
 *     `cb_ex_external` into `cb_ex`. `cb_ex` contains 1 mean tile
 *     followed by 1 tile of 1/sqrt(var + eps) for each assigned row
 *   - The core row's sender core (the first column of cores) collects
 *     all of these combined tiles and multicasts to all cores in the row
 *     into `cb_ex_global`
 * 2. Two-stage reduce:
 *   - Used for width-sharded tensors, where each core has a
 *     tensor-height-tall slice of the tensor
 *   - The top row of cores are designated as "second-stage readers"
 *   - As in single-stage reduce, each core in a row computes
 *     its partial mean and variance for each of its assigned rows
 *     for its width shard then combines it with the other partials
 *     in its core row. This is the first stage of the reduce.
 *   - Second stage of the reduce: The reader kernels add the
 *     first stage's combined results into the second stage readers'
 *     `cb_ex_external`. The results in `cb_ex_external` are combined
 *     in the same way.
 *   - The final combined results are collected by the sender core
 *     and multicasted to all cores into `cb_ex_global`
 *
 * After one of the two reduce paths above, the rest of the layernorm
 * calculation is done using the global mean and 1/sqrt(var + eps) results.
 *
 * @note Depending on the tensor and core grid shape, some cores may
 *       not participate in the combine (i.e., `is_allgather_worker`
 *       will be false). These cores do their partial reduction and
 *       receive the multicasted global results and perform
 *       the rest of layernorm for their row(s) width slices.
 */
namespace NAMESPACE {
namespace {
// Get the set size of the next block in the Welford combine
inline auto get_next_set_size(
    const uint32_t block,
    const uint32_t num_blocks_combine,
    const bool is_second_stage_reader,
    const uint32_t num_blocks_first_stage,
    const uint32_t second_stage_w,
    const uint32_t block_w,
    const uint32_t last_block_w) {
    if (is_second_stage_reader) {
        // The next block is either one of the second-stage
        // blocks from our core column  or one of the
        // first-stage blocks from our core row (if row major).
        // This logic relies on the fact that the second stage
        // readers in a two-stage reduce get the reduce results
        // from its core column (for row major) streamed in last
        // (after the results for its core row)
        return block >= num_blocks_first_stage ? second_stage_w : block_w;
    }

    // We're either not doing a two-stage reduce or we're
    // not a second stage reader, so the next block will either
    // be a full block or a partial one if it's the last
    return block == num_blocks_combine - 1 ? last_block_w : block_w;
}
}  // namespace
void MAIN {
    // ============================================================================
    // Kernel setup
    // ============================================================================

    // ---------------------------------------------------------------------------
    // Compile-time arguments
    // ---------------------------------------------------------------------------
    constexpr uint32_t is_top_row = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(3);
    constexpr uint32_t block_wt = get_compile_time_arg_val(5);
    constexpr uint32_t block_ht_const = get_compile_time_arg_val(4);
    volatile uint32_t block_ht_volatile = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_wt_const = get_compile_time_arg_val(6);
    volatile uint32_t subblock_wt_volatile = get_compile_time_arg_val(6);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(7);
    const bool is_allgather_worker = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(9);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(10) == 1;
    constexpr bool FLOAT32_REDUCTION = get_compile_time_arg_val(11) == 1;
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(13);
    constexpr uint32_t tile_width = get_compile_time_arg_val(14);
    constexpr uint32_t last_tile_w = get_compile_time_arg_val(15);
    constexpr uint32_t W = get_compile_time_arg_val(16);
    constexpr uint32_t eps = get_compile_time_arg_val(17);
    constexpr uint32_t per_core_recip_lut_size = get_compile_time_arg_val(18);

    // ---------------------------------------------------------------------------
    // CB definitions
    // ---------------------------------------------------------------------------
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;
    constexpr uint32_t cb_x = tt::CBIndex::c_24;          // x minus mean
    constexpr uint32_t cb_xmm = tt::CBIndex::c_18;        // x minus mean
    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;  // Interleaved E[x] and Var[x] partial results
    constexpr uint32_t cb_ex = tt::CBIndex::c_9;          // Interleaved E[x] and Var[x] global reduce
    constexpr uint32_t cb_ex_external = tt::CBIndex::c_10;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;  // Interleaved E[x] and Var[x] final global mcast result
    constexpr uint32_t cb_transpose = tt::CBIndex::c_22;  // Transpose interleaved E[x] and Var[x] to columns
                                                          // (workaround for bug in transpose_wh_dest)
    constexpr uint32_t cb_fusion = tt::CBIndex::c_18;     // stream gamma/beta
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_reciprocals = tt::CBIndex::c_25;  // LUT of pre-computed reciprocals for Welford's algorithm
    constexpr uint32_t cb_im = (do_gamma | do_beta) ? cb_x : cb_out;
    constexpr uint32_t cb_outgamma = do_beta ? cb_fusion : cb_out;
#ifdef FUSE_PRE_ADD
    constexpr uint32_t cb_in = cb_x;
#else
    constexpr uint32_t cb_in = cb_in0;
#endif

    // ---------------------------------------------------------------------------
    // Derived quantities
    // ---------------------------------------------------------------------------
    // set block_ht to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_ht = (block_wt == 1) ? block_ht_volatile : block_ht_const;
    const uint32_t subblock_wt = (block_wt <= 2) ? subblock_wt_volatile : subblock_wt_const;

    // This value is the same for all cores, except ones that have padding tiles
    // in them. In that case, don't reduce over the padding elements.
    const uint32_t num_reduce_tiles_per_block_h = get_arg_val<uint32_t>(0);
    const uint32_t partial_reduce_W = (num_reduce_tiles_per_block_h - 1) * tile_width + last_tile_w;

    // We split the Welford calls into full tiles and a final partial tile (if any)
    const bool all_full_tiles = partial_reduce_W % tile_width == 0;
    const uint32_t num_full_welford_tiles =
        all_full_tiles ? num_reduce_tiles_per_block_h : num_reduce_tiles_per_block_h - 1;
    const uint32_t partial_welford_tile_w = all_full_tiles ? 0 : last_tile_w;

    // This is the number of tile rows to process
    const uint32_t num_tiles_per_allgather_worker = is_allgather_worker ? get_arg_val<uint32_t>(1) : 0;

    // These are for two-stage reductions
    const bool use_two_stage_reduce = is_allgather_worker ? get_arg_val<uint32_t>(2) == 1 : false;
    const bool is_second_stage_reader = is_allgather_worker ? get_arg_val<uint32_t>(3) == 1 : false;
    constexpr uint32_t block_w = block_wt * tile_width;
    constexpr uint32_t last_block_w = block_w - tile_width + last_tile_w;
    uint32_t first_stage_w =
        use_two_stage_reduce ? num_blocks_first_stage * block_w : (num_blocks_first_stage - 1) * block_w + last_block_w;
    uint32_t second_stage_w = use_two_stage_reduce ? (num_blocks_first_stage - 1) * block_w + last_block_w : 0;

    // The number of blocks to combine.
    // If we're the second stage reader, we're reducing the
    // entire tensor width.
    // If we're part of a two-stage reduce and not a reader,
    // or we're part of a single-stage reduce, we're reducing
    // width is only along our row
    uint32_t num_blocks_combine =
        is_second_stage_reader ? num_blocks_first_stage + num_blocks_second_stage - 1 : num_blocks_first_stage;

    // Number of tiles for block_ht results (interleaved mean and var)
    const uint32_t num_block_ht_result_tiles = 2 * block_ht;

    // Only used for the transpose workaround
    constexpr uint32_t num_dest_regs = FLOAT32_DTYPE ? 4 : 8;

    // Welford destination registers
    constexpr uint32_t welford_input_dst = 0;
    constexpr uint32_t welford_mean_dst = 1;
    constexpr uint32_t welford_var_dst = 2;

    // Pointer to the reciprocal LUT

    using recip_lut_t = std::array<uint32_t, per_core_recip_lut_size>;
    auto p_reciprocals = norm::kernel_util::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0);

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

    // ============================================================================
    // Main kernel logic
    // ============================================================================

    // ---------------------------------------------------------------------------
    // Op initialization
    // ---------------------------------------------------------------------------
#ifdef FUSE_PRE_ADD
    binary_op_init_common(cb_in0, cb_in1, cb_in);
#else
    unary_op_init_common(cb_in, cb_ex_partial);
#endif

    // ---------------------------------------------------------------------------
    // Pre-add x + y
    // ---------------------------------------------------------------------------
#ifdef FUSE_PRE_ADD
    reconfig_data_format_srcb(cb_in0, cb_in1);
    add_tiles_init(cb_in0, cb_in1);
    cb_reserve_back(cb_in, num_tiles_per_block);
    for (uint32_t i = 0; i < block_ht; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                add_tiles(cb_in0, cb_in1, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                pack_tile(sbi, cb_in);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        index_h_offset += block_wt;
    }
    cb_push_back(cb_in, num_tiles_per_block);
    cb_wait_front(cb_in, num_tiles_per_block);
#endif

    // ---------------------------------------------------------------------------
    // Compute E[x] and Var[x] using Welford's algorithm
    // ---------------------------------------------------------------------------
    reconfig_data_format_srca(cb_in);
    cb_reserve_back(cb_ex_partial, num_block_ht_result_tiles);
    transpose_wh_init_short(cb_in);
    welford_init();
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_ht; i++) {
        tile_regs_acquire();
        welford_clear();
        uint32_t sample_idx = 0;

        // Do the full Welford tiles
        for (uint32_t w = 0; w < num_full_welford_tiles; w++) {
            transpose_wh_tile(cb_in, w + index_h_offset, welford_input_dst);
            welford_update<per_core_recip_lut_size>(welford_input_dst, sample_idx, *p_reciprocals);
            sample_idx += tile_width;
        }
        // Do the partial Welford tile, if any
        if (partial_welford_tile_w > 0) {
            transpose_wh_tile(cb_in, block_wt - 1, welford_input_dst);
            welford_update_rows<per_core_recip_lut_size>(
                welford_input_dst, sample_idx, 0, partial_welford_tile_w, *p_reciprocals);
        }
        welford_finalize_to_row<per_core_recip_lut_size>(welford_mean_dst, partial_reduce_W - 1, *p_reciprocals);
        // We should transpose back to columns here
        // However, transpose_wh_dest() is currently buggy.
        // So we transpose to an intermediate CB downstream
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(welford_mean_dst, cb_ex_partial);
        pack_tile(welford_var_dst, cb_ex_partial);
        tile_regs_release();
        index_h_offset += block_wt;
    }
    cb_push_back(cb_ex_partial, num_block_ht_result_tiles);
    cb_wait_front(cb_ex_partial, num_block_ht_result_tiles);

    // ---------------------------------------------------------------------------
    // Combine Welford local partials with external partials
    // cb_ex <-- cb_ex_external, cb_ex_partial
    // If reduction is single-stage, or this core is a second-stage reader,
    // then cb_ex contains mean and 1/sqrt(var + eps) interleaved.
    // Otherwise, cb_ex contains mean and var interleaved.
    // ---------------------------------------------------------------------------
    reconfig_data_format_srca(cb_ex_partial);
    if constexpr (is_allgather_worker) {
        cb_reserve_back(cb_ex, 2 * num_tiles_per_allgather_worker);
        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            norm::kernel_util::compute::combine_welford_partials(
                cb_ex_external,
                cb_ex,
                num_blocks_combine,
                [&](uint32_t b) {
                    return get_next_set_size(
                        b,
                        num_blocks_combine,
                        is_second_stage_reader,
                        num_blocks_first_stage,
                        second_stage_w,
                        block_w,
                        last_block_w);
                },
                norm::kernel_util::compute::RSqrtPolicy{!(use_two_stage_reduce && !is_second_stage_reader), eps});

            // Just needed to stay in sync with the readers
            if (use_two_stage_reduce && !is_second_stage_reader) {
                // Number of second-stage tiles = 2 * (num_blocks_second_stage - 1)
                // The -1 is the account for the row-column overlap core
                // between first stage (row) and second stage (column) (if row major).
                // The factor of 2 is because each block has 2 tiles (mean, var).
                constexpr uint32_t num_second_stage_tiles = 2 * (num_blocks_second_stage - 1);
                cb_wait_front(cb_ex_external, num_second_stage_tiles);
                cb_pop_front(cb_ex_external, num_second_stage_tiles);
            }
        }
        cb_push_back(cb_ex, 2 * num_tiles_per_allgather_worker);
        cb_wait_front(cb_ex, 2 * num_tiles_per_allgather_worker);
    }

    // ---------------------------------------------------------------------------
    // Receive the global reduce result and transpose back to columns
    // ---------------------------------------------------------------------------
    cb_wait_front(cb_ex_global, num_block_ht_result_tiles);
    cb_reserve_back(cb_transpose, num_block_ht_result_tiles);
    transpose_wh_init_short(cb_ex_global);
    uint32_t processed_tiles = 0;
    while (processed_tiles < num_block_ht_result_tiles) {
        uint32_t tiles_to_load = std::min(num_block_ht_result_tiles - processed_tiles, num_dest_regs);
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_to_load; i++) {
            transpose_wh_tile(cb_ex_global, processed_tiles + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < tiles_to_load; i++) {
            pack_tile(i, cb_transpose);
        }
        tile_regs_release();
        processed_tiles += tiles_to_load;
    }
    cb_push_back(cb_transpose, num_block_ht_result_tiles);
    cb_pop_front(cb_ex_global, num_block_ht_result_tiles);

    cb_wait_front(cb_transpose, num_block_ht_result_tiles);

    // ---------------------------------------------------------------------------
    // Compute x - E[x]
    // ---------------------------------------------------------------------------
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_in, cb_transpose);
    }
    index_h_offset = 0;
    sub_bcast_cols_init_short(cb_in, cb_transpose);
    cb_reserve_back(cb_xmm, num_tiles_per_block);
    for (uint32_t i = 0; i < block_ht; i++) {
        index_subblock_w_offset = 0;
        const auto mean_idx = 2 * i;
        cb_wait_front(cb_transpose, mean_idx + 1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(cb_in, cb_transpose, index, mean_idx, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                pack_tile(sbi, cb_xmm);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        cb_pop_front(cb_in, block_wt);
        // Don't pop transpose buffer until after the mul below
    }
    cb_push_back(cb_xmm, num_tiles_per_block);
#ifndef FUSE_PRE_ADD
    reconfig_data_format_srca(cb_in, cb_xmm);
#endif
    cb_wait_front(cb_xmm, num_tiles_per_block);

    if constexpr (do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(cb_out);
    }

    // ---------------------------------------------------------------------------
    // Scale by 1/sqrt(Var[x] + eps)
    // ---------------------------------------------------------------------------
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_transpose);
    }
    mul_bcast_cols_init_short(cb_xmm, cb_transpose);
    index_h_offset = 0;
    cb_reserve_back(cb_im, num_tiles_per_block);
    for (uint32_t i = 0; i < block_ht; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles_bcast_cols(cb_xmm, cb_transpose, index, /*1/sqrt(var+eps) idx*/ 1, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                pack_tile(sbi, cb_im);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        index_h_offset += block_wt;
        cb_pop_front(cb_transpose, 2);
    }
    cb_push_back(cb_im, num_tiles_per_block);
    cb_pop_front(cb_xmm, num_tiles_per_block);

    // ---------------------------------------------------------------------------
    // Scale by gamma
    // ---------------------------------------------------------------------------
    cb_wait_front(cb_im, num_tiles_per_block);
    if constexpr (do_gamma) {
        reconfig_data_format(cb_im, cb_gamma);
        if constexpr (do_beta == 0) {
            pack_reconfig_data_format(cb_out);
        }
        mul_bcast_rows_init_short(cb_im, cb_gamma);
        cb_wait_front(cb_gamma, block_wt);
        index_h_offset = 0;
        cb_reserve_back(cb_outgamma, num_tiles_per_block);
        for (uint32_t i = 0; i < block_ht; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_wt; w++) {
                    index = w + index_subblock_w_offset;
                    mul_tiles_bcast_rows(cb_im, cb_gamma, index + index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                    pack_tile(sbi, cb_outgamma);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_wt;
            }
            index_h_offset += block_wt;
        }
        cb_push_back(cb_outgamma, num_tiles_per_block);
        cb_pop_front(cb_im, num_tiles_per_block);
        cb_wait_front(cb_outgamma, num_tiles_per_block);
    }

    // ---------------------------------------------------------------------------
    // Add beta
    // ---------------------------------------------------------------------------
    if constexpr (do_beta) {
        reconfig_data_format(cb_fusion, cb_beta);
        pack_reconfig_data_format(cb_out);
        add_bcast_rows_init_short(cb_fusion, cb_beta);
        cb_wait_front(cb_beta, block_wt);
        index_h_offset = 0;
        cb_reserve_back(cb_out, num_tiles_per_block);
        for (uint32_t i = 0; i < block_ht; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_wt; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles_bcast_rows(cb_fusion, cb_beta, index + index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t sbi = 0; sbi < subblock_wt; sbi++) {
                    pack_tile(sbi, cb_out);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_wt;
            }
            index_h_offset += block_wt;
        }
        cb_push_back(cb_out, num_tiles_per_block);
        cb_pop_front(cb_fusion, num_tiles_per_block);
        cb_wait_front(cb_out, num_tiles_per_block);
    }
}

}  // namespace NAMESPACE
