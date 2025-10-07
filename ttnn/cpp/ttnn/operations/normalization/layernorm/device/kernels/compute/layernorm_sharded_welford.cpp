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
#include "ttnn/operations/normalization/kernel_util/compute/combine_welford.hpp"

// SPLIT REDUCE across Cores
namespace NAMESPACE {
namespace {
// Get the set size of the next block in the Welford combine.
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

    // set block_ht to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_ht = (block_wt == 1) ? block_ht_volatile : block_ht_const;
    const uint32_t subblock_wt = (block_wt <= 2) ? subblock_wt_volatile : subblock_wt_const;

    // This value is the same for all cores, except ones that have padding tiles
    // in them. In that case, don't reduce over the padding elements.
    const uint32_t num_reduce_tiles_per_block_h = get_arg_val<uint32_t>(0);
    const uint32_t partial_reduce_W = (num_reduce_tiles_per_block_h - 1) * tile_width + last_tile_w;

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

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
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

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

#ifdef FUSE_PRE_ADD
    constexpr uint32_t cb_in = cb_x;
    binary_op_init_common(cb_in0, cb_in1, cb_in);
    // pack_reconfig_data_format(cb_in);
#else
    constexpr uint32_t cb_in = cb_in0;
    unary_op_init_common(cb_in, cb_ex_partial);
#endif
    constexpr uint32_t cb_im = (do_gamma | do_beta) ? cb_x : cb_out;
    constexpr uint32_t cb_outgamma = do_beta ? cb_fusion : cb_out;

// pre-add x + y
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
            for (uint32_t i = 0; i < subblock_wt; i++) {
                pack_tile(i, cb_in);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        index_h_offset += block_wt;
    }
    cb_push_back(cb_in, num_tiles_per_block);
    cb_wait_front(cb_in, num_tiles_per_block);
#endif  // FUSE_PRE_ADD

    // Compute E[x] and Var[x] using Welford's algorithm
    const uint32_t num_partial_tiles = num_block_ht_result_tiles;

    reconfig_data_format_srca(cb_in);

    cb_reserve_back(cb_ex_partial, num_partial_tiles);
    transpose_wh_init_short(cb_in);
    welford_init();
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_ht; i++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
            transpose_wh_tile(cb_in, w + index_h_offset, welford_input_dst);
            welford_tile<welford_input_dst, welford_mean_dst, welford_var_dst, true, 0>(
                w * tile_width, partial_reduce_W, 0, {});
        }
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
    cb_push_back(cb_ex_partial, num_partial_tiles);
    cb_wait_front(cb_ex_partial, num_partial_tiles);
    // tt::compute::common::print_full_tile(cb_ex_partial, 0, true);
    // tt::compute::common::print_full_tile(cb_ex_partial, 1, true);

    reconfig_data_format_srca(cb_ex_partial);

    // Combine Welford local partials with external partials
    // cb_ex <-- cb_ex_external, cb_ex_partial
    // where "ex" is mean and var interleaved.
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
        }
        cb_push_back(cb_ex, 2 * num_tiles_per_allgather_worker);
        cb_wait_front(cb_ex, 2 * num_tiles_per_allgather_worker);
    }

    cb_wait_front(cb_ex_global, num_block_ht_result_tiles);
    cb_reserve_back(cb_transpose, num_block_ht_result_tiles);
    transpose_wh_init(cb_ex_global, cb_transpose);
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

    // Compute (x - E[x])
    // Pack to cb_xmm
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_in, cb_transpose);
    }
    index_h_offset = 0;
    sub_bcast_cols_init_short(cb_in, cb_transpose);
    cb_reserve_back(cb_xmm, num_tiles_per_block);

    // See above notes about keeping mean/var as rows
    // and about needing a full tranpose init
    // transpose_wh_init(cb_ex_global, cb_ex_global);
    for (uint32_t i = 0; i < block_ht; i++) {
        index_subblock_w_offset = 0;
        const auto mean_idx = 2 * i;
        cb_wait_front(cb_transpose, mean_idx + 1);
        // DPRINT << "mean tile:" << ENDL();
        // tt::compute::common::print_full_tile(cb_transpose, mean_idx, true);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            // // Transpose back to columns
            // transpose_wh_tile(cb_ex_global, mean_idx, cb_ex_global);
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(cb_in, cb_transpose, index, mean_idx, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_wt; i++) {
                pack_tile(i, cb_xmm);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        cb_pop_front(cb_in, block_wt);
        // Don't pop global ex buffer until after the mul below
    }
    cb_push_back(cb_xmm, num_tiles_per_block);
#ifndef FUSE_PRE_ADD
    reconfig_data_format_srca(cb_in, cb_xmm);
#endif
    cb_wait_front(cb_xmm, num_tiles_per_block);
    // DPRINT << "xmm after packing" << ENDL();
    // uint32_t offset = 0;
    // for (uint32_t i = 0; i < block_ht; i++) {
    //     tt::compute::common::print_full_tile(cb_xmm, offset, true);
    //     offset += block_wt;
    // }

    if constexpr (do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(cb_out);
    }

    // (x - Ex) * 1/[sqrt(Var + eps)]
    // Pack to cb_im
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_transpose);
    }
    mul_bcast_cols_init_short(cb_xmm, cb_transpose);
    index_h_offset = 0;
    cb_reserve_back(cb_im, num_tiles_per_block);
    for (uint32_t i = 0; i < block_ht; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            // if (j == 0) {
            //     tt::compute::common::print_full_tile(cb_xmm, index_h_offset, true);
            //     tt::compute::common::print_full_tile(cb_transpose, 1, true);
            // }
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles_bcast_cols(cb_xmm, cb_transpose, index, /*1/sqrt(var+eps) idx*/ 1, w);
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_wt; i++) {
                pack_tile(i, cb_im);
            }
            tile_regs_release();

            index_subblock_w_offset += subblock_wt;
        }
        index_h_offset += block_wt;
        cb_pop_front(cb_transpose, 2);
    }
    cb_push_back(cb_im, num_tiles_per_block);

    cb_pop_front(cb_xmm, num_tiles_per_block);
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
                for (uint32_t i = 0; i < subblock_wt; i++) {
                    pack_tile(i, cb_outgamma);
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
                for (uint32_t i = 0; i < subblock_wt; i++) {
                    pack_tile(i, cb_out);
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
