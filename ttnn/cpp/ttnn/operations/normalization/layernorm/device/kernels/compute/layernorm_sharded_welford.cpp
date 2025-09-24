// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/eltwise_unary/fill.h"

// SPLIT REDUCE across Cores
namespace NAMESPACE {
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
    constexpr uint32_t W = get_compile_time_arg_val(14);
    constexpr uint32_t total_width = get_compile_time_arg_val(15);
    constexpr uint32_t last_tile_data_width = get_compile_time_arg_val(16);

    const uint32_t num_reduce_tiles_per_block_h =
        get_arg_val<uint32_t>(0);  // This value is the same for all cores, except ones that have padding tiles in it.
                                   // In that case, skip reduce for padding tiles.
    const uint32_t num_tiles_per_allgather_worker = is_allgather_worker ? get_arg_val<uint32_t>(1) : 0;
    const bool use_two_stage_reduce = is_allgather_worker ? get_arg_val<uint32_t>(2) == 1 : false;
    const bool is_second_stage_reader = is_allgather_worker ? get_arg_val<uint32_t>(3) == 1 : false;

    uint32_t num_blocks_reduce;
    if (is_second_stage_reader) {
        num_blocks_reduce = num_blocks_first_stage + num_blocks_second_stage - 1;
    } else {
        num_blocks_reduce = num_blocks_first_stage;
    }

    bool enable_sqrt;
    if (use_two_stage_reduce and not is_second_stage_reader) {
        enable_sqrt = false;
    } else {
        enable_sqrt = true;
    }

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;
    constexpr uint32_t cb_x = tt::CBIndex::c_24;          // x minus mean
    constexpr uint32_t cb_xmm = tt::CBIndex::c_18;        // x minus mean
    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;  // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CBIndex::c_9;          // E[x] global reduce
    constexpr uint32_t cb_ex_external = tt::CBIndex::c_10;
    constexpr uint32_t cb_ex_partial2 = tt::CBIndex::c_11;  // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2 = tt::CBIndex::c_12;          // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = tt::CBIndex::c_13;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;  // E[x] global reduce
    constexpr uint32_t cb_xmm2 = cb_x;                    // xmm^2
    constexpr uint32_t cb_ex2pe = tt::CBIndex::c_20;      // E[(x-E[x])^2]+eps
    constexpr uint32_t cb_fusion = tt::CBIndex::c_18;     // stream gamma/beta
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    binary_op_init_common(cb_in0, cb_in0, cb_x);

    // set block_h to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_h = (block_wt == 1) ? block_ht_volatile : block_ht_const;
    const uint32_t subblock_w = (block_wt <= 2) ? subblock_wt_volatile : subblock_wt_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

#ifdef FUSE_PRE_ADD
    constexpr uint32_t cb_in = cb_x;
#else
    constexpr uint32_t cb_in = cb_in0;
#endif
    constexpr uint32_t cb_im = (do_gamma | do_beta) ? cb_x : cb_out;
    constexpr uint32_t cb_outgamma = do_beta ? cb_fusion : cb_out;

// pre-add x + y
#ifdef FUSE_PRE_ADD
    reconfig_data_format_srcb(cb_in0, cb_in1);
    add_tiles_init(cb_in0, cb_in1);
    cb_reserve_back(cb_in, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                add_tiles(cb_in0, cb_in1, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_in);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    cb_push_back(cb_in, num_tiles_per_block);
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_scaler);
    cb_wait_front(cb_in, num_tiles_per_block);
#else
    reconfig_data_format_srcb(cb_in0, cb_scaler);
#endif  // FUSE_PRE_ADD

    // Compute E[x] and Var[x] using Welford's algorithm
    constexpr uint32_t block_w = block_wt * tile_width;
    constexpr uint32_t num_partial_tiles = 2 * block_h;  // 1 mean tile and 1 var tile per block_h tile
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_x, num_tiles_per_block);
    cb_reserve_back(cb_ex_partial, num_partial_tiles);
    reconfig_data_format_srca(cb_x);
    welford_init();
    // Note: Using full init instead of short due to a bug
    // See PR tt-metal #650
    transpose_wh_init(cb_x, cb_x);
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_h; i++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
            transpose_wh_tile(cb_x, w + index_h_offset, dst0);
            welford_tile<in_dst, mean_dst, var_dst, true, 0>(w * tile_width, block_w, 0, {});
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(mean_dst, cb_ex_partial);
        pack_tile(var_dst, cb_ex_partial);
        tile_regs_release();
        index_h_offset += block_wt;
    }
    cb_push_back(cb_ex_partial, num_partial_tiles);

    reconfig_data_format_srca(cb_in, cb_ex_external);

    // Welford combine local partials with external partials
    // cb_ex <-- cb_ex_external, cb_ex_partial
    // where "ex" is mean and var interleaved.
    if constexpr (is_allgather_worker) {
        // Accumulate mean and M2 in dst regs
        constexpr uint32_t mean_acc_dst = 2;
        constexpr uint32_t m2_acc_dst = 3;

        // Work with two tiles (1 mean and 1 var) at a time
        constexpr uint32_t mean_cb_idx = 0;
        constexpr uint32_t var_cb_idx = 1;

        cb_reserve_back(cb_ex, num_tiles_per_allgather_worker);
        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            tile_regs_acquire();
            for (uint32_t b = 0; b < num_blocks_reduce; b++) {
                // Wait for 1 mean tile and 1 var tile
                cb_wait_front(cb_ex_external, 2);

                const auto n_a = b * tile_width;
                const auto n_b = b == num_blocks_reduce - 1 ? last_tile_data_width : tile_width;
                const auto n_ab = n_a + n_b;
                const auto n_b_over_n_ab = n_b / n_ab;
                const auto n_a_n_b_over_n_ab = n_a * n_b_over_n_ab;

                // Copy accumulated mean (x_a) to dst0
                copy_dest_values_init();
                copy_dest_values(dst0, mean_acc_dst);

                // Compute delta = x_b - x_a, store in dst0
                binary_dest_reuse_tiles_init<ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_ex_external);
                binary_dest_reuse_tiles<ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    cb_ex_external, mean_cb_idx, dst0);

                // Fill dst1 with n_b / n_ab
                fill_tile_init();
                fill_tile(dst1, n_b_over_n_ab);

                // Multiply delta by n_b / n_ab, store in dst1
                // (delta remains in dst0)
                mul_binary_tile_init();
                mul_binary_tile(dst0, dst1, dst1);

                // Accumulate mean
                add_binary_tile_init();
                add_binary_tile(mean_acc_dst, dst1, mean_acc_dst);

                // Square delta
                square_tile_init();
                square_tile(dst0);

                // Fill dst1 with n_a * n_b / n_ab
                fill_tile_init();
                fill_tile(dst1, n_a_n_b_over_n_ab);

                // Multiply delta^2 by n_a * n_b / n_ab, store in dst0
                mul_binary_tile_init();
                mul_binary_tile(dst0, dst1, dst0);

                // Accumulate into M2
                add_binary_tile_init();
                add_binary_tile(m2_acc_dst, dst0, m2_acc_dst);

                // Copy var_b into dst0
                copy_tile_to_dst_init_short(cb_ex_external);
                copy_tile(cb_ex_external, var_cb_idx, dst0);

                // Fill dst1 with n_b
                fill_tile_init();
                fill_tile(dst1, n_b);

                // Multiply var_b by n_b to get M2_b, store in dst0
                mul_binary_tile_init();
                mul_binary_tile(dst0, dst1, dst0);

                // Accumulate into M2
                add_binary_tile_init();
                add_binary_tile(m2_acc_dst, dst0, m2_acc_dst);

                cb_pop_front(cb_ex_external, 2);
            }

            // Divide final accumulated M2 by W to get final var
            fill_tile_init();
            fill_tile(dst0, static_cast<float>(W));
            div_binary_tile_init();
            div_binary_tile(m2_acc_dst, dst0, m2_acc_dst);

            tile_regs_commit();
            tile_regs_wait();

            // Pack accumulated mean and var to cb_ex
            pack_tile(mean_acc_dst, cb_ex);
            pack_tile(m2_acc_dst, cb_ex);
            tile_regs_release();
        }
        cb_push_back(cb_ex, 2 * num_tiles_per_allgather_worker);
        cb_wait_front(cb_ex, 2 * num_tiles_per_allgather_worker);
    }

    // x - E[x]
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_in, cb_ex_global);
    }
    index_h_offset = 0;
    reconfig_data_format_srca(cb_ex_external, cb_in);
    sub_bcast_cols_init_short(cb_in, cb_ex_global);
    cb_reserve_back(cb_xmm, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        cb_wait_front(cb_ex_global, 1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(cb_in, cb_ex_global, index, 0, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_xmm);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        cb_pop_front(cb_ex_global, 1);
        cb_pop_front(cb_in, block_w);
    }
    cb_push_back(cb_xmm, num_tiles_per_block);
#ifndef FUSE_PRE_ADD
    reconfig_data_format_srca(cb_in, cb_xmm);
#endif
    cb_wait_front(cb_xmm, num_tiles_per_block);

    // (x - E[x])^2, cb_mm2 <-- cb_xmm
    mul_tiles_init(cb_xmm, cb_xmm);
    index_h_offset = 0;
    cb_reserve_back(cb_xmm2, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles(cb_xmm, cb_xmm, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_xmm2);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    cb_push_back(cb_xmm2, num_tiles_per_block);

    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);
    }

    cb_wait_front(cb_xmm2, num_tiles_per_block);

    // Var(x)
    cb_reserve_back(cb_ex_partial2, block_h);
    reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_xmm2, cb_scaler, cb_ex_partial2);
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_h; i++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
            reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(
                cb_xmm2, cb_scaler, w + index_h_offset, scaler0, dst0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial2);
        tile_regs_release();
        index_h_offset += block_w;
    }
    reduce_uninit();
    cb_pop_front(cb_xmm2, num_tiles_per_block);
    cb_push_back(cb_ex_partial2, block_h);

    // global reduce, cb_ex <-- cb_ex_external, cb_ex_partial
    if constexpr (is_allgather_worker) {
        reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_ex_external2, cb_scaler_global, cb_ex2);
        cb_reserve_back(cb_ex2, num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            cb_wait_front(cb_scaler_global, 1);

            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks_reduce; w++) {
                cb_wait_front(cb_ex_external2, 1);
                reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(
                    cb_ex_external2, cb_scaler_global, 0, scaler0, dst0);
                cb_pop_front(cb_ex_external2, 1);
            }
            if (use_two_stage_reduce && !is_second_stage_reader) {
                cb_wait_front(cb_ex_external2, num_blocks_second_stage - 1);
                cb_pop_front(cb_ex_external2, num_blocks_second_stage - 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2);
            tile_regs_release();
        }
        reduce_uninit();
        cb_push_back(cb_ex2, num_tiles_per_allgather_worker);

        if (enable_sqrt) {
            for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
                // 1/[sqrt(Var + eps)],
                cb_wait_front(cb_ex2, 1);
                cb_reserve_back(cb_ex2pe, 1);
                tile_regs_acquire();
                add_tiles_init(cb_ex2, cb_eps);
                add_tiles(cb_ex2, cb_eps, i, 0, dst0);
                tile_regs_wait();
                rsqrt_tile_init<LEGACY_RSQRT>();
                rsqrt_tile<LEGACY_RSQRT>(dst0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_ex2pe);
                cb_push_back(cb_ex2pe, 1);
                tile_regs_release();
            }
        }
    }

    if constexpr (do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(cb_out);
    }

    // (x - Ex) * 1/[sqrt(Var + eps)]
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_ex_global);
    }
    mul_bcast_cols_init_short(cb_xmm, cb_ex_global);
    index_h_offset = 0;
    cb_reserve_back(cb_im, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        cb_wait_front(cb_ex_global, 1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles_bcast_cols(cb_xmm, cb_ex_global, index, 0, w);
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_im);
            }
            tile_regs_release();

            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
        cb_pop_front(cb_ex_global, 1);
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
        cb_wait_front(cb_gamma, block_w);
        index_h_offset = 0;
        cb_reserve_back(cb_outgamma, num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    mul_tiles_bcast_rows(cb_im, cb_gamma, index + index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_outgamma);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        cb_push_back(cb_outgamma, num_tiles_per_block);
        cb_pop_front(cb_im, num_tiles_per_block);
        cb_wait_front(cb_outgamma, num_tiles_per_block);
    }

    if constexpr (do_beta) {
        reconfig_data_format(cb_fusion, cb_beta);
        pack_reconfig_data_format(cb_out);
        add_bcast_rows_init_short(cb_fusion, cb_beta);
        cb_wait_front(cb_beta, block_w);
        index_h_offset = 0;
        cb_reserve_back(cb_out, num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles_bcast_rows(cb_fusion, cb_beta, index + index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            index_h_offset += block_w;
        }
        cb_push_back(cb_out, num_tiles_per_block);
        cb_pop_front(cb_fusion, num_tiles_per_block);
        cb_wait_front(cb_out, num_tiles_per_block);
    }
}

}  // namespace NAMESPACE
