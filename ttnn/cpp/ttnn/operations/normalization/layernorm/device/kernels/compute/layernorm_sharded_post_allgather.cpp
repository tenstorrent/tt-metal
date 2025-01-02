// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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

// SPLIT REDUCE across Cores
namespace NAMESPACE {
void MAIN {
    constexpr uint32_t is_top_row = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(3);
    constexpr uint32_t block_w = get_compile_time_arg_val(5);
    constexpr uint32_t block_h_const = get_compile_time_arg_val(4);
    volatile uint32_t block_h_volatile = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w_const = get_compile_time_arg_val(6);
    volatile uint32_t subblock_w_volatile = get_compile_time_arg_val(6);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(7);
    const bool is_allgather_worker = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(9);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(11);

    const uint32_t num_reduce_tiles_per_block_h =
        get_arg_val<uint32_t>(0);  // This value is the same for all cores, except ones that have padding tiles in it.
                                   // In that case, skip reduce for padding tiles.
    const uint32_t num_tiles_per_allgather_worker = is_allgather_worker ? get_arg_val<uint32_t>(1) : 0;
    const bool use_two_stage_reduce = is_allgather_worker ? get_arg_val<uint32_t>(2) == 1 : false;
    const bool is_second_stage_reader = is_allgather_worker ? get_arg_val<uint32_t>(3) == 1 : false;
    const uint32_t num_distributed_blocks = is_allgather_worker ? get_arg_val<uint32_t>(4) : 0;

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
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_scaler_global = tt::CBIndex::c_4;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;

    constexpr uint32_t cb_ex = tt::CBIndex::c_9;              // E[x] global reduce
    constexpr uint32_t cb_ex2 = tt::CBIndex::c_12;            // E[x^2]
    constexpr uint32_t cb_stats = tt::CBIndex::c_7;           // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_stats_reduced = tt::CBIndex::c_28;  // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;      // E[x] global reduce
    constexpr uint32_t cb_reciprocal = tt::CBIndex::c_27;     // [E[x^2]-E[x]^2]+eps
    constexpr uint32_t cb_fusion = tt::CBIndex::c_25;         // stream gamma/beta
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_var = tt::CBIndex::c_26;
    constexpr uint32_t cb_ex_sqr = tt::CBIndex::c_24;  // E[x]^2

#ifdef RMSNORM
    binary_op_init_common(cb_stats, cb_scaler_global, cb_var);
    constexpr uint32_t stats_tiles = 1;
    constexpr uint32_t cb_xmm = cb_in0;  // x
#else
    binary_op_init_common(cb_stats, cb_scaler_global, cb_stats_reduced);
    constexpr uint32_t stats_tiles = 2;
    constexpr uint32_t cb_xmm = tt::CBIndex::c_25;  // x minus mean
#endif

    // set block_h to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_h = (block_w == 1) ? block_h_volatile : block_h_const;
    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

    constexpr uint32_t cb_im = (do_gamma | do_beta) ? cb_ex_sqr : cb_out;
    constexpr uint32_t cb_outgamma = do_beta ? cb_fusion : cb_out;

    // global reduce, cb_ex <-- cb_ex_external, cb_ex_partial
    if constexpr (is_allgather_worker) {
        if (enable_sqrt) {
#ifdef RMSNORM
            cb_reserve_back(cb_var, 1);
#else
            cb_reserve_back(cb_stats_reduced, 1);
            cb_reserve_back(cb_ex2, 1);
#endif

            cb_wait_front(cb_scaler_global, 1);
            reduce_init_delta<false>(cb_var, cb_stats, cb_scaler_global);
            tile_regs_acquire();
            // striding over cb_stats, consisting [E(X), E(X^2)] from all the distributed devices in interleaved order
            for (uint32_t w = 0; w < stats_tiles * num_distributed_blocks; w++) {
                reduce_tile(
                    cb_stats,
                    cb_scaler_global,
                    0,
                    scaler0,
                    w % stats_tiles);  // reducing E(x) and E(x^2) separately to different dst
                cb_pop_front(cb_stats, 1);
            }
            tile_regs_commit();
            tile_regs_wait();

#ifdef RMSNORM
            pack_tile(dst0, cb_var);
#else
            pack_tile(dst0, cb_stats_reduced);
            pack_tile(dst1, cb_ex2);
#endif
            tile_regs_release();
            reduce_revert_delta(cb_var);
#ifdef RMSNORM
            cb_push_back(cb_var, stats_tiles);
#else
            cb_push_back(cb_stats_reduced, 1);
            cb_push_back(cb_ex2, 1);
#endif

#ifndef RMSNORM
            // calculate var = E(x^2) - E(x)^2
            // E(x)^2
            reconfig_data_format(cb_stats_reduced, cb_stats_reduced);
            cb_reserve_back(cb_ex_sqr, 1);
            cb_wait_front(cb_stats_reduced, 1);
            tile_regs_acquire();
            mul_tiles_init();
            mul_tiles(cb_stats_reduced, cb_stats_reduced, 0, 0, dst0);  // first tile in stats is always E(x)
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex_sqr);
            cb_push_back(cb_ex_sqr, 1);
            tile_regs_release();

            // E(x^2) - E(x)^2
            reconfig_data_format_srca(cb_stats_reduced, cb_ex2);
            reconfig_data_format_srcb(cb_stats_reduced, cb_ex_sqr);
            pack_reconfig_data_format(cb_var);
            cb_wait_front(cb_ex2, 1);
            cb_wait_front(cb_ex_sqr, 1);
            cb_reserve_back(cb_var, 1);
            tile_regs_acquire();
            sub_tiles_init();
            sub_tiles(cb_ex2, cb_ex_sqr, 0, 0, dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_var);
            cb_push_back(cb_var, 1);
            tile_regs_release();
            cb_pop_front(cb_ex2, 1);
            cb_pop_front(cb_ex_sqr, 1);
#endif

            // 1/[sqrt(Var + eps)],
            reconfig_data_format(cb_var, cb_eps);  // cb_var is cb_stats in case of RMS norm
            pack_reconfig_data_format(cb_stats_reduced);
            cb_wait_front(cb_var, 1);
            cb_wait_front(cb_eps, 1);
            cb_reserve_back(cb_stats_reduced, 1);

            add_tiles_init();
            tile_regs_acquire();
            add_tiles(cb_var, cb_eps, 0, 0, dst0);
            tile_regs_wait();
            sqrt_tile_init();
            sqrt_tile(dst0);
            recip_tile_init();
            recip_tile(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_stats_reduced);
            tile_regs_release();
            cb_pop_front(cb_var, 1);
            cb_pop_front(cb_eps, 1);
            cb_push_back(cb_stats_reduced, 1);
        }
    }

#ifndef RMSNORM
    // x - E[x]
    reconfig_data_format(cb_in0, cb_ex_global);
    pack_reconfig_data_format(cb_xmm);
    index_h_offset = 0;
    sub_bcast_cols_init_short();
    cb_reserve_back(cb_xmm, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        cb_wait_front(cb_ex_global, 1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(cb_in0, cb_ex_global, index, 0, w);
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
        cb_pop_front(cb_in0, block_w);
    }
    cb_push_back(cb_xmm, num_tiles_per_block);
#endif

    if constexpr (do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(cb_out);
    } else {
        pack_reconfig_data_format(cb_im);
    }

    // (x - Ex) * 1/[sqrt(Var + eps)]
    reconfig_data_format(cb_xmm, cb_ex_global);
    mul_bcast_cols_init_short();
    index_h_offset = 0;
    cb_reserve_back(cb_im, num_tiles_per_block);
#ifndef RMSNORM
    cb_wait_front(cb_xmm, num_tiles_per_block);
#endif
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
        mul_bcast_rows_init_short();
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
        add_bcast_rows_init_short();
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
