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
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_scaler_global = tt::CBIndex::c_4;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;
    constexpr uint32_t cb_x = tt::CBIndex::c_24;  // x minus mean
#if defined RMSNORM and not defined FUSE_PRE_ADD
    constexpr uint32_t cb_xmm = cb_in0;  // x minus mean
#else
    constexpr uint32_t cb_xmm = tt::CBIndex::c_18;  // x minus mean
#endif
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
    const uint32_t block_h = (block_w == 1) ? block_h_volatile : block_h_const;
    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

#ifdef FUSE_PRE_ADD
#ifdef RMSNORM
    constexpr uint32_t cb_in = cb_xmm;
#else
    constexpr uint32_t cb_in = cb_x;
#endif
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
#ifndef RMSNORM
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_scaler);
#else
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_in);
#endif
    cb_wait_front(cb_in, num_tiles_per_block);
#else
#ifndef RMSNORM
    reconfig_data_format_srcb(cb_in0, cb_scaler);
#endif  // RMSNORM
#endif  // FUSE_PRE_ADD

#ifndef RMSNORM
    // E[x],
    index_h_offset = 0;
    reduce_init(cb_in, cb_scaler, cb_ex_partial);
    cb_wait_front(cb_scaler, 1);
    cb_reserve_back(cb_ex_partial, block_h);
    for (uint32_t i = 0; i < block_h; i++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
            reduce_tile(cb_in, cb_scaler, w + index_h_offset, scaler0, dst0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial);
        tile_regs_release();
        index_h_offset += block_w;
    }
    reduce_uninit();
    cb_push_back(cb_ex_partial, block_h);

    reconfig_data_format_srca(cb_in, cb_ex_external);

    // global reduce, cb_ex <-- cb_ex_external, cb_ex_partial
    if constexpr (is_allgather_worker) {
        reduce_init(cb_ex_external, cb_scaler_global, cb_ex);
        cb_reserve_back(cb_ex, num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            cb_wait_front(cb_scaler_global, 1);
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks_reduce; w++) {
                cb_wait_front(cb_ex_external, 1);
                reduce_tile(cb_ex_external, cb_scaler_global, 0, scaler0, dst0);
                cb_pop_front(cb_ex_external, 1);
            }
            if (use_two_stage_reduce && !is_second_stage_reader) {
                cb_wait_front(cb_ex_external, num_blocks_second_stage - 1);
                cb_pop_front(cb_ex_external, num_blocks_second_stage - 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex);
            tile_regs_release();
        }
        reduce_uninit();
        cb_push_back(cb_ex, num_tiles_per_allgather_worker);
        cb_wait_front(cb_ex, num_tiles_per_allgather_worker);
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
#endif

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

#if defined RMSNORM and not defined FUSED_PRE_ADD
    reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);
#else
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);
    }
#endif

    cb_wait_front(cb_xmm2, num_tiles_per_block);

// Var(x)
#ifdef RMSNORM
    cb_wait_front(cb_scaler, 1);
#endif
    cb_reserve_back(cb_ex_partial2, block_h);
    reduce_init(cb_xmm2, cb_scaler, cb_ex_partial2);
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_h; i++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
            reduce_tile(cb_xmm2, cb_scaler, w + index_h_offset, scaler0, dst0);
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
        reduce_init(cb_ex_external2, cb_scaler_global, cb_ex2);
        cb_reserve_back(cb_ex2, num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            cb_wait_front(cb_scaler_global, 1);

            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks_reduce; w++) {
                cb_wait_front(cb_ex_external2, 1);
                reduce_tile(cb_ex_external2, cb_scaler_global, 0, scaler0, dst0);
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
                // sqrt(Var + eps)
                sqrt_tile_init();
                sqrt_tile(dst0);
                tile_regs_wait();
                // 1/[sqrt(Var + eps)]
                recip_tile_init();
                recip_tile(dst0);
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
#if defined RMSNORM and not defined FUSE_PRE_ADD
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_ex_global);
    } else {
        reconfig_data_format_srca(cb_ex2, cb_xmm);
    }
#else
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_ex_global);
    }
#endif
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
