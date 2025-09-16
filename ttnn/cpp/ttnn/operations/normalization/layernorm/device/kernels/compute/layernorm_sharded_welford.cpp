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
#include "compute_kernel_api/welford.h"

// SPLIT REDUCE across Cores
namespace NAMESPACE {
void MAIN {
    constexpr uint32_t is_top_row = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(3);
    constexpr uint32_t block_wt = get_compile_time_arg_val(5);
    constexpr uint32_t block_h_const = get_compile_time_arg_val(4);
    volatile uint32_t block_h_volatile = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_wt_const = get_compile_time_arg_val(6);
    volatile uint32_t subblock_wt_volatile = get_compile_time_arg_val(6);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(7);
    const bool is_allgather_worker = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(9);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(11);
    constexpr bool fuse_pre_add = static_cast<bool>(get_compile_time_arg_val(13));
    constexpr uint32_t tile_width = get_compile_time_arg_val(14);

    const uint32_t num_reduce_tiles_per_block_h =
        get_arg_val<uint32_t>(0);  // This value is the same for all cores, except ones that have padding tiles in it.
                                   // In that case, skip reduce for padding tiles.

    // Number of tile rows per allgather worker
    // TODO RM: Rename this for clarity
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

    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t in_dst = 0;    // Input dest reg for Welford's
    constexpr uint32_t mean_dst = 1;  // Mean dest reg for Welford's
    constexpr uint32_t var_dst = 2;   // Variance dest reg for Welford's

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_scaler_global = tt::CBIndex::c_4;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta = tt::CBIndex::c_6;
    constexpr uint32_t cb_x = fuse_pre_add ? tt::CBIndex::c_24 : tt::CBIndex::cb_in0;  // input or fused pre-add result
    constexpr uint32_t cb_xmm = tt::CBIndex::c_18;                                     // x minus mean
    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;                               // E[x] partial result
    constexpr uint32_t cb_ex_combine = tt::CBIndex::c_9;                               // E[x] buffer for global combine
    constexpr uint32_t cb_varx_partial = tt::CBIndex::c_11;                            // Var[x] partial result
    constexpr uint32_t cb_varx_combine = tt::CBIndex::c_12;  // Var[x] buffer for global combine
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;     // Final global E[x]
    constexpr uint32_t cb_varx_global = tt::CBIndex::c_10;   // Final global Var[x]
    constexpr uint32_t cb_ex2pe = tt::CBIndex::c_20;         // E[(x-E[x])^2]+eps
    constexpr uint32_t cb_fusion = tt::CBIndex::c_18;        // stream gamma/beta
    constexpr uint32_t cb_out = tt::CBIndex::c_16;           // output

    binary_op_init_common(cb_in0, cb_in0, cb_x);

    // set block_h to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_h = (block_wt == 1) ? block_h_volatile : block_h_const;
    const uint32_t subblock_wt = (block_wt <= 2) ? subblock_wt_volatile : subblock_wt_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

    constexpr uint32_t cb_im = (do_gamma | do_beta) ? cb_x : cb_out;
    constexpr uint32_t cb_outgamma = do_beta ? cb_fusion : cb_out;

    // pre-add x + y
    if constexpr (fuse_pre_add) {
        reconfig_data_format_srcb(cb_in0, cb_in1);
        add_tiles_init(cb_in0, cb_in1);
        cb_reserve_back(cb_x, num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
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
                    pack_tile(i, cb_x);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_wt;
            }
            index_h_offset += block_wt;
        }
        cb_push_back(cb_x, num_tiles_per_block);
        reconfig_data_format(cb_in0, cb_x, cb_in1, cb_scaler);
        cb_wait_front(cb_x, num_tiles_per_block);
    } else {
        reconfig_data_format_srcb(cb_in0, cb_scaler);
    }

    // Compute E[x] and Var[x] using Welford's algorithm
    constexpr uint32_t block_w = block_wt * tile_width;
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_x, num_tiles_per_block);
    cb_reserve_back(cb_ex_partial, block_h);
    cb_reserve_back(cb_varx_partial, block_h);
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
            welford_tile<in_dst, mean_dst, var_dst, true, false>(w * tile_width, block_w, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(mean_dst, cb_ex_partial);
        pack_tile(var_dst, cb_varx_partial);
        tile_regs_release();
        index_h_offset += block_w;
    }
    cb_push_back(cb_ex_partial, block_h);
    cb_push_back(cb_varx_partial, block_h);

    // x - E[x]
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_x, cb_ex_global);
    }
    index_h_offset = 0;
    reconfig_data_format_srca(cb_x);
    sub_bcast_cols_init_short(cb_x, cb_ex_global);
    cb_reserve_back(cb_xmm, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        cb_wait_front(cb_ex_global, 1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(cb_x, cb_ex_global, index, 0, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_wt; i++) {
                pack_tile(i, cb_xmm);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_wt;
        }
        cb_pop_front(cb_ex_global, 1);
        cb_pop_front(cb_x, block_wt);
    }
    cb_push_back(cb_xmm, num_tiles_per_block);

    cb_wait_front(cb_xmm, num_tiles_per_block);

    // Compute 1/[sqrt(Var + eps)]
    if constexpr (is_allgather_worker && enable_sqrt) {
        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            // 1/[sqrt(Var + eps)],
            cb_wait_front(cb_ex2, 1);
            cb_reserve_back(cb_ex2pe, 1);
            tile_regs_acquire();
            add_tiles_init(cb_ex2, cb_eps);
            add_tiles(cb_ex2, cb_eps, i, 0, dst0);
            tile_regs_wait();
            rsqrt_tile_init<true>();
            rsqrt_tile<true>(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe);
            cb_push_back(cb_ex2pe, 1);
            tile_regs_release();
        }
    }

    if constexpr (do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(cb_out);
    }

    // Compute (x - E[x]) * 1/[sqrt(Var + eps)]
    // The 1/[sqrt(Var + eps)] factors are sent back in cb_ex_global
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_ex_global);
    }

    cb_wait_front(cb_xmm, num_tiles_per_block);
    mul_bcast_cols_init_short(cb_xmm, cb_ex_global);
    index_h_offset = 0;
    cb_reserve_back(cb_im, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        cb_wait_front(cb_ex_global, 1);
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_wt; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles_bcast_cols(cb_xmm, cb_ex_global, index, 0, w);
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
        cb_wait_front(cb_gamma, block_wt);
        index_h_offset = 0;
        cb_reserve_back(cb_outgamma, num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
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
        for (uint32_t i = 0; i < block_h; i++) {
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
