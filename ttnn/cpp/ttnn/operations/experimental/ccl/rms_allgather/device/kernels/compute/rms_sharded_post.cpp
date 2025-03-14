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
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(0);
    constexpr uint32_t block_w = get_compile_time_arg_val(1);
    constexpr uint32_t subblock_w_const = get_compile_time_arg_val(2);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(3);
    constexpr bool is_allgather_worker = get_compile_time_arg_val(4) == 1;  // TODO FIX THIS
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(5);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(7);

    volatile uint32_t subblock_w_volatile = subblock_w_const;

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
    constexpr uint32_t cb_stats_reduced = tt::CBIndex::c_21;  // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;      // E[x] global reduce
    constexpr uint32_t cb_reciprocal = tt::CBIndex::c_20;     // [E[x^2]-E[x]^2]+eps
    constexpr uint32_t cb_fusion = tt::CBIndex::c_18;         // stream gamma/beta
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_var = tt::CBIndex::c_19;
    constexpr uint32_t cb_ex_sqr = tt::CBIndex::c_24;  // E[x]^2

    binary_op_init_common(cb_stats, cb_scaler_global, cb_var);
    constexpr uint32_t stats_tiles = 1;
    constexpr uint32_t cb_xmm = cb_in0;  // x

    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

    constexpr uint32_t cb_im = cb_ex_sqr;
    constexpr uint32_t cb_outgamma = cb_out;

    // global reduce, cb_ex <-- cb_ex_external, cb_ex_partial
    if constexpr (is_allgather_worker) {
        const bool enable_sqrt = get_arg_val<uint32_t>(0) == 1;
        if (enable_sqrt) {
            uint32_t num_distributed_blocks = get_arg_val<uint32_t>(1);
            cb_reserve_back(cb_var, 1);

            cb_wait_front(cb_scaler_global, 1);
            reduce_init_delta<false>(cb_stats, cb_scaler_global, cb_var);
            tile_regs_acquire();
            // striding over cb_stats, consisting [E(X), E(X^2)] from all the distributed devices in interleaved order
            for (uint32_t w = 0; w < num_distributed_blocks; w++) {
                reduce_tile(
                    cb_stats,
                    cb_scaler_global,
                    0,
                    scaler0,
                    0);  // reducing E(x) and E(x^2) separately to different dst
                cb_pop_front(cb_stats, 1);
            }
            tile_regs_commit();
            tile_regs_wait();

            pack_tile(dst0, cb_var);
            tile_regs_release();
            reduce_revert_delta(cb_var);
            cb_push_back(cb_var, stats_tiles);

            // 1/[sqrt(Var + eps)],
            reconfig_data_format(cb_var, cb_eps);  // cb_var is cb_stats in case of RMS norm
            pack_reconfig_data_format(cb_stats_reduced);
            cb_wait_front(cb_var, 1);
            cb_wait_front(cb_eps, 1);
            cb_reserve_back(cb_stats_reduced, 1);

            add_tiles_init(cb_var, cb_eps);
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
    pack_reconfig_data_format(cb_im);

    // (x - Ex) * 1/[sqrt(Var + eps)]
    reconfig_data_format(cb_xmm, cb_ex_global);
    mul_bcast_cols_init_short(cb_xmm, cb_ex_global);
    index_h_offset = 0;
    cb_reserve_back(cb_im, num_tiles_per_block);
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
    cb_push_back(cb_im, num_tiles_per_block);

    cb_pop_front(cb_xmm, num_tiles_per_block);
    cb_wait_front(cb_im, num_tiles_per_block);

    reconfig_data_format(cb_im, cb_gamma);
    pack_reconfig_data_format(cb_out);
    mul_bcast_rows_init_short(cb_im, cb_gamma);
    cb_wait_front(cb_gamma, block_w);
    index_h_offset = 0;
    cb_reserve_back(cb_outgamma, num_tiles_per_block);
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
        cb_push_back(cb_outgamma, subblock_w);
    }
    index_h_offset += block_w;
    cb_pop_front(cb_im, num_tiles_per_block);
}

}  // namespace NAMESPACE
