// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
    constexpr bool is_allgather_worker = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(5);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(7);

    // Circular Buffers Pre
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(8);
    constexpr uint32_t cb_scaler_global = get_compile_time_arg_val(9);
    constexpr uint32_t cb_ex_partial2 = get_compile_time_arg_val(10);
    constexpr uint32_t cb_ex2 = get_compile_time_arg_val(11);
    constexpr uint32_t fuse_preadd_cb_in = get_compile_time_arg_val(12);  // original
    constexpr uint32_t cb_ex_external2 = get_compile_time_arg_val(13);
    constexpr uint32_t cb_to_allgather_writer = get_compile_time_arg_val(14);  // output
    constexpr uint32_t cb_x = get_compile_time_arg_val(15);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(16);  // Residual
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(17);  // Input

    // Circular Buffers Post
    constexpr uint32_t cb_out = get_compile_time_arg_val(18);    // non reshard output or CB to resharder
    constexpr uint32_t cb_stats = get_compile_time_arg_val(19);  // Input Stats Tensor
    constexpr uint32_t cb_xmm = get_compile_time_arg_val(20);    // Input Tensor
    constexpr uint32_t cb_eps = get_compile_time_arg_val(21);
    constexpr uint32_t post_cb_scaler_global = get_compile_time_arg_val(22);
    constexpr uint32_t cb_var = get_compile_time_arg_val(23);
    constexpr uint32_t cb_im = get_compile_time_arg_val(24);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(25);
    constexpr uint32_t cb_stats_reduced = get_compile_time_arg_val(26);
    constexpr uint32_t cb_ex_global = get_compile_time_arg_val(27);
    constexpr uint32_t signaling_cb = get_compile_time_arg_val(28);

    constexpr uint32_t num_blocks_second_stage_reduction = num_blocks_first_stage + num_blocks_second_stage - 1;

    volatile uint32_t subblock_w_volatile = subblock_w_const;

    const uint32_t num_reduce_tiles_per_block_h =
        get_arg_val<uint32_t>(0);  // This value is the same for all cores, except ones that have padding tiles in it.
                                   // In that case, skip reduce for padding tiles.

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;
#ifdef FUSE_PRE_ADD
    constexpr uint32_t cb_in = fuse_preadd_cb_in;
#else
    constexpr uint32_t cb_in = cb_in0;
#endif

    constexpr uint32_t cb_x2 = cb_x;  // x^2

    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

// pre-add x + y
#ifdef FUSE_PRE_ADD
    binary_op_init_common(cb_in0, cb_in1, cb_in);
    reconfig_data_format(cb_in0, cb_in1);
    pack_reconfig_data_format(cb_in);
    reconfig_data_format(cb_in0, cb_in1);
    add_tiles_init(cb_in0, cb_in1);
    cb_reserve_back(cb_in, num_tiles_per_block);
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
    cb_push_back(cb_in, num_tiles_per_block);
    cb_wait_front(cb_in, num_tiles_per_block);
    pack_reconfig_data_format(cb_in, cb_x2);
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_in);
#else
    binary_op_init_common(cb_in, cb_in, cb_x2);
#endif

    // X^2
    mul_tiles_init(cb_in, cb_in);
    index_h_offset = 0;
    cb_reserve_back(cb_x2, num_tiles_per_block);
    index_subblock_w_offset = 0;
    for (uint32_t j = 0; j < num_subblocks_w; j++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < subblock_w; w++) {
            index = w + index_subblock_w_offset + index_h_offset;
            mul_tiles(cb_in, cb_in, index, index, w);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < subblock_w; i++) {
            pack_tile(i, cb_x2);
        }
        tile_regs_release();
        index_subblock_w_offset += subblock_w;
    }
    index_h_offset += block_w;
    cb_push_back(cb_x2, num_tiles_per_block);

    // E(x^2)
    reconfig_data_format_srca(cb_in, cb_x2);
    reconfig_data_format_srcb(cb_in, cb_scaler);

    cb_wait_front(cb_x2, num_tiles_per_block);
    cb_wait_front(cb_scaler, 1);

    cb_reserve_back(cb_ex_partial2, 1);  // RMS E(x2) #Layernorm //E(x) and E(x^2)

    reduce_init(cb_x2, cb_scaler, cb_ex_partial2);
    index_h_offset = 0;
    tile_regs_acquire();
    for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
        reduce_tile(cb_x2, cb_scaler, w + index_h_offset, scaler0, dst0);
    }

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, cb_ex_partial2);
    tile_regs_release();
    index_h_offset += block_w;
    reduce_uninit();
    cb_pop_front(cb_x2, num_tiles_per_block);
    cb_push_back(cb_ex_partial2, 1);

    // global reduce, cb_ex <-- cb_ex_external2, cb_ex_partial2
    if constexpr (is_allgather_worker) {
        const uint32_t num_tiles_per_allgather_worker = get_arg_val<uint32_t>(1);
        const bool use_two_stage_reduce = get_arg_val<uint32_t>(2) == 1;
        const bool is_second_stage_reader = get_arg_val<uint32_t>(3) == 1;
        uint32_t num_blocks_reduce;
        num_blocks_reduce = (is_second_stage_reader) ? num_blocks_second_stage_reduction : num_blocks_first_stage;
        const uint32_t cb_reduction_out =
            (!use_two_stage_reduce or is_second_stage_reader) ? cb_to_allgather_writer : cb_ex2;
        cb_wait_front(cb_scaler_global, 1);
        reconfig_data_format_srca(cb_x2, cb_ex_external2);
        reconfig_data_format_srcb(cb_scaler, cb_scaler_global);
        reduce_init(cb_ex_external2, cb_scaler_global, cb_reduction_out);
        cb_reserve_back(cb_reduction_out, num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {  // loops over height
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks_reduce;
                 w++) {  // Need to read this interleaved now, we have SUM(X) and SUM(X^2) interleaved
                cb_wait_front(cb_ex_external2, 1);
                reduce_tile(
                    cb_ex_external2,
                    cb_scaler_global,
                    0,
                    scaler0,
                    0);  // E(x) and E(x^2) interleaved so we reduce each one into
                         // different dest reg
                cb_pop_front(cb_ex_external2, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_reduction_out);
            tile_regs_release();
        }
        reduce_uninit();
        cb_push_back(cb_reduction_out, num_tiles_per_allgather_worker);
    }

    // Waits for stats tensor to have valid data
    cb_wait_front(signaling_cb, 1);
    cb_pop_front(signaling_cb, 1);
    constexpr uint32_t post_dst0 = 0;
    constexpr uint32_t post_scaler0 = 0;
    binary_op_init_common(cb_stats, post_cb_scaler_global, cb_var);
    index_subblock_w_offset = 0;
    index_h_offset = 0;
    index = 0;

    constexpr uint32_t cb_outgamma = cb_out;
    if constexpr (is_allgather_worker) {
        const bool enable_sqrt = get_arg_val<uint32_t>(4) == 1;
        if (enable_sqrt) {
            uint32_t num_distributed_blocks = get_arg_val<uint32_t>(5);
            cb_reserve_back(cb_var, 1);
            cb_wait_front(post_cb_scaler_global, 1);
            reduce_init(cb_stats, post_cb_scaler_global, cb_var);
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_distributed_blocks; w++) {
                reduce_tile(
                    cb_stats,
                    post_cb_scaler_global,
                    0,
                    post_scaler0,
                    0);  // reducing E(x) and E(x^2) separately to different dst
                cb_pop_front(cb_stats, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(post_dst0, cb_var);
            tile_regs_release();
            reduce_uninit();
            cb_push_back(cb_var, 1);

            // 1/[sqrt(Var + eps)],
            reconfig_data_format(cb_var, cb_eps);  // cb_var is cb_stats in case of RMS norm
            pack_reconfig_data_format(cb_stats_reduced);
            cb_wait_front(cb_var, 1);
            cb_wait_front(cb_eps, 1);

            add_tiles_init(cb_var, cb_eps);
            tile_regs_acquire();
            add_tiles(cb_var, cb_eps, 0, 0, post_dst0);
            tile_regs_wait();
            sqrt_tile_init();
            sqrt_tile(post_dst0);
            recip_tile_init();
            recip_tile(post_dst0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_stats_reduced, 1);
            pack_tile(post_dst0, cb_stats_reduced);
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
