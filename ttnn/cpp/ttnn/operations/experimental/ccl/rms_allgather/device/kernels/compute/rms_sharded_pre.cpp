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
    const bool is_allgather_worker = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(5);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(7);

    constexpr uint32_t cb_scaler = get_compile_time_arg_val(8);
    constexpr uint32_t cb_scaler_global = get_compile_time_arg_val(9);
    constexpr uint32_t cb_ex_partial2 = get_compile_time_arg_val(10);
    constexpr uint32_t cb_ex2 = get_compile_time_arg_val(11);
    constexpr uint32_t fuse_preadd_cb_in = get_compile_time_arg_val(12);
    constexpr uint32_t cb_ex_external2 = get_compile_time_arg_val(13);
    constexpr uint32_t cb_to_allgather_writer = get_compile_time_arg_val(14);  // output
    constexpr uint32_t cb_x = get_compile_time_arg_val(15);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(16);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(17);  // Input

    constexpr uint32_t num_blocks_second_stage_reduction = num_blocks_first_stage + num_blocks_second_stage - 1;

    volatile uint32_t subblock_w_volatile = subblock_w_const;

    const uint32_t num_reduce_tiles_per_block_h =
        get_arg_val<uint32_t>(0);  // This value is the same for all cores, except ones that have padding tiles in it.
                                   // In that case, skip reduce for padding tiles.

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
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
#else
    binary_op_init_common(cb_in, cb_in, cb_x2);
#endif

#ifdef FUSE_PRE_ADD
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_in);
#endif

    // X^2
    mul_tiles_init(cb_in0, cb_in0);
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

    reduce_init_delta<false>(cb_x2, cb_scaler, cb_ex_partial2);
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
    reduce_revert_delta(cb_ex_partial2);
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
        reduce_init_delta<false>(cb_ex_external2, cb_scaler_global, cb_reduction_out);
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
        reduce_revert_delta(cb_reduction_out);
        cb_push_back(cb_reduction_out, num_tiles_per_allgather_worker);
    }
}

}  // namespace NAMESPACE
