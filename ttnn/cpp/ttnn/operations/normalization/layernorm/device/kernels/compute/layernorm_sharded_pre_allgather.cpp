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
    constexpr uint32_t num_blocks_first_stage                     = get_compile_time_arg_val(3);
    constexpr uint32_t block_w                        = get_compile_time_arg_val(5);
    constexpr uint32_t block_h_const                  = get_compile_time_arg_val(4);
    volatile uint32_t block_h_volatile                = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w_const               = get_compile_time_arg_val(6);
    volatile uint32_t subblock_w_volatile             = get_compile_time_arg_val(6);
    constexpr uint32_t num_subblocks_w                = get_compile_time_arg_val(7);
    const bool is_allgather_worker                    = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t num_tiles_per_block            = get_compile_time_arg_val(9);
    constexpr bool FLOAT32_DTYPE                      = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t num_blocks_second_stage        = get_compile_time_arg_val(11);

    const uint32_t num_reduce_tiles_per_block_h             = get_arg_val<uint32_t>(0); // This value is the same for all cores, except ones that have padding tiles in it. In that case, skip reduce for padding tiles.
    const uint32_t num_tiles_per_allgather_worker           = is_allgather_worker ? get_arg_val<uint32_t>(1) : 0;
    const bool use_two_stage_reduce                         = is_allgather_worker ? get_arg_val<uint32_t>(2) == 1 : false;
    const bool is_second_stage_reader                       = is_allgather_worker ? get_arg_val<uint32_t>(3) == 1 : false;

    uint32_t num_blocks_reduce;
    if (is_second_stage_reader) {
        num_blocks_reduce = num_blocks_first_stage + num_blocks_second_stage - 1;
    } else {
        num_blocks_reduce = num_blocks_first_stage;
    }

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_scaler = tt::CB::c_in2;
    constexpr uint32_t cb_scaler_global = tt::CB::c_in4;
    constexpr uint32_t cb_x = tt::CB::c_intermed0; // x minus mean
    constexpr uint32_t cb_ex = tt::CB::dataflow1; // E[x] global reduce

    constexpr uint32_t cb_ex2 = tt::CB::dataflow4; // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_x2 = cb_x; // x^2
    constexpr uint32_t cb_out = tt::CB::c_out0;

    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3; // E[x^2] partial reduce
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5; // E[x^2] partials recieved from other cores
    const uint32_t cb_reduction_out = is_second_stage_reader ? cb_out : cb_ex2;


    binary_op_init_common(cb_in0, cb_in0, cb_x2);

    // set block_h to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_h = (block_w == 1) ? block_h_volatile : block_h_const;
    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

    uint32_t num_tiles_per_partial_result = 2;
    #ifdef RMSNORM
    num_tiles_per_partial_result = 1;
    #endif

    cb_wait_front(cb_scaler, 1);
    #ifndef RMSNORM
    unpack_reconfig_data_format_srcb(cb_in0, cb_scaler);
    // E[x],
    index_h_offset = 0;
    reduce_init_delta<false>();

    cb_reserve_back(cb_ex_partial2, block_h);
    for (uint32_t i = 0; i < block_h; i++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
            reduce_tile(cb_in0, cb_scaler, w+index_h_offset, scaler0, dst0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial2);
        tile_regs_release();
        index_h_offset += block_w;
    }
    reduce_revert_delta();
    cb_push_back(cb_ex_partial2, block_h);
    unpack_reconfig_data_format_srcb(cb_scaler, cb_in0);
    #endif // not RMSNORM

    // X^2
    mul_tiles_init();
    index_h_offset = 0;
    cb_reserve_back(cb_x2, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles(cb_in0, cb_in0, index, index, w);
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
    }
    cb_push_back(cb_x2, num_tiles_per_block);

    // E(x^2)
    unpack_reconfig_data_format_srca(cb_in0, cb_x2);
    unpack_reconfig_data_format_srcb(cb_in0, cb_scaler);

    cb_wait_front(cb_x2, num_tiles_per_block);

    cb_reserve_back(cb_ex_partial2, block_h); // RMS E(x2) #Layernorm //E(x) and E(x^2)

    reduce_init_delta<false>();
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_h; i++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
            reduce_tile(cb_x2, cb_scaler, w+index_h_offset, scaler0, dst0);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial2);
        tile_regs_release();
        index_h_offset += block_w;
    }
    reduce_revert_delta();
    cb_pop_front(cb_x2, num_tiles_per_block);
    cb_push_back(cb_ex_partial2, block_h);

    // global reduce, cb_ex <-- cb_ex_external2, cb_ex_partial2
    if constexpr(is_allgather_worker) {
        cb_wait_front(cb_scaler_global, 1);
        unpack_reconfig_data_format_srca(cb_x2, cb_ex_external2);
        unpack_reconfig_data_format_srcb(cb_scaler, cb_scaler_global);
        reduce_init_delta<false>();
        cb_reserve_back(cb_reduction_out, num_tiles_per_partial_result*num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) { // loops over height
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_tiles_per_partial_result*num_blocks_reduce; w++) { // Need to read this interleaved now, we have SUM(X) and SUM(X^2) interleaved
                cb_wait_front(cb_ex_external2, 1);
                reduce_tile(cb_ex_external2, cb_scaler_global, 0, scaler0, w % num_tiles_per_partial_result); // E(x) and E(x^2) interleaved so we reduce each one into different dest reg
                cb_pop_front(cb_ex_external2, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_reduction_out);
            #ifndef RMSNORM
            pack_tile(dst1, cb_reduction_out);
            #endif
            tile_regs_release();
        }
        reduce_revert_delta();
        cb_push_back(cb_reduction_out, num_tiles_per_partial_result*num_tiles_per_allgather_worker);
    }

}

}
