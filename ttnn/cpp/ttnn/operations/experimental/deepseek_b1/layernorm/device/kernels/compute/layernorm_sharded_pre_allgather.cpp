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
    constexpr bool FLOAT32_REDUCTION = get_compile_time_arg_val(11) == 1;
    // LEGACY_RSQRT at index 12 is not used but needed for consistency across sharded compute kernels
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(13);

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

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_in = cb_in0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_scaler_global = tt::CBIndex::c_4;
    constexpr uint32_t cb_x = tt::CBIndex::c_24;  // x minus mean
    constexpr uint32_t cb_ex = tt::CBIndex::c_9;  // E[x] global reduce

    constexpr uint32_t cb_ex2 = tt::CBIndex::c_12;  // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_x2 = cb_x;                // x^2
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t cb_ex_partial2 = tt::CBIndex::c_11;   // E[x^2] partial reduce
    constexpr uint32_t cb_ex_external2 = tt::CBIndex::c_13;  // E[x^2] partials received from other cores
    const uint32_t cb_reduction_out = (!use_two_stage_reduce or is_second_stage_reader) ? cb_out : cb_ex2;

    // set block_h to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_h = (block_w == 1) ? block_h_volatile : block_h_const;
    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

    uint32_t num_tiles_per_partial_result = 2;
    num_tiles_per_partial_result = 1;

    // pre-add x + y
    binary_op_init_common(cb_in, cb_in, cb_x2);

    reconfig_data_format_srcb(cb_in, cb_scaler);
    pack_tile(dst1, cb_reduction_out);
