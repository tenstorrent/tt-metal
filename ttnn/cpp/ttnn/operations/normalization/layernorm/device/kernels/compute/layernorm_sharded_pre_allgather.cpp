// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// SPLIT REDUCE across Cores
void kernel_main() {
    constexpr uint32_t num_blocks_first_stage = get_arg(args::num_blocks_first_stage);
    constexpr uint32_t block_w = get_arg(args::block_wt);
    constexpr uint32_t block_h_const = get_arg(args::block_ht);
    volatile uint32_t block_h_volatile = get_arg(args::block_ht);
    constexpr uint32_t subblock_w_const = get_arg(args::subblock_wt);
    volatile uint32_t subblock_w_volatile = get_arg(args::subblock_wt);
    constexpr uint32_t num_subblocks_w = get_arg(args::num_subblocks_w);
    const bool is_allgather_worker = get_arg(args::is_all_to_all_worker) == 1;
    constexpr uint32_t num_tiles_per_block = get_arg(args::block_ht_block_wt);
    constexpr bool FLOAT32_DTYPE = get_arg(args::fp32_dest_acc_en) == 1;
    constexpr bool FLOAT32_REDUCTION = get_arg(args::float32_reduction) == 1;
    constexpr uint32_t num_blocks_second_stage = get_arg(args::num_blocks_second_stage);

    const uint32_t num_reduce_tiles_per_block_h = get_arg(args::num_reduce_tiles_per_block_h);
    const uint32_t num_tiles_per_allgather_worker = is_allgather_worker ? get_vararg(0) : 0;
    const bool use_two_stage_reduce = is_allgather_worker ? get_vararg(1) == 1 : false;
    const bool is_second_stage_reader = is_allgather_worker ? get_vararg(2) == 1 : false;

    uint32_t num_blocks_reduce;
    if (is_second_stage_reader) {
        num_blocks_reduce = num_blocks_first_stage + num_blocks_second_stage - 1;
    } else {
        num_blocks_reduce = num_blocks_first_stage;
    }

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = dfb::cb_in0;
#ifdef FUSE_PRE_ADD
    constexpr uint32_t cb_in1 = dfb::cb_inb;
    constexpr uint32_t cb_in = dfb::cb_in0_pre;  // c_14 in pre-allgather mode
#else
    constexpr uint32_t cb_in = cb_in0;
#endif
    DataflowBuffer cb_in_obj(cb_in);
    constexpr uint32_t cb_scaler = dfb::cb_scaler;
    constexpr uint32_t cb_scaler_global = dfb::cb_scaler_global;
    constexpr uint32_t cb_x = dfb::cb_x;
    constexpr uint32_t cb_ex = dfb::cb_ex;

    constexpr uint32_t cb_ex2 = dfb::cb_ex2;
    constexpr uint32_t cb_x2 = cb_x;
    constexpr uint32_t cb_out = dfb::cb_out;

    constexpr uint32_t cb_ex_partial2 = dfb::cb_ex_partial2;
    constexpr uint32_t cb_ex_external2 = dfb::cb_ex_external2;
    const uint32_t cb_reduction_out = (!use_two_stage_reduce or is_second_stage_reader) ? cb_out : cb_ex2;

    DataflowBuffer cb_scaler_obj(cb_scaler);
    DataflowBuffer cb_x2_obj(cb_x2);
    DataflowBuffer cb_ex_partial2_obj(cb_ex_partial2);
    DataflowBuffer cb_scaler_global_obj(cb_scaler_global);
    DataflowBuffer cb_ex_external2_obj(cb_ex_external2);

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

// pre-add x + y
#ifdef FUSE_PRE_ADD
    binary_op_init_common(cb_in0, cb_in1, cb_in);
    add_tiles_init(cb_in0, cb_in1);
    cb_in_obj.reserve_back(num_tiles_per_block);
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
    cb_in_obj.push_back(num_tiles_per_block);
    cb_in_obj.wait_front(num_tiles_per_block);
    pack_reconfig_data_format(cb_in, cb_x2);
#else
    binary_op_init_common(cb_in, cb_in, cb_x2);
#endif

#ifndef RMSNORM
    cb_scaler_obj.wait_front(1);
#ifdef FUSE_PRE_ADD
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_scaler);
#else
    reconfig_data_format_srcb(cb_in, cb_scaler);
#endif
    // E[x],
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        cb_in,
        cb_scaler,
        cb_ex_partial2,
        compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    reconfig_data_format(cb_in, cb_in);
#else
#ifdef FUSE_PRE_ADD
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_in);
#endif
#endif  // not RMSNORM

    // X^2
    mul_tiles_init(cb_in0, cb_in0);
    index_h_offset = 0;
    cb_x2_obj.reserve_back(num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
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
    }
    cb_x2_obj.push_back(num_tiles_per_block);

    // E(x^2)
    cb_x2_obj.wait_front(num_tiles_per_block);
#ifdef RMSNORM
    cb_scaler_obj.wait_front(1);
#endif  // RMSNORM

    // RMS E(x2) #Layernorm //E(x) and E(x^2)
    compute_kernel_lib::
        reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
            cb_x2,
            cb_scaler,
            cb_ex_partial2,
            compute_kernel_lib::ReduceInputBlockShape::of(block_h, num_reduce_tiles_per_block_h),
            compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(block_w));
    reconfig_data_format(cb_x2, cb_scaler);
    cb_pop_front(cb_x2, num_tiles_per_block);

    // global reduce, cb_ex <-- cb_ex_external2, cb_ex_partial2
    if constexpr (is_allgather_worker) {
        cb_scaler_global_obj.wait_front(1);
        reconfig_data_format_srca(cb_x2, cb_ex_external2);
        reconfig_data_format_srcb(cb_scaler, cb_scaler_global);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_ex_external2, cb_scaler_global, cb_reduction_out);
        pack_reconfig_data_format(cb_reduction_out);
        DataflowBuffer(cb_reduction_out).reserve_back(num_tiles_per_partial_result * num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {  // loops over height
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_tiles_per_partial_result * num_blocks_reduce;
                 w++) {  // Need to read this interleaved now, we have SUM(X) and SUM(X^2) interleaved
                cb_ex_external2_obj.wait_front(1);
                reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(
                    cb_ex_external2,
                    cb_scaler_global,
                    0,
                    scaler0,
                    w % num_tiles_per_partial_result);  // E(x) and E(x^2) interleaved so we reduce each one into
                                                        // different dest reg
                cb_ex_external2_obj.pop_front(1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_reduction_out);
#ifndef RMSNORM
            pack_tile(dst1, cb_reduction_out);
#endif
            tile_regs_release();
        }
        reduce_uninit();
        DataflowBuffer(cb_reduction_out).push_back(num_tiles_per_partial_result * num_tiles_per_allgather_worker);
    }
}
