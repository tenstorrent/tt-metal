// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

namespace ckl = compute_kernel_lib;

// SPLIT REDUCE across Cores
void kernel_main() {
    constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(0);
    constexpr uint32_t block_w = get_compile_time_arg_val(1);
    constexpr uint32_t subblock_w_const = get_compile_time_arg_val(2);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(3);
    constexpr bool is_allgather_worker = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(5);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t num_blocks_second_stage = get_compile_time_arg_val(7);

    // Circular Buffers Pre
    constexpr uint32_t dfb_scaler_id = get_compile_time_arg_val(8);
    constexpr uint32_t dfb_scaler_global_id = get_compile_time_arg_val(9);
    constexpr uint32_t dfb_ex_partial2_id = get_compile_time_arg_val(10);
    constexpr uint32_t dfb_ex2_id = get_compile_time_arg_val(11);
    constexpr uint32_t fuse_preadd_dfb_in_id = get_compile_time_arg_val(12);  // original
    constexpr uint32_t dfb_ex_external2_id = get_compile_time_arg_val(13);
    constexpr uint32_t dfb_to_allgather_writer_id = get_compile_time_arg_val(14);  // output
    constexpr uint32_t dfb_x_id = get_compile_time_arg_val(15);
    constexpr uint32_t dfb_in1_id = get_compile_time_arg_val(16);  // Residual
    constexpr uint32_t dfb_in0_id = get_compile_time_arg_val(17);  // Input

    // Circular Buffers Post
    constexpr uint32_t dfb_out_id = get_compile_time_arg_val(18);    // non reshard output or DFB to resharder
    constexpr uint32_t dfb_stats_id = get_compile_time_arg_val(19);  // Input Stats Tensor
    constexpr uint32_t dfb_xmm_id = get_compile_time_arg_val(20);    // Input Tensor
    constexpr uint32_t dfb_eps_id = get_compile_time_arg_val(21);
    constexpr uint32_t post_dfb_scaler_global_id = get_compile_time_arg_val(22);
    constexpr uint32_t dfb_var_id = get_compile_time_arg_val(23);
    constexpr uint32_t dfb_im_id = get_compile_time_arg_val(24);
    constexpr uint32_t dfb_gamma_id = get_compile_time_arg_val(25);
    constexpr uint32_t dfb_stats_reduced_id = get_compile_time_arg_val(26);
    constexpr uint32_t dfb_ex_global_id = get_compile_time_arg_val(27);
    constexpr uint32_t signaling_dfb_id = get_compile_time_arg_val(28);

    constexpr uint32_t num_blocks_second_stage_reduction = num_blocks_first_stage + num_blocks_second_stage - 1;

    volatile uint32_t subblock_w_volatile = subblock_w_const;

    const uint32_t num_reduce_tiles_per_block_h =
        get_arg_val<uint32_t>(0);  // This value is the same for all cores, except ones that have padding tiles in it.
                                   // In that case, skip reduce for padding tiles.

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;
#ifdef FUSE_PRE_ADD
    constexpr uint32_t dfb_in_id = fuse_preadd_dfb_in_id;
#else
    constexpr uint32_t dfb_in_id = dfb_in0_id;
#endif

    constexpr uint32_t dfb_x2_id = dfb_x_id;  // x^2

    const uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

// pre-add x + y
#ifdef FUSE_PRE_ADD
    compute_kernel_hw_startup(dfb_in0_id, dfb_in1_id, dfb_in_id);
    ckl::add<
        ckl::input(dfb_in0_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block),
        ckl::input(dfb_in1_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block),
        ckl::output(dfb_in_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::tiles(num_tiles_per_block).block_size(subblock_w));
    index_h_offset += block_w;
    DataflowBuffer(dfb_in_id).wait_front(num_tiles_per_block);
    pack_reconfig_data_format(dfb_in_id, dfb_x2_id);
    reconfig_data_format(dfb_in0_id, dfb_in_id, dfb_in1_id, dfb_in_id);
#else
    compute_kernel_hw_startup(dfb_in_id, dfb_in_id, dfb_x2_id);
#endif

    ckl::square<
        ckl::input(dfb_in_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block),
        ckl::output(dfb_x2_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled)>(
        ckl::IterationShape::tiles(num_tiles_per_block).block_size(subblock_w));

    // E(x^2)
    reconfig_data_format(dfb_scaler_id, dfb_x2_id);

    DataflowBuffer(dfb_x2_id).wait_front(num_tiles_per_block);
    DataflowBuffer(dfb_scaler_id).wait_front(1);

    DataflowBuffer(dfb_ex_partial2_id).reserve_back(1);  // RMS E(x2) #Layernorm //E(x) and E(x^2)

    reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb_x2_id, dfb_scaler_id, dfb_ex_partial2_id);
    index_h_offset = 0;
    tile_regs_acquire();
    for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
        // TODO(#38448): Temporary workaround pending further debug; do not copy this pattern elsewhere.
        tensix_sync();
        reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(dfb_x2_id, dfb_scaler_id, w + index_h_offset, scaler0, dst0);
    }

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, dfb_ex_partial2_id);
    tile_regs_release();
    index_h_offset += block_w;
    reduce_uninit();
    DataflowBuffer(dfb_x2_id).pop_front(num_tiles_per_block);
    DataflowBuffer(dfb_ex_partial2_id).push_back(1);

    // global reduce, dfb_ex_id <-- dfb_ex_external2_id, dfb_ex_partial2_id
    if constexpr (is_allgather_worker) {
        const uint32_t num_tiles_per_allgather_worker = get_arg_val<uint32_t>(1);
        const bool use_two_stage_reduce = get_arg_val<uint32_t>(2) == 1;
        const bool is_second_stage_reader = get_arg_val<uint32_t>(3) == 1;
        uint32_t num_blocks_reduce;
        num_blocks_reduce = (is_second_stage_reader) ? num_blocks_second_stage_reduction : num_blocks_first_stage;
        const auto reduce_block = ckl::ReduceInputBlockShape::of(num_tiles_per_allgather_worker, num_blocks_reduce);

        if (!use_two_stage_reduce || is_second_stage_reader) {
            ckl::reduce<
                PoolType::AVG,
                ReduceDim::REDUCE_ROW,
                dfb_ex_external2_id,
                dfb_scaler_global_id,
                dfb_to_allgather_writer_id,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT>(reduce_block);
        } else {
            ckl::reduce<
                PoolType::AVG,
                ReduceDim::REDUCE_ROW,
                dfb_ex_external2_id,
                dfb_scaler_global_id,
                dfb_ex2_id,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT>(reduce_block);
        }
    }

    // Waits for stats tensor to have valid data
    DataflowBuffer(signaling_dfb_id).wait_front(1);
    DataflowBuffer(signaling_dfb_id).pop_front(1);
    constexpr uint32_t post_dst0 = 0;
    constexpr uint32_t post_scaler0 = 0;
    index_subblock_w_offset = 0;
    index_h_offset = 0;
    index = 0;

    constexpr uint32_t dfb_outgamma_id = dfb_out_id;
    if constexpr (is_allgather_worker) {
        const bool enable_sqrt = get_arg_val<uint32_t>(4) == 1;
        if (enable_sqrt) {
            uint32_t num_distributed_blocks = get_arg_val<uint32_t>(5);

            ckl::reduce<
                PoolType::AVG,
                ReduceDim::REDUCE_ROW,
                dfb_stats_id,
                post_dfb_scaler_global_id,
                dfb_var_id,
                ckl::ReduceInputPolicy::NoWaitNoPop,
                ckl::ReduceDataFormatReconfigMode::INPUT>(ckl::ReduceInputBlockShape::row(num_distributed_blocks));
            DataflowBuffer(dfb_stats_id).pop_front(num_distributed_blocks);

            // Reduce distributed E[x^2], then compute 1/sqrt(E[x^2] + eps).
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<ckl::BinaryFpuOp::Add, ckl::input(dfb_var_id), ckl::input(dfb_eps_id)>{},
                ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::On, ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(dfb_stats_reduced_id)>{});
        }
    }
    // Normalize x with the gathered reciprocal RMS, then apply gamma.
    ckl::mul<
        ckl::input(dfb_xmm_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
        ckl::input(dfb_ex_global_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
        ckl::output(dfb_im_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::tiles(num_tiles_per_block).block_size(subblock_w));

    ckl::mul<
        ckl::input(dfb_im_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
        ckl::input(
            dfb_gamma_id,
            ckl::BroadcastDim::Row,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::None,
            ckl::OperandKind::Block),
        ckl::output(dfb_outgamma_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::PerBlockSize)>(
        ckl::IterationShape::tiles(num_tiles_per_block).block_size(subblock_w));
}
