// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

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
    ckl::add<
        cb_in0,
        cb_in1,
        cb_in,
        ckl::BroadcastDim::None,
        ckl::input(ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Block),
        ckl::input(ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Block),
        ckl::output(ckl::OutputLifecycle::Bulk)>(ckl::EltwiseShape::tiles(num_tiles_per_block, subblock_w));
    index_h_offset += block_w;
    cb_wait_front(cb_in, num_tiles_per_block);
    pack_reconfig_data_format(cb_in, cb_x2);
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_in);
#else
    binary_op_init_common(cb_in, cb_in, cb_x2);
#endif

    ckl::square<
        cb_in,
        cb_x2,
        ckl::input(ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Block),
        ckl::output(ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled)>(
        ckl::EltwiseShape::tiles(num_tiles_per_block, subblock_w));

    // E(x^2)
    reconfig_data_format(cb_scaler, cb_x2);

    cb_wait_front(cb_x2, num_tiles_per_block);
    cb_wait_front(cb_scaler, 1);

    cb_reserve_back(cb_ex_partial2, 1);  // RMS E(x2) #Layernorm //E(x) and E(x^2)

    reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_x2, cb_scaler, cb_ex_partial2);
    index_h_offset = 0;
    tile_regs_acquire();
    for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
        // TODO(#38448): Temporary workaround pending further debug; do not copy this pattern elsewhere.
        tensix_sync();
        reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_x2, cb_scaler, w + index_h_offset, scaler0, dst0);
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
        const auto reduce_block = ckl::ReduceInputBlockShape::of(num_tiles_per_allgather_worker, num_blocks_reduce);

        if (!use_two_stage_reduce || is_second_stage_reader) {
            ckl::reduce<
                PoolType::AVG,
                ReduceDim::REDUCE_ROW,
                cb_ex_external2,
                cb_scaler_global,
                cb_to_allgather_writer,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT>(reduce_block);
        } else {
            ckl::reduce<
                PoolType::AVG,
                ReduceDim::REDUCE_ROW,
                cb_ex_external2,
                cb_scaler_global,
                cb_ex2,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT>(reduce_block);
        }
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

            ckl::reduce<
                PoolType::AVG,
                ReduceDim::REDUCE_ROW,
                cb_stats,
                post_cb_scaler_global,
                cb_var,
                ckl::ReduceInputPolicy::NoWaitNoPop,
                ckl::ReduceDataFormatReconfigMode::INPUT>(ckl::ReduceInputBlockShape::row(num_distributed_blocks));
            cb_pop_front(cb_stats, num_distributed_blocks);

            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::BinaryFpu<cb_var, cb_eps>{},
                ckl::Rsqrt<ckl::Approx::Exact, ckl::Legacy::On, ckl::Dst::D0>{},
                ckl::PackTile<cb_stats_reduced>{});
        }
    }
    ckl::mul<
        cb_xmm,
        cb_ex_global,
        cb_im,
        ckl::BroadcastDim::Col,
        ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
        ckl::input(ckl::InputLifecycle::Bulk),
        ckl::output(ckl::OutputLifecycle::Bulk)>(ckl::EltwiseShape::tiles(num_tiles_per_block, subblock_w));

    ckl::mul<
        cb_im,
        cb_gamma,
        cb_outgamma,
        ckl::BroadcastDim::Row,
        ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
        ckl::input(ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
        ckl::output(ckl::OutputLifecycle::ReserveAllPushPerChunk)>(
        ckl::EltwiseShape::tiles(num_tiles_per_block, subblock_w));
}
