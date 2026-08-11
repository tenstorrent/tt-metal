// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes distributed rmsnorm statistics: E(x**2).
 */

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // add
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

namespace ckl = compute_kernel_lib;

// The statistics pass reads either the raw input or the fused a + b result, depending on whether a
// residual was supplied. Only the buffer selected here is bound on this build.
#ifdef FUSE_PRE_ADD
constexpr auto dfb_inp_id = dfb::fused;
#else
constexpr auto dfb_inp_id = dfb::in0;
#endif

void kernel_main() {
    constexpr auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto blk = get_arg(args::blk);
    constexpr auto num_cores_y = get_arg(args::num_cores_y);
    constexpr bool unpack_fp32_active = get_arg(args::unpack_fp32_active) != 0;
    // Accurate mode only supports SUM; with the reader's scaler of 1.0, SUM and AVG are equivalent.
    constexpr auto reduce_type = unpack_fp32_active ? PoolType::SUM : PoolType::AVG;
    constexpr auto reduce_fp32_mode = unpack_fp32_active ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;

#ifdef FUSE_PRE_ADD
    compute_kernel_hw_startup(dfb::in0, dfb::res, dfb_inp_id);
#else
    compute_kernel_hw_startup(dfb_inp_id, dfb::reduce, dfb::x2);
#endif

    constexpr auto squaring_shape = ckl::IterationShape::tiles(Wt).block_size(blk);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#ifdef FUSE_PRE_ADD
        if constexpr (unpack_fp32_active) {
            ckl::binary_sfpu<
                ckl::AddBinary<>,
                ckl::input(
                    dfb::in0, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
                ckl::input(
                    dfb::res, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
                ckl::output(dfb_inp_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                squaring_shape);
        } else {
            ckl::add<
                ckl::input(
                    dfb::in0, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
                ckl::input(
                    dfb::res, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
                ckl::output(dfb_inp_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                squaring_shape);
        }
#endif

        if constexpr (unpack_fp32_active) {
            ckl::unary<
                ckl::Square<>,
                ckl::input(dfb_inp_id, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::OperandKind::Block),
                ckl::output(dfb::x2, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(squaring_shape);
        } else {
            ckl::square<
                ckl::input(dfb_inp_id, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::OperandKind::Block),
                ckl::output(dfb::x2, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(squaring_shape);
        }

        compute_kernel_lib::reduce<
            reduce_type,
            ReduceDim::REDUCE_ROW,
            dfb::x2,
            dfb::reduce,
            dfb::out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            reduce_fp32_mode>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        DataflowBuffer(dfb_inp_id).pop_front(Wt);
        DataflowBuffer(dfb::reduce).pop_front(1);
    }

#ifdef IS_MERGE_CORE
    if constexpr (unpack_fp32_active) {
        DataflowBuffer dfb_x2_merge(dfb::x2_merge);
        DataflowBuffer dfb_out_final(dfb::out_final);
        DataflowBuffer dfb_zero(dfb::zero);
        constexpr int dst0 = 0;

        // Wait for all num_cores_y tiles
        dfb_x2_merge.wait_front(num_cores_y);
        dfb_zero.wait_front(1);

        // Initialize accumulation
        // TODO(#52395): compute_kernel_hw_startup is a call-once API; this mid-kernel re-init (preserving the
        // pre-cleanup full-init behaviour) should become a targeted DST re-arm.
        compute_kernel_hw_startup(dfb::x2_merge, dfb::zero, dfb::out_final);
        reconfig_data_format(dfb::x2_merge, dfb::zero);
        pack_reconfig_data_format(dfb::out_final);
        // Add all the column's partials together. The accurate path sums them in Dest on the SFPU;
        // add_tiles would pull each through SrcA/SrcB and round it to TF32.
        if constexpr (unpack_fp32_active) {
            copy_tile_to_dst_init_short(dfb::x2_merge);
            add_binary_tile_init();
        } else {
            add_init(dfb::x2_merge, dfb::zero, true);
        }

        tile_regs_acquire();
        if constexpr (unpack_fp32_active) {
            copy_tile(dfb::x2_merge, 0, dst0);
            for (uint32_t i = 1; i < num_cores_y; i++) {
                copy_tile(dfb::x2_merge, i, dst0 + 1);
                add_binary_tile(dst0, dst0 + 1, dst0);
            }
        } else {
            for (uint32_t i = 0; i < num_cores_y; i++) {
                add_tiles(dfb::x2_merge, dfb::zero, i, 0, dst0);
            }
        }
        tile_regs_commit();

        dfb_x2_merge.pop_front(num_cores_y);

        dfb_out_final.reserve_back(1);

        tile_regs_wait();
        pack_tile(dst0, dfb::out_final);
        tile_regs_release();

        dfb_out_final.push_back(1);
    } else {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(num_cores_y),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb::x2_merge, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::input(dfb::zero, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
                ckl::Dst::D0,
                ckl::DestAccumulation::WholeShape>{},
            ckl::PackTile<ckl::output(
                dfb::out_final,
                ckl::ReservePolicy::PerOuter,
                ckl::PushPolicy::PerOuter,
                ckl::DataFormatReconfig::Enabled,
                ckl::PackRelu::Disabled,
                ckl::L1Accumulation::Disabled,
                ckl::DestAccumulation::WholeShape)>{});
    }
#endif
}
