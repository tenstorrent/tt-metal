// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
For rmsnorm it computes E(x**2) and returns it as a one tile wide output
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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // add
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

namespace ckl = compute_kernel_lib;

// The statistics pass reads either the raw input or the fused a + b result, depending on whether a
// residual was supplied. Only the buffer selected here is bound on this build, so the alias is gated
// at the preprocessor: naming an unbound handle would not compile even on a discarded branch.
#ifdef FUSE_PRE_ADD
constexpr auto dfb_inp_id = dfb::fused;  // fused a + b
#else
constexpr auto dfb_inp_id = dfb::in0;  // just a
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
    DataflowBuffer dfb_inp(dfb_inp_id);
    DataflowBuffer dfb_reduce(dfb::reduce);
    constexpr uint32_t onetile = 1;
#ifdef IS_MERGE_CORE
    DataflowBuffer dfb_zero(dfb::zero);
#endif
#ifdef FUSE_PRE_ADD
    constexpr auto in0_input =
        ckl::input(dfb::in0, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block);
    constexpr auto res_input =
        ckl::input(dfb::res, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block);
#endif
    constexpr auto input_squared =
        ckl::input(dfb_inp_id, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::InputTileMapping::Block);

#ifdef FUSE_PRE_ADD
    compute_kernel_hw_startup(dfb::in0, dfb::res, dfb_inp_id);
#else
    compute_kernel_hw_startup(dfb_inp_id, dfb::reduce, dfb::x2);
#endif

    constexpr auto squaring_shape = ckl::IterationShape::tiles(Wt).block_size(blk);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Fuse pre-add: dfb_inp = dfb::in0 + dfb::res (absent entirely when there is no residual)
#ifdef FUSE_PRE_ADD
        if constexpr (unpack_fp32_active) {
            ckl::binary_sfpu<
                ckl::AddBinary<>,
                in0_input,
                res_input,
                ckl::output(dfb_inp_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                squaring_shape);
        } else {
            ckl::add<
                in0_input,
                res_input,
                ckl::output(dfb_inp_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                squaring_shape);
        }
#endif

        if constexpr (unpack_fp32_active) {
            ckl::unary<
                ckl::Square<>,
                input_squared,
                ckl::output(dfb::x2, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(squaring_shape);
        } else {
            ckl::square<
                input_squared,
                ckl::output(dfb::x2, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(squaring_shape);
        }

        // BulkWaitBulkPop: All Wt tiles already in the buffer (see cumulative wait above)
        compute_kernel_lib::reduce<
            reduce_type,
            ReduceDim::REDUCE_ROW,
            dfb::x2,
            dfb::reduce,
            dfb::out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            reduce_fp32_mode>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        dfb_inp.pop_front(Wt);
        dfb_reduce.pop_front(1);
    }

    // On a merge core, do a final sum over the column's partial statistics and write the result to
    // the output buffer. Only the merge-core build binds that output buffer, so the whole block is
    // gated at the preprocessor rather than on a runtime flag.
#ifdef IS_MERGE_CORE
    {
        DataflowBuffer dfb_x2_merge(dfb::x2_merge);
        DataflowBuffer dfb_out_final(dfb::out_final);
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
            copy_init(dfb::x2_merge);
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

        dfb_out_final.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(dst0, dfb::out_final);
        tile_regs_release();

        dfb_out_final.push_back(onetile);
    }
#endif
}
