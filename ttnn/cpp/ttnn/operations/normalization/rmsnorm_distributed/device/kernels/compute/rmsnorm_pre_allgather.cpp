// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Produces one E[x^2] tile per row; the scalar statistic occupies the leftmost column.

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

// The statistics pass reads either the raw input or the fused a + b result, depending on whether a
// residual was supplied. Only the buffer selected here is bound on this build, so the alias is gated
// at the preprocessor: naming an unbound handle would not compile even on a discarded branch.
#ifdef FUSE_PRE_ADD
constexpr auto dfb_inp_id = dfb::fused;
#else
constexpr auto dfb_inp_id = dfb::in0;
#endif

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto blk = get_arg(args::blk);
    constexpr bool unpack_fp32_active = get_arg(args::unpack_fp32_active) != 0;
    // Accurate mode only supports SUM; with the reader's scaler of 1.0, SUM and AVG are equivalent.
    constexpr auto reduce_type = unpack_fp32_active ? PoolType::SUM : PoolType::AVG;
    constexpr auto reduce_fp32_mode = unpack_fp32_active ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;
    DataflowBuffer dfb_reduce(dfb::reduce);
#ifdef FUSE_PRE_ADD
    constexpr auto in0_input =
        ckl::input(dfb::in0, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block);
    constexpr auto res_input =
        ckl::input(dfb::res, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block);
#endif
    constexpr auto input_squared =
        ckl::input(dfb_inp_id, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block);

#ifdef FUSE_PRE_ADD
    compute_kernel_hw_startup(dfb::in0, dfb::res, dfb_inp_id);
#else
    compute_kernel_hw_startup(dfb_inp_id, dfb::reduce, dfb::x2);
#endif

    constexpr auto squaring_shape = ckl::IterationShape::grid(Wt / blk, blk);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Fuse pre-add: dfb_inp = dfb::in0 + dfb::res (absent entirely when there is no residual)
#ifdef FUSE_PRE_ADD
        if constexpr (unpack_fp32_active) {
            ckl::binary_sfpu<
                ckl::AddBinary<>,
                in0_input,
                res_input,
                ckl::output(dfb_inp_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(squaring_shape);
        } else {
            ckl::
                add<in0_input, res_input, ckl::output(dfb_inp_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
                    squaring_shape);
        }
#endif

        if constexpr (unpack_fp32_active) {
            ckl::unary<
                ckl::Square<>,
                input_squared,
                ckl::output(dfb::x2, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(squaring_shape);
        } else {
            ckl::square<input_squared, ckl::output(dfb::x2, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
                squaring_shape);
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
    }
    dfb_reduce.pop_front(1);
}
