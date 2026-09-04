// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_arg(args::num_output_tiles);
    constexpr bool wt_need_bcast = (get_arg(args::wt_need_bcast) == 1);
    constexpr bool ht_need_bcast = (get_arg(args::ht_need_bcast) == 1);

    compute_kernel_hw_startup(dfb::in1, dfb::in0, dfb::out0);

    constexpr bool has_bcast = ht_need_bcast || wt_need_bcast;
    constexpr auto bcast_dim = (ht_need_bcast && wt_need_bcast) ? ckl::BroadcastDim::Scalar
                               : ht_need_bcast                  ? ckl::BroadcastDim::Row
                               : wt_need_bcast                  ? ckl::BroadcastDim::Col
                                                                : ckl::BroadcastDim::None;

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_output_tiles),
        ckl::Optional<
            has_bcast,
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(
                    dfb::in1,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Scalar,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    dfb::in0,
                    bcast_dim,
                    ckl::WaitPolicy::PerTile,
                    ckl::PopPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>>{},
        ckl::Optional<
            !has_bcast,
            ckl::CopyTile<
                ckl::input(
                    dfb::in0, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D0>>{},
        ckl::PackTile<ckl::output(
            dfb::out0, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}
