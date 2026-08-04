// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    auto B = get_arg(args::B);
    auto Ht = get_arg(args::Ht);
    auto Wt = get_arg(args::Wt);

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);

    ckl::eltwise_chain(
        ckl::EltwiseShape::grid(B * Ht, Wt),
        ckl::BinaryFpu<  // cb_lhs: one tile per (row,col)
            ckl::input(
                dfb::in0,
                ckl::WaitPolicy::PerTile,
                ckl::PopPolicy::PerTile,
                ckl::DataFormatReconfig::Disabled),  // cb_rhs: streamed
                                                     // broadcast, one
                                                     // per row
            ckl::input(dfb::in1, ckl::WaitPolicy::PerOuter, ckl::PopPolicy::PerOuter, ckl::DataFormatReconfig::Disabled),
            CHAIN_BCAST_OP,
            CHAIN_BCAST_DIM>{},
        ckl::PackTile<ckl::output(
            dfb::out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}
