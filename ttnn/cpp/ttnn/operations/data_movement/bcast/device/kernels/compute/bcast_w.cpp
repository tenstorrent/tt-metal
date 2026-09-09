// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    auto B = get_arg(args::B);
    auto Ht = get_arg(args::Ht);
    auto Wt = get_arg(args::Wt);

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);

    ckl::eltwise_chain(
        ckl::IterationShape::grid(B * Ht, Wt),
        ckl::BinaryFpu<
            CHAIN_BCAST_OP,
            // dfb::in0: one tile per (row,col)
            ckl::input(dfb::in0, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            // dfb::in1: one broadcast tile per row
            ckl::input(
                dfb::in1,
                CHAIN_BCAST_DIM,
                ckl::WaitPolicy::PerTile,
                ckl::PopPolicy::PerTile,
                ckl::InputTileMapping::Col,
                ckl::DataFormatReconfig::Disabled)>{},
        // Output remains one tile per (row,col); only the column-shaped input is streamed per row.
        ckl::PackTile<ckl::output(
            dfb::out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}
