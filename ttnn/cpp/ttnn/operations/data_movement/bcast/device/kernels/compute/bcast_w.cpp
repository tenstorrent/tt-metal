// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    auto B = get_arg(args::B);
    auto Ht = get_arg(args::Ht);
    auto Wt = get_arg(args::Wt);

    // The factory launches this kernel across the full device grid and assigns zero work to idle
    // cores. Preserve the legacy kernel's no-op behavior instead of forming an empty grid shape.
    if (B == 0 || Ht == 0 || Wt == 0) {
        return;
    }

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);

    ckl::eltwise_chain(
        ckl::IterationShape::grid(B * Ht, Wt),
        ckl::BinaryFpu<
            CHAIN_BCAST_OP,
            // dfb_lhs_id: one tile per (row,col)
            ckl::input(dfb::in0, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            // dfb_rhs_id: one broadcast tile per row
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
