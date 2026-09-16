// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_dim = get_arg(args::per_core_block_dim);

    compute_kernel_hw_startup(dfb::in, dfb::out);

    // The raw kernel owned one output window per outer block: reserve and publish
    // per_core_block_dim output pages around its per-tile typecast walk. PerOuter expresses
    // that directly; the input retains its raw per-tile wait/pop lifecycle.
    constexpr auto input =
        ckl::input(dfb::in, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled);
    ckl::typecast<
        input,
        ckl::output(
            dfb::out, ckl::ReservePolicy::PerOuter, ckl::PushPolicy::PerOuter, ckl::DataFormatReconfig::Disabled)>(
        ckl::IterationShape::grid(per_core_block_cnt, per_core_block_dim));
}
