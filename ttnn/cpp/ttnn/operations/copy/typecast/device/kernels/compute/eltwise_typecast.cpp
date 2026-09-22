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
    // Passed explicitly rather than read off the buffers since an Int8 tensor is carried in a UInt8
    // buffer.
    constexpr uint32_t in_data_format = get_arg(args::in_data_format);
    constexpr uint32_t out_data_format = get_arg(args::out_data_format);

    compute_kernel_hw_startup(dfb::in, dfb::out);

    // The raw kernel owned one output window per outer block: reserve and publish
    // per_core_block_dim output pages around its per-tile typecast walk. PerOuter expresses
    // that directly; the input retains its raw per-tile wait/pop lifecycle.
    constexpr auto input =
        ckl::input(dfb::in, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled);
    constexpr auto output = ckl::output(
        dfb::out, ckl::ReservePolicy::PerOuter, ckl::PushPolicy::PerOuter, ckl::DataFormatReconfig::Disabled);
    ckl::unary<ckl::Typecast<in_data_format, out_data_format>, input, output>(
        ckl::IterationShape::grid(per_core_block_cnt, per_core_block_dim));
}
