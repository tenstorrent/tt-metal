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

    // dfb::in  — the typecast source pages, filled by this factory's reader
    // The writer drains dfb::out on interleaved paths; sharded paths may leave it resident in
    // borrowed output storage.
    constexpr uint32_t total_tiles = per_core_block_cnt * per_core_block_dim;
    ckl::typecast<
        ckl::input(dfb::in, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
        ckl::output(
            dfb::out,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>(
        ckl::IterationShape::tiles(total_tiles).block_size(per_core_block_dim));
}
