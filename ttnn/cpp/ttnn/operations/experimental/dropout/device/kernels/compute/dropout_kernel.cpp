// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"  // Dropout owns PRNG initialization.
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_dim = get_arg(args::per_core_block_dim);
    constexpr uint32_t int_probability = get_arg(args::int_probability);
    constexpr uint32_t int_scale_factor = get_arg(args::int_scale_factor);

    uint32_t seed = get_arg(args::seed);

    DataflowBuffer dfb_in0_obj(dfb::in0);
    DataflowBuffer dfb_out_obj(dfb::out);

    compute_kernel_hw_startup(dfb::in0, dfb::out);

    constexpr uint32_t total_tiles = per_core_block_cnt * per_core_block_dim;
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(total_tiles).block_size(per_core_block_dim),
        ckl::CopyTile<
            ckl::input(dfb::in0, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{dfb_in0_obj},
        ckl::Dropout<ckl::Dst::D0>{int_probability, int_scale_factor, seed},
        ckl::PackTile<ckl::output(
            dfb::out,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{dfb_out_obj});
}
