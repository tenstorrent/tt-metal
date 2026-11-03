// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"  // FillInt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/int.hpp"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    compute_kernel_hw_startup(dfb::in_a, dfb::out);

    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t scalar_arg = get_arg(args::scalar_arg);
    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);  // set to 1

    DataflowBuffer dfb_in0(dfb::in_a);
    DataflowBuffer dfb_in1(dfb::in_b);
    DataflowBuffer dfb_in2(dfb::in_c);
    DataflowBuffer dfb_out(dfb::out);

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles).block_size(num_tiles_per_cycle),
        ckl::CopyTile<
            ckl::input(
                dfb::in_a,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{dfb_in0},
        ckl::CopyTile<
            ckl::input(
                dfb::in_b,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{dfb_in1},
        ckl::CopyTile<
            ckl::input(
                dfb::in_c,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D2>{dfb_in2},
        ckl::FillInt<ADDCMUL_DATA_FORMAT, ckl::Dst::D3>{scalar_arg},
        ckl::MulIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D3, ckl::Dst::D1, ckl::Dst::D3>{},  // D3 = scalar*in1
        ckl::MulIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D3, ckl::Dst::D2, ckl::Dst::D2>{},  // D2 = D3*in2
        ckl::AddIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D0>{},  // D0 = in0 + D2
        ckl::PackTile<ckl::output(
            dfb::out,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{dfb_out});
}
