// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"  // FillInt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/int.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    constexpr auto dfb_in0_id = tt::CBIndex::c_0;
    constexpr auto dfb_in1_id = tt::CBIndex::c_1;
    constexpr auto dfb_in2_id = tt::CBIndex::c_2;
    constexpr auto dfb_out_id = tt::CBIndex::c_3;

    compute_kernel_hw_startup(dfb_in0_id, dfb_out_id);

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles).block_size(num_tiles_per_cycle),
        ckl::CopyTile<
            ckl::input(
                dfb_in0_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyTile<
            ckl::input(
                dfb_in1_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::CopyTile<
            ckl::input(
                dfb_in2_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D2>{},
        ckl::FillInt<ADDCMUL_DATA_FORMAT, ckl::Dst::D3>{scalar_arg},
        ckl::MulIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D3, ckl::Dst::D1, ckl::Dst::D3>{},  // D3 = scalar*in1
        ckl::MulIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D3, ckl::Dst::D2, ckl::Dst::D2>{},  // D2 = D3*in2
        ckl::AddIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D0>{},  // D0 = in0 + D2
        ckl::PackTile<ckl::output(
            dfb_out_id,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
}
