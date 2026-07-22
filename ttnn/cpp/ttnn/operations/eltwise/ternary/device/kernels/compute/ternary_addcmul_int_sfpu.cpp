// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"  // FillInt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_int.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_in0, cb_out);

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles, num_tiles_per_cycle),
        ckl::CopyTile<
            ckl::input(
                cb_in0, ckl::InputLifecycle::Chunked, ckl::OperandKind::Block, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyTile<
            ckl::input(
                cb_in1, ckl::InputLifecycle::Chunked, ckl::OperandKind::Block, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::CopyTile<
            ckl::input(
                cb_in2, ckl::InputLifecycle::Chunked, ckl::OperandKind::Block, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D2>{},
        ckl::FillInt<ADDCMUL_DATA_FORMAT, ckl::Dst::D3>{scalar_arg},
        ckl::MulIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D3, ckl::Dst::D1, ckl::Dst::D3>{},  // D3 = scalar*in1
        ckl::MulIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D3, ckl::Dst::D2, ckl::Dst::D2>{},  // D2 = D3*in2
        ckl::AddIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D0>{},  // D0 = in0 + D2
        ckl::PackTile<ckl::output(cb_out, ckl::OutputLifecycle::Chunked, ckl::DataFormatReconfig::Disabled)>{});
}
