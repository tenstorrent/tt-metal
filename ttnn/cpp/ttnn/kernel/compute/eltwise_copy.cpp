// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in, cb_out);

    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(per_core_tile_cnt),
        compute_kernel_lib::CopyTile<compute_kernel_lib::input(
            cb_in,
            compute_kernel_lib::WaitPolicy::PerTile,
            compute_kernel_lib::PopPolicy::PerTile,
            compute_kernel_lib::DataFormatReconfig::Disabled)>{},
        compute_kernel_lib::PackTile<compute_kernel_lib::output(
            cb_out,
            compute_kernel_lib::ReservePolicy::PerTile,
            compute_kernel_lib::PushPolicy::PerTile,
            compute_kernel_lib::DataFormatReconfig::Disabled)>{});
}
