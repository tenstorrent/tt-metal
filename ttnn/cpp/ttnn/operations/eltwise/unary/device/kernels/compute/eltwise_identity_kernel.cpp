// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    compute_kernel_hw_startup(cb_input, cb_output);

    compute_kernel_lib::copy<
        compute_kernel_lib::input(
            cb_input,
            compute_kernel_lib::WaitPolicy::PerTile,
            compute_kernel_lib::PopPolicy::PerTile,
            compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::output(
            cb_output,
            compute_kernel_lib::ReservePolicy::PerTile,
            compute_kernel_lib::PushPolicy::PerTile,
            compute_kernel_lib::DataFormatReconfig::Disabled)>(compute_kernel_lib::EltwiseShape::tiles(num_tiles));
}
