// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto dfb_input_id = tt::CBIndex::c_0;
    constexpr auto dfb_output_id = tt::CBIndex::c_2;

    compute_kernel_hw_startup(dfb_input_id, dfb_output_id);

    compute_kernel_lib::copy<
        compute_kernel_lib::input(
            dfb_input_id,
            compute_kernel_lib::WaitPolicy::PerTile,
            compute_kernel_lib::PopPolicy::PerTile,
            compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::output(
            dfb_output_id,
            compute_kernel_lib::ReservePolicy::PerTile,
            compute_kernel_lib::PushPolicy::PerTile,
            compute_kernel_lib::DataFormatReconfig::Disabled)>(compute_kernel_lib::IterationShape::tiles(num_tiles));
}
