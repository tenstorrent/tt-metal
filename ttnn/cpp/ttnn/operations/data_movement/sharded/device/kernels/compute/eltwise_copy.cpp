// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto dfb_in_id = tt::CBIndex::c_0;
    constexpr auto dfb_out_id = tt::CBIndex::c_16;

    compute_kernel_hw_startup(dfb_in_id, dfb_out_id);

    compute_kernel_lib::copy<
        compute_kernel_lib::input(
            dfb_in_id,
            compute_kernel_lib::WaitPolicy::PerTile,
            compute_kernel_lib::PopPolicy::PerTile,
            compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::output(
            dfb_out_id,
            compute_kernel_lib::ReservePolicy::PerTile,
            compute_kernel_lib::PushPolicy::PerTile,
            compute_kernel_lib::DataFormatReconfig::Disabled)>(
        compute_kernel_lib::IterationShape::tiles(per_core_tile_cnt));
}
