// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: A Metal 2.0 fork of this kernel lives beside it, as
// eltwise_copy_metal2.cpp. Ops ported to Metal 2.0 bind the fork; this file serves
// the consumers still on the legacy API. Until the last of them migrates and
// this file is retired, changes here likely belong in the fork too.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr auto dfb_in_id = tt::CBIndex::c_0;
    constexpr auto dfb_out_id = tt::CBIndex::c_16;

    compute_kernel_hw_startup(dfb_in_id, dfb_out_id);

    ckl::copy<
        ckl::input(dfb_in_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
        ckl::output(
            dfb_out_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>(
        ckl::IterationShape::tiles(per_core_tile_cnt));
}
