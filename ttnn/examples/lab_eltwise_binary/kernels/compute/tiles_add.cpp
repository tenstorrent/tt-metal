// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

void kernel_main() {
    // Note: The argument index to get_compile_time_arg_val() must be a compile time constant.
    const uint32_t n_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in0 = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_in1 = static_cast<uint32_t>(tt::CBIndex::c_1);
    constexpr uint32_t cb_out0 = static_cast<uint32_t>(tt::CBIndex::c_16);

    // Boot the engine once at MAIN entry (D5: first statement of MAIN).
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);

    // Per-tile FPU add: cb_in0 + cb_in1 -> cb_out0.
    compute_kernel_lib::binary_add<cb_in0, cb_in1, cb_out0>(n_tiles);
}
