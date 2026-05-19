// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    compute_kernel_hw_startup(cb_input, cb_output);
    compute_kernel_lib::copy<cb_input, cb_output>(num_tiles);
}
