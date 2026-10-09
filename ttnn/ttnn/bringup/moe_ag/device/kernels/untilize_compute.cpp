// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Active-row untilize, compute: the header's block count of W-tile blocks, c_0 (bfp8 tiles) -> c_16 (bf16 rows).
// CT: 0 W
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    cb_wait_front(tt::CBIndex::c_2, 1);
    const uint32_t n = read_tile_value(tt::CBIndex::c_2, 0, 0);
    cb_pop_front(tt::CBIndex::c_2, 1);
    if (n) {
        compute_kernel_lib::untilize<W, tt::CBIndex::c_0, tt::CBIndex::c_16>(n);
    }
}
