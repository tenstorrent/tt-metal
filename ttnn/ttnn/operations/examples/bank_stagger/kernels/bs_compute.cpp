// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_stagger example compute (unpack/math/pack TRISCs).
//
// Tilizes each 32-row x chunk_wt-tile block from cb_in into cb_out. Byte-identical
// for every variant: the issue-order rotation lives entirely in the reader and
// writer, and the block in cb_in is the same bytes regardless of read order.

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t chunk_wt = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    compute_kernel_lib::tilize<chunk_wt, cb_in, cb_out>(num_blocks * kernel_iters);
}
