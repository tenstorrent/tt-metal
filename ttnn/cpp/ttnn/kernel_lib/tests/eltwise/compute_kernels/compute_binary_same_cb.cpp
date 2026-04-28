// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

// y = x + x via add(cb, cb, ...). Drives the same-CB wait/pop dedup path.

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;

    using namespace compute_kernel_lib::eltwise;

    binary_op_init_common(cb_in, cb_in, cb_out);

    add(cb_in, cb_in, cb_out, BinaryInputBlockShape::of(1, num_tiles));
}
