// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

// y = a + b, streaming.
// Compile-time: [num_tiles]

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_out = 16;

    using namespace compute_kernel_lib::eltwise;

    binary_op_init_common(cb_a, cb_b, cb_out);

    add(cb_a, cb_b, cb_out, BinaryInputBlockShape::of(1, num_tiles));
}
