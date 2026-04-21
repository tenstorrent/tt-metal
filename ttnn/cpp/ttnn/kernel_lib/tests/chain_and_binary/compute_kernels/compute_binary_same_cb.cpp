// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Test: y = x + x via binary_op with icb_a == icb_b.
//
// Exercises the same-CB wait/pop deduplication path in binary_op_helpers.inl.
// When icb_a == icb_b, every cb_wait_front(icb_b, ...) and cb_pop_front(icb_b, ...)
// call is skipped at runtime via the `same_cb` flag. Without the dedup, the CB
// would be waited twice and popped twice per tile, which either deadlocks
// (insufficient tiles on the second wait) or skips every other tile's data.

#include "api/compute/common_globals.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    binary_op_init_common(cb_input, cb_input, cb_output);

    using namespace compute_kernel_lib;
    // Ht = 1, Wt = num_tiles: stream all tiles as a flat row.
    add(cb_input, cb_input, cb_output, BinaryInputBlockShape::of(1, num_tiles));
}
}  // namespace NAMESPACE
