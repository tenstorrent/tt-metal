// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(cb_in, cb_in, cb_out);

    using namespace compute_kernel_lib;
    sum_of_squares<input(cb_in), row_output(cb_out)>(IterationShape::grid(Ht, Wt).block_size(block_size));
}
