// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    uint32_t responsibility = get_arg_val<uint32_t>(0);

    constexpr uint32_t src0_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t src1_cb_index = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(src0_cb_index, src1_cb_index);
    compute_kernel_lib::tilize<src0_cb_index, src1_cb_index>(1, responsibility);
}
