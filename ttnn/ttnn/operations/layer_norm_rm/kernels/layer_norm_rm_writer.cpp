// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Writer Kernel (NCRISC / NOC1)
//
// Writes untilized output sticks from cb_rm_out to DRAM.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    // ── Compile-time args ──
    constexpr uint32_t total_num_rows = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);

    constexpr auto output_ta_args = TensorAccessorArgs<2>();

    // ── Runtime args ──
    const uint32_t output_addr = get_arg_val<uint32_t>(0);

    // ── CB index ──
    constexpr uint32_t cb_rm_out = 16;

    // ── Write output sticks ──
    const auto output_accessor = TensorAccessor(output_ta_args, output_addr);
    dataflow_kernel_lib::write_sticks_after_untilize<cb_rm_out>(output_accessor, total_num_rows, row_bytes);
}
