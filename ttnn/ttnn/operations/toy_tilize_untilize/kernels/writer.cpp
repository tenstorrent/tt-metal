// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_out = 16;
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t total_num_rows = get_arg_val<uint32_t>(1);

    const auto accessor = TensorAccessor(dst_args, dst_addr);
    dataflow_kernel_lib::write_sticks_after_untilize<cb_rm_out>(accessor, total_num_rows, row_bytes);
}
