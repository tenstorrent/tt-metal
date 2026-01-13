// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// STUB READER: Reads row-major sticks from DRAM to CB
void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto src_tensor_args = TensorAccessorArgs<1>();  // TensorAccessor args start at index 1

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_0;

    // Setup TensorAccessor (page_size = stick_size for row-major)
    const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);

    // Read sticks from DRAM to CB
    // For stub: just read all sticks sequentially
    uint32_t stick_id = start_stick_id;
    for (uint32_t i = 0; i < num_sticks; i++) {
        cb_reserve_back(cb_id_out, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_out);
        noc_async_read(s.get_noc_addr(stick_id), l1_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_id_out, 1);
        stick_id++;
    }
}
