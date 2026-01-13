// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// STUB WRITER: Writes row-major sticks from CB to DRAM
void kernel_main() {
    // Compile-time args
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(0);
    constexpr auto dst_tensor_args = TensorAccessorArgs<1>();  // TensorAccessor args start at index 1

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in = tt::CBIndex::c_16;

    // Setup TensorAccessor (page_size = stick_size for row-major)
    const auto d = TensorAccessor(dst_tensor_args, dst_addr, output_stick_size);

    // Write sticks from CB to DRAM
    uint32_t stick_id = start_stick_id;
    for (uint32_t i = 0; i < num_sticks; i++) {
        cb_wait_front(cb_id_in, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in);
        noc_async_write(l1_read_addr, d.get_noc_addr(stick_id), output_stick_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id_in, 1);
        stick_id++;
    }
}
