// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
// Writes untilized RM sticks from c_16 to DRAM via TensorAccessor.

#include "api/dataflow/dataflow_api.h"

// CB indices
constexpr uint32_t cb_output_rm = 16;  // c_16: Untilized RM output

// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr auto output_tensor_args = TensorAccessorArgs<2>();

void kernel_main() {
    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t nblocks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Setup output TensorAccessor
    const auto output_accessor = TensorAccessor(output_tensor_args, dst_addr, stick_size);

    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < nblocks; ++block) {
        // Wait for Wt pages of untilized RM data
        cb_wait_front(cb_output_rm, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_output_rm);

        // Write 32 RM sticks to DRAM
        for (uint32_t i = 0; i < 32; ++i) {
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        // Release the pages
        cb_pop_front(cb_output_rm, Wt);
    }
}
