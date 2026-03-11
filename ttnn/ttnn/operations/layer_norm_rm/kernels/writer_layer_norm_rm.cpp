// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
// Writes untilized RM sticks from cb_out to DRAM via TensorAccessor

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);  // W * 2 bytes
    constexpr uint32_t Wt = get_compile_time_arg_val(1);          // tiles per row
    constexpr uint32_t Ht = get_compile_time_arg_val(2);          // total tile-rows

    // TensorAccessor args for output start at index 3
    constexpr auto output_accessor_args = TensorAccessorArgs<3>();

    // CB index
    constexpr uint32_t cb_out = 16;

    // ========== Runtime args ==========
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    // ========== Setup TensorAccessor for output ==========
    const auto output_accessor = TensorAccessor(output_accessor_args, dst_addr, stick_size);

    // ========== Main loop: write output RM sticks per tile-row ==========
    uint32_t stick_id = 0;  // Global stick index for TensorAccessor

    for (uint32_t ht = 0; ht < Ht; ht++) {
        // Wait for Wt tiles of untilized data from compute
        cb_wait_front(cb_out, Wt);

        // Get L1 base address for reading
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        // Write 32 RM sticks per tile-row
        for (uint32_t s = 0; s < 32; s++) {
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        // Release the tiles
        cb_pop_front(cb_out, Wt);
    }
}
