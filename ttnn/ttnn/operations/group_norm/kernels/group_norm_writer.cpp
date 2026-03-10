// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Group Norm - Writer Kernel
// Writes untilized RM sticks from cb_output_rm (CB 17) to DRAM.
// For each tile-row: wait for Ct pages, write 32 sticks, pop.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ========== COMPILE-TIME ARGS ==========
    constexpr uint32_t cb_output_rm = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);    // 32
    constexpr uint32_t num_tile_rows = get_compile_time_arg_val(3);  // N * Ht
    constexpr uint32_t Ct = get_compile_time_arg_val(4);
    constexpr auto output_accessor_args = TensorAccessorArgs<5>();

    // ========== RUNTIME ARGS ==========
    uint32_t arg_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);

    // ========== SETUP TENSOR ACCESSOR ==========
    auto output_accessor = TensorAccessor(output_accessor_args, output_addr, output_stick_size);

    // ========== MAIN LOOP: WRITE RM STICKS ==========
    uint32_t stick_id = 0;

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        cb_wait_front(cb_output_rm, Ct);
        uint32_t l1_addr = get_read_ptr(cb_output_rm);

        for (uint32_t s = 0; s < tile_height; ++s) {
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_addr, noc_addr, output_stick_size);
            l1_addr += output_stick_size;
            stick_id++;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_output_rm, Ct);
    }
}
