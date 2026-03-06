// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
// Runs on RISCV_1 (NCRISC), writes RM sticks from L1 to DRAM via NOC1
//
// Per tile-row: waits for Wt pages in cb_out (from untilize),
// extracts 32 RM sticks, writes each to DRAM output via TensorAccessor.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr auto output_accessor_args = TensorAccessorArgs<4>();

    // Runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_accessor_args, dst_addr, output_stick_size);

    uint32_t stick_id = start_stick_id;

    for (uint32_t row = 0; row < N; row++) {
        // Wait for Wt pages from untilize
        cb_wait_front(cb_id_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        // Write 32 RM sticks to DRAM
        for (uint32_t s = 0; s < tile_height; s++) {
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
            noc_async_write(l1_read_addr, noc_addr, output_stick_size);
            l1_read_addr += output_stick_size;
            stick_id++;
        }
        noc_async_write_barrier();

        cb_pop_front(cb_id_out, Wt);
    }
}
