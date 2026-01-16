// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Stub writer kernel for layernorm_fused_rm
// Writes output RM sticks

void kernel_main() {
    // Compile-time args
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);

    // TensorAccessor compile-time args
    constexpr auto dst_args = TensorAccessorArgs<4>();

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Create TensorAccessor
    const auto d = TensorAccessor(dst_args, dst_addr, stick_size);

    // STUB: Write output rows
    // Each tile row is 32 sticks tall, we write them all at once
    // CB page = one stick, so we pop 32 pages per tile row
    uint32_t stick_id = start_stick_id;
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        cb_wait_front(cb_out_rm, 32);
        uint32_t l1_addr = get_read_ptr(cb_out_rm);

        // Write 32 sticks (one tile row height)
        for (uint32_t stick_in_tile_row = 0; stick_in_tile_row < 32; stick_in_tile_row++) {
            noc_async_write(l1_addr, d.get_noc_addr(stick_id), stick_size);
            stick_id++;
            l1_addr += stick_size;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out_rm, 32);
    }
}
