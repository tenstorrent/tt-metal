// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scatter writer – ROW_MAJOR path (BRISC).
// Dual-RISC: loads input sticks into cb_input, writes output sticks to DRAM.
//
// Divergence from the prototype template: each batch ends with async_writes_flushed() and one
// async_write_barrier() closes the kernel, instead of a barrier per batch.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_stick = get_arg_val<uint32_t>(2);
    const uint32_t num_sticks = get_arg_val<uint32_t>(3);

    // Compile-time args
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    // Actual page sizes in bytes — passed explicitly because get_tile_size()
    // returns L1 words (not bytes) for RM CBs.
    constexpr uint32_t input_page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t output_page_bytes = get_compile_time_arg_val(3);
    constexpr auto input_ta_args = TensorAccessorArgs<4>();
    constexpr auto output_ta_args = TensorAccessorArgs<input_ta_args.next_compile_time_args_offset()>();

    constexpr uint32_t one_page = 1;

    const auto input_accessor = TensorAccessor(input_ta_args, input_addr, input_page_bytes);
    const auto output_accessor = TensorAccessor(output_ta_args, output_addr, output_page_bytes);

    Noc noc;
    CircularBuffer in_cb(cb_input);
    CircularBuffer out_cb(cb_output);

    for (uint32_t s = 0; s < num_sticks; s++) {
        const uint32_t stick_id = start_stick + s;

        // Load input stick
        in_cb.reserve_back(one_page);
        noc.async_read(
            input_accessor, in_cb, input_page_bytes, {.page_id = stick_id, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        in_cb.push_back(one_page);

        // Write output stick
        out_cb.wait_front(one_page);
        noc.async_write(
            out_cb, output_accessor, output_page_bytes, {.offset_bytes = 0}, {.page_id = stick_id, .offset_bytes = 0});
        noc.async_writes_flushed();
        out_cb.pop_front(one_page);
    }
    noc.async_write_barrier();
}
