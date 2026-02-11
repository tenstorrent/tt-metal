// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Reader kernel for ND sharded concat.
 *
 * Each core holds a shard of every input and a shard of the output (same grid for all).
 * This kernel reads from a subset of input CBs in concat order and writes into the output CB.
 * The host splits work between reader and writer: reader handles inputs [start_input_id, end_input_id)
 * so that two RISC-V processors can run in parallel (reader on one, writer on the other).
 *
 * Compile-time args: [output_cb_id, page_size, num_input_tensors]
 * Runtime args: [start_input_id, end_input_id, (num_pages, write_offset_pages) for each input in [start, end)]
 *
 * Input CBs 0..num_input_tensors-1 are backed by the input buffers (each core sees its shard).
 * Output data is written to output_cb at byte offset (write_offset_pages * page_size).
 */

#include <stdint.h>

void kernel_main() {
    // --- Compile-time: output CB, page size, total number of input tensors ---
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t num_input_tensors = get_compile_time_arg_val(2);

    // --- Runtime: which input range this kernel instance processes ---
    const uint32_t start_input_id = get_arg_val<uint32_t>(0);
    const uint32_t end_input_id = get_arg_val<uint32_t>(1);

    const uint32_t base_l1_write_addr = get_write_ptr(output_cb);
    uint32_t arg_idx = 2;

    for (uint32_t input_id = start_input_id; input_id < end_input_id; input_id++) {
        const uint32_t num_pages = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t write_offset_pages = get_arg_val<uint32_t>(arg_idx++);

        if (num_pages == 0) {
            continue;
        }

        // Source: this core's shard of input_id (CB is backed by the input buffer).
        uint32_t l1_read_addr = get_read_ptr(input_id);
        const uint64_t noc_addr_src = get_noc_addr(l1_read_addr);

        // Destination: output CB at the offset for this input in concat order.
        const uint32_t l1_write_addr = base_l1_write_addr + (write_offset_pages * page_size);

        // Copy this input's shard (num_pages pages) into the output region.
        noc_async_read_one_packet_set_state(noc_addr_src, page_size);
        for (uint32_t page = 0; page < num_pages; page++) {
            noc_async_read_one_packet_with_state<true>(l1_read_addr, l1_write_addr + page * page_size);
            l1_read_addr += page_size;
        }
        noc_async_read_barrier();
    }
}
