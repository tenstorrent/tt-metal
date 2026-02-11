// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Writer kernel for ND sharded concat.
 *
 * Same algorithm as reader_concat_nd_sharded: reads from a subset of input CBs in concat order
 * and writes into the output CB. The host assigns the reader to inputs [0, mid) and the writer
 * to inputs [mid, num_input_tensors), so both RISC-V processors run in parallel and the
 * output CB (backed by the output buffer) is filled with the concatenated shard.
 *
 * Compile-time args: [output_cb_id, page_size, num_input_tensors]
 * Runtime args: [start_input_id, end_input_id, (num_pages, write_offset_pages) for each input in [start, end)]
 *
 * No separate "read from CB and write to buffer" step: the output CB is the output buffer
 * (set_globally_allocated_address), so writing to the CB is writing to the final output.
 */

#include <stdint.h>

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t num_input_tensors = get_compile_time_arg_val(2);

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

        uint32_t l1_read_addr = get_read_ptr(input_id);
        const uint64_t noc_addr_src = get_noc_addr(l1_read_addr);

        const uint32_t l1_write_addr = base_l1_write_addr + (write_offset_pages * page_size);

        noc_async_read_one_packet_set_state(noc_addr_src, page_size);
        for (uint32_t page = 0; page < num_pages; page++) {
            noc_async_read_one_packet_with_state<true>(l1_read_addr, l1_write_addr + page * page_size);
            l1_read_addr += page_size;
        }
        noc_async_read_barrier();
    }
}
