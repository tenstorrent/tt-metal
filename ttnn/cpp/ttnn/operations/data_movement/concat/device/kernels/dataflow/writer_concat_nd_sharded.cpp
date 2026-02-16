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

    // TODO - TO implement
}
