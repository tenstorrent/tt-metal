// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "experimental/circular_buffer.h"

// Generic block-sharded concat kernel.
// Processes a list of transfer descriptors, each describing a cross-core or local
// NOC read from a source shard into the output shard.
//
// Each transfer copies a rectangular region: num_rows rows of copy_size bytes,
// with independent source and destination strides.
//
// Source addresses are specified as (cb_id, offset) rather than absolute L1
// addresses so that program-cache reuse works correctly after
// UpdateDynamicCircularBufferAddress updates the backing buffer.

void kernel_main() {
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_transfers = get_compile_time_arg_val(1);

    experimental::CircularBuffer output_cb(output_cb_id);
    uint32_t dst_base = output_cb.get_write_ptr();
    uint32_t arg_idx = 0;

    for (uint32_t t = 0; t < num_transfers; t++) {
        uint32_t src_noc_x = get_arg_val<uint32_t>(arg_idx++);
        uint32_t src_noc_y = get_arg_val<uint32_t>(arg_idx++);
        uint32_t src_cb_id = get_arg_val<uint32_t>(arg_idx++);
        uint32_t src_l1_offset = get_arg_val<uint32_t>(arg_idx++);
        uint32_t src_stride = get_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_offset = get_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_stride = get_arg_val<uint32_t>(arg_idx++);
        uint32_t copy_size = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_rows = get_arg_val<uint32_t>(arg_idx++);

        experimental::CircularBuffer src_cb(src_cb_id);
        uint32_t src_l1_addr = src_cb.get_read_ptr() + src_l1_offset;
        uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_l1_addr);
        uint32_t dst_addr = dst_base + dst_offset;

        for (uint32_t row = 0; row < num_rows; row++) {
            noc_async_read(src_noc_addr, dst_addr, copy_size);
            src_noc_addr += src_stride;
            dst_addr += dst_stride;
        }
    }

    noc_async_read_barrier();
}
