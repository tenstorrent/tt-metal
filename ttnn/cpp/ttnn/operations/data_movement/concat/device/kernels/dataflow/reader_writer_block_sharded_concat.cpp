// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

// Dual-RISC block-sharded concat kernel.
// Used as both reader (BRISC/NOC0) and writer (NCRISC/NOC1) with the host
// splitting transfer descriptors between the two to double NOC bandwidth.
//
// num_transfers is a runtime arg so reader and writer can have different counts.
// Each transfer copies a rectangular region: num_rows rows of copy_size bytes,
// with independent source and destination strides.
//
// Source addresses are specified as (cb_id, offset) rather than absolute L1
// addresses so that program-cache reuse works correctly after
// UpdateDynamicCircularBufferAddress updates the backing buffer.

void kernel_main() {
    constexpr uint32_t output_dfb_id = get_compile_time_arg_val(0);

    Noc noc;
    DataflowBuffer output_dfb(output_dfb_id);
    uint32_t dst_base = output_dfb.get_write_ptr();

    uint32_t arg_idx = 0;
    uint32_t num_transfers = get_arg_val<uint32_t>(arg_idx++);

    for (uint32_t t = 0; t < num_transfers; t++) {
        uint32_t src_noc_x = get_arg_val<uint32_t>(arg_idx++);
        uint32_t src_noc_y = get_arg_val<uint32_t>(arg_idx++);
        uint32_t src_dfb_id = get_arg_val<uint32_t>(arg_idx++);
        uint32_t src_l1_offset = get_arg_val<uint32_t>(arg_idx++);
        uint32_t src_stride = get_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_offset = get_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_stride = get_arg_val<uint32_t>(arg_idx++);
        uint32_t copy_size = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_rows = get_arg_val<uint32_t>(arg_idx++);

        DataflowBuffer src_dfb(src_dfb_id);
        uint32_t src_l1_addr = src_dfb.get_read_ptr() + src_l1_offset;
        uint32_t dst_addr = dst_base + dst_offset;

        UnicastEndpoint src;
        for (uint32_t row = 0; row < num_rows; row++) {
            CoreLocalMem<uint32_t> dst(dst_addr);
            noc.async_read(
                src,
                dst,
                copy_size,
                {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = src_l1_addr},
                {.offset_bytes = 0});
            src_l1_addr += src_stride;
            dst_addr += dst_stride;
        }
    }

    noc.async_read_barrier();
}
