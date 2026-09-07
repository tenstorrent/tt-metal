// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rw_overlap DUPLEX ROOFLINE PROBE -- reader half (NCRISC / NoC0).
//
// NOT a correct program.  This kernel exists to answer ONE question that decides
// the whole `rw_overlap` idea:
//
//     when a DRAM read stream and a DRAM write stream run with NO dependency
//     between them at all, do they add up (full duplex) or do they share one
//     bandwidth pool (a DRAM roofline)?
//
// So there is deliberately NO circular-buffer handshake with the writer: the
// reader lands every page in a fixed L1 scratch ring and never pushes.  Nothing
// can serialize the two streams except the hardware.  The read ADDRESS PATTERN
// and TRANSACTION SHAPE are the shipped rms_norm_ttnn reader's: `BLOCK` whole
// interleaved tile pages issued back to back, then ONE noc_async_read_barrier().
//
// RAW API, no dataflow helper: the helper contract (reserve/push at tile
// granularity) is exactly the synchronization this probe must NOT have.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_scratch = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t block = get_compile_time_arg_val(1);
    constexpr uint32_t ring_pages = get_compile_time_arg_val(2);
    constexpr uint32_t enabled = get_compile_time_arg_val(3);
    constexpr auto in_args = TensorAccessorArgs<4>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    if constexpr (enabled == 0) {
        return;
    }

    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);
    const uint32_t l1_base = get_write_ptr(cb_scratch);

    uint32_t p = 0;
    uint32_t slot = 0;
    while (p < num_pages) {
        const uint32_t b = (num_pages - p) < block ? (num_pages - p) : block;
        for (uint32_t i = 0; i < b; ++i) {
            noc_async_read(in_acc.get_noc_addr(start_page + p + i), l1_base + slot * page_bytes, page_bytes);
            slot += 1;
            if (slot == ring_pages) {
                slot = 0;
            }
        }
        noc_async_read_barrier();
        p += b;
    }
}
