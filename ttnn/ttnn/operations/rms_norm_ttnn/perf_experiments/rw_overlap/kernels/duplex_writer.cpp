// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rw_overlap DUPLEX ROOFLINE PROBE -- writer half (BRISC / NoC1).
//
// Mirror of duplex_reader.cpp: writes `num_pages` whole interleaved tile pages
// out of a fixed L1 scratch ring with NO circular-buffer handshake, so the write
// stream is independent of the read stream by construction.  The bytes written
// are whatever is in L1 (garbage) -- this is a BANDWIDTH probe, never a correct
// program; the correctness-gated candidates live in bench_op.py.
//
// Transaction shape matches the shipped rms_norm_ttnn writer: `BLOCK` whole tile
// pages issued back to back, then ONE noc_async_write_barrier().

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_scratch = 16;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t block = get_compile_time_arg_val(1);
    constexpr uint32_t ring_pages = get_compile_time_arg_val(2);
    constexpr uint32_t enabled = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    if constexpr (enabled == 0) {
        return;
    }

    const auto out_acc = TensorAccessor(out_args, dst_addr, page_bytes);
    const uint32_t l1_base = get_read_ptr(cb_scratch);

    uint32_t p = 0;
    uint32_t slot = 0;
    while (p < num_pages) {
        const uint32_t b = (num_pages - p) < block ? (num_pages - p) : block;
        for (uint32_t i = 0; i < b; ++i) {
            noc_async_write(l1_base + slot * page_bytes, out_acc.get_noc_addr(start_page + p + i), page_bytes);
            slot += 1;
            if (slot == ring_pages) {
                slot = 0;
            }
        }
        noc_async_write_barrier();
        p += b;
    }
}
