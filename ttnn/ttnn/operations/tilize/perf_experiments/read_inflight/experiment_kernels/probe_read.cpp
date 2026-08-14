// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_inflight DIAGNOSTIC probe — the READ STAGE ALONE, nothing downstream.
//
// No CB handshake, no compute, no writer: this program is one reader kernel per
// core that pulls `num_pages` DRAM pages of `page_bytes` into a fixed L1 window,
// barriering every `pages_per_barrier` pages. It answers the one question the
// bake-off result hinges on: is the crossover's read LATENCY-bound (so more
// bytes in flight per barrier buys bandwidth) or TRANSACTION-bound (so it does
// not, and only a bigger transfer would)?
//
//   * sweep `pages_per_barrier` at fixed page size -> the in-flight-window axis
//   * sweep `page_bytes` at fixed total bytes     -> the transaction-count axis
//
// Correctness of the probe itself is checked by the caller (it reads the same
// pages the real bench reads, into an L1 window it then never publishes), so
// this file is a measurement instrument only.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_scratch = 0;

    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_barrier = get_compile_time_arg_val(1);
    constexpr uint32_t window_bytes = get_compile_time_arg_val(2);  // L1 window we cycle through
    constexpr auto src_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    if (num_pages == 0) {
        return;
    }
    const auto acc = TensorAccessor(src_args, src_addr);
    const uint32_t base = get_write_ptr(cb_scratch);

    uint32_t l1 = base;
    uint32_t in_group = 0;
    for (uint32_t p = 0; p < num_pages; ++p) {
        noc_async_read(acc.get_noc_addr(start_page + p), l1, page_bytes);
        l1 += page_bytes;
        if (l1 - base + page_bytes > window_bytes) {
            l1 = base;
        }
        if (++in_group == pages_per_barrier) {
            noc_async_read_barrier();
            in_group = 0;
        }
    }
    noc_async_read_barrier();
}
