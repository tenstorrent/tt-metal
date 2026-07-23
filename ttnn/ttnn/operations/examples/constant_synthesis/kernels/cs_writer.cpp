// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// constant_synthesis writer (BRISC / NoC1).
//
// Fans the constant out to every output page in DRAM. The NoC write pattern is
// equivalent for both variants — one async write of page_bytes per output page,
// `block` writes in flight per barrier — so the writer never contributes to the
// measured delta; the strategy lives entirely in where the source bytes come
// from (see cs_reader.cpp). Runs on NoC1 so its writes overlap the baseline
// reader's NoC0 reads.
//
//   synthesize == 0  (stream_from_dram): drain each streamed block from cb_data
//       and write it out — `block` writes in flight per barrier.
//
//   synthesize == 1  (synthesize): the reader supplied ONE resident template
//       page; replicate it to every output page (same `block`-per-barrier write
//       pattern) without re-reading it.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_data = 0;

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t synthesize = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr uint32_t block = get_compile_time_arg_val(3);  // writes in flight per barrier
    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, dst_addr, page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        if constexpr (synthesize) {
            // One resident template page, replicated to every output page. No re-read.
            // Same `block` writes-in-flight-per-barrier pattern as the baseline.
            cb_wait_front(cb_data, 1);
            const uint32_t l1 = get_read_ptr(cb_data);
            uint32_t p = 0;
            while (p < num_rows) {
                const uint32_t b = (num_rows - p) < block ? (num_rows - p) : block;
                for (uint32_t i = 0; i < b; ++i) {
                    noc_async_write(l1, out_acc.get_noc_addr(start_row + p + i), page_bytes);
                }
                noc_async_write_barrier();  // ONE barrier for `b` writes -> up to `block` in flight
                p += b;
            }
            cb_pop_front(cb_data, 1);
        } else {
            // Drain each streamed block and write it out — identical NoC write pattern.
            uint32_t p = 0;
            while (p < num_rows) {
                const uint32_t b = (num_rows - p) < block ? (num_rows - p) : block;
                cb_wait_front(cb_data, b);
                const uint32_t l1 = get_read_ptr(cb_data);
                for (uint32_t i = 0; i < b; ++i) {
                    noc_async_write(l1 + i * page_bytes, out_acc.get_noc_addr(start_row + p + i), page_bytes);
                }
                noc_async_write_barrier();  // ONE barrier for `b` writes -> up to `block` in flight
                cb_pop_front(cb_data, b);
                p += b;
            }
        }
    }
}
