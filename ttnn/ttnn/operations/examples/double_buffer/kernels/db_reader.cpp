// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// double_buffer reader (NCRISC / NoC0).
//
// Reads this core's contiguous range of interleaved DRAM pages into the
// reader->compute CB (cb_in), then hands them to compute.
//
// Two things about this kernel are the whole point of the example:
//
//   1. READS PER BARRIER (`block`). We issue `block` async reads back-to-back
//      and then wait on ONE barrier for all of them. block=1 is the trap:
//      read-one / wait / read-one / wait is LATENCY-bound — every tile pays a
//      full DRAM round trip before the next read even starts, so the NoC sits
//      idle most of the time. With block>1, up to `block` reads are in flight
//      at once and the transfers pipeline, so we approach DRAM BANDWIDTH
//      instead of being pinned to per-read latency.
//
//   2. DEPTH (set by the program descriptor via cb_in's total_size). cb_in holds
//      `depth * block` tiles. depth=1 (single-buffered) means the reader must
//      wait for the whole previous block to be drained before refilling; depth=2
//      (double-buffered) lets it prefetch the next block while compute/writer
//      drain the current one.
//
// The kernel source is byte-identical for both single- and double-buffered runs
// and for every `block`; those are compile-time args / CB sizes, not code
// changes.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    constexpr auto in_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        uint32_t p = 0;
        while (p < num_pages) {
            const uint32_t b = (num_pages - p) < block ? (num_pages - p) : block;
            cb_reserve_back(cb_in, b);
            const uint32_t l1_write_addr = get_write_ptr(cb_in);
            for (uint32_t i = 0; i < b; ++i) {
                noc_async_read(in_acc.get_noc_addr(start_page + p + i), l1_write_addr + i * page_bytes, page_bytes);
            }
            noc_async_read_barrier();  // ONE barrier for `b` reads -> up to `block` reads in flight
            cb_push_back(cb_in, b);
            p += b;
        }
    }
}
