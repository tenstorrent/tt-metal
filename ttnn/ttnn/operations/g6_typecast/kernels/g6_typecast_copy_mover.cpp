// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast R0 `pure_copy`, chunk mode (fewer pages than cores): ONE data-movement kernel per core
// (RISCV_0 on NoC0) copies its byte chunk src -> local L1 scratch -> dst: read, barrier, write, barrier.
// No CB handshake (the CB is only the L1 scratch allocation). The host places each chunk's core on the
// NoC0-optimal worker row of the page's DRAM bank (Perf 1, perf_experiments/pure_copy_arms: one 4 KB
// page 498 ns from the bank row vs 658 ns elsewhere). A single kernel beat a reader/writer pair here
// (F321 499 vs 614 ns, Q2230 634 vs 781 ns): with one batch per core the CB handshake is pure latency.
// Raw NoC calls bypass no helper: kernel_lib has no data-movement helper for an un-transformed
// page/chunk copy.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_typecast_copy_units.hpp"
#include "g6_typecast_zones.hpp"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);  // aligned page = accessor stride
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t buf_bytes = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t page0 = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t off = get_arg_val<uint32_t>(2);
    const uint32_t len = get_arg_val<uint32_t>(3);

    const auto src = TensorAccessor(src_args, src_addr, page_bytes);
    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    const uint32_t buf = get_write_ptr(cb_scratch);

    bool first = true;
    g6_copy::for_each_batch(page0, count, off, len, buf_bytes, [&](uint32_t p, uint32_t n, uint32_t o, uint32_t b) {
        if (!first) {
            G6_ZONE("copy_wait_scratch");
            noc_async_writes_flushed();  // previous batch's data has left the scratch
        }
        first = false;
        {
            G6_ZONE("copy_read");
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_read(src.get_noc_addr(p + i, o), buf + i * b, b);
            }
            noc_async_read_barrier();
        }
        G6_ZONE("copy_write_issue");
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write(buf + i * b, dst.get_noc_addr(p + i, o), b);
        }
    });
    G6_ZONE("copy_write_barrier");
    noc_async_write_barrier();
}
