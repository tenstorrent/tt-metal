// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// pure_copy_arms mover: ONE data-movement kernel does DRAM/L1 -> local L1 -> DRAM/L1 for the bytes
// it owns. No compute kernel, no CB handshake (the CB is only an L1 scratch allocation), no row
// compaction: the bit-identical typecast arms (u32<->i32) are a byte copy. Raw NoC calls bypass no
// helper (there is no data-movement helper for an un-transformed page/chunk copy).
// Used single (one RISC per core) and dual (both RISCs, each its own slice and scratch CB).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "copy_units.hpp"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);  // aligned page = accessor stride
    constexpr uint32_t buf_cb = get_compile_time_arg_val(1);
    constexpr uint32_t buf_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t wnoc_sel = get_compile_time_arg_val(3);  // 2: write on this kernel's NoC; 0/1: that NoC
    constexpr auto src_args = TensorAccessorArgs<4>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t page0 = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t off = get_arg_val<uint32_t>(4);
    const uint32_t len = get_arg_val<uint32_t>(5);
    if (count == 0) {
        return;
    }
    const auto src = TensorAccessor(src_args, src_addr, page_bytes);
    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    const uint32_t buf = get_write_ptr(buf_cb);
    const uint8_t wnoc = wnoc_sel == 2 ? noc_index : static_cast<uint8_t>(wnoc_sel);

    bool first = true;
    for_each_batch(page0, count, off, len, buf_bytes, [&](uint32_t p, uint32_t n, uint32_t o, uint32_t b) {
        if (!first) {
            noc_async_writes_flushed(wnoc);  // previous batch's data has left the scratch
        }
        first = false;
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(src.get_noc_addr(p + i, o), buf + i * b, b);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write(buf + i * b, dst.get_noc_addr(p + i, o, wnoc), b, wnoc);
        }
    });
    noc_async_write_barrier(wnoc);
}
