// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode down reader (RISCV_1, NOC_0). One output column per core: in0 = the k routed experts' h rows
// (k * kts 1x32 tiles, already scaled by the routing weights), in1 = each routed expert's kts weight tiles of this
// column. The weights are DRAM width-sharded over the banks: bank b holds columns [b * cpb, (b + 1) * cpb) of every
// expert as [expert][column][k] tiles, so the core reads k contiguous chunks from its column's bank. One custom_mm
// over K = k * kts sums the experts.
// The writer reads the second half of the experts on NOC_1 into the same CB and signals through cb_sync.
// compile-time args: [cb_in0, cb_w, cb_route, kts, k, cols, tile_bytes, route_page_bytes, banks, cb_sync]
//                    + TensorAccessorArgs(route, h)
// runtime args: [w_addr, route_addr, h_addr, col]
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline uint32_t bf16_to_uint(uint16_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = static_cast<uint32_t>(bits) << 16;
    return static_cast<uint32_t>(v.f);
}

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t route_addr = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_arg_val<uint32_t>(2);
    const uint32_t col = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_route = get_compile_time_arg_val(2);
    constexpr uint32_t kts = get_compile_time_arg_val(3);
    constexpr uint32_t k = get_compile_time_arg_val(4);
    constexpr uint32_t cols = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t route_page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t banks = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(9);
    constexpr auto r_args = TensorAccessorArgs<10>();
    constexpr auto h_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr uint32_t cpb = cols / banks;
    static_assert(cpb * banks == cols, "columns must split evenly over the banks");
    constexpr uint32_t tiny_bytes = 64;
    const auto s_r = TensorAccessor(r_args, route_addr, route_page_bytes);
    const auto s_h = TensorAccessor(h_args, h_addr, tiny_bytes);

    cb_reserve_back(cb_route, 1);
    const uint32_t route_l1 = get_write_ptr(cb_route);
    noc_async_read(s_r.get_noc_addr(0), route_l1, route_page_bytes);
    cb_reserve_back(cb_in0, k * kts);
    uint32_t l1 = get_write_ptr(cb_in0);
    for (uint32_t t = 0; t < k * kts; ++t) {
        noc_async_read(s_h.get_noc_addr(t), l1, tiny_bytes);
        l1 += tiny_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_in0, k * kts);

    volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(route_l1);
    const uint32_t bank = col / cpb;
    const uint32_t chunk_bytes = kts * tile_bytes;
    cb_reserve_back(cb_w, k * kts);
    l1 = get_write_ptr(cb_w);
    for (uint32_t s = 0; s < k / 2; ++s) {
        const uint32_t offset = (bf16_to_uint(row[k + s]) * cpb + col % cpb) * chunk_bytes;
#ifndef SKIP_WEIGHTS
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, w_addr + offset), l1, chunk_bytes);
#endif
        l1 += chunk_bytes;
    }
    noc_async_read_barrier();
    cb_wait_front(cb_sync, 1);  // the writer's half has landed
    cb_pop_front(cb_sync, 1);
    cb_push_back(cb_w, k * kts);
}
