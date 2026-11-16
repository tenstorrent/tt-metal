// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode gate|up weight reader (RISCV_1, NOC_0). One work unit per core: (routing slot s, column pair j).
// Reads the slot's expert id from the packed routing row (RouterTopk.op_packed: [w_0..w_{k-1} | id_0..id_{k-1}] bf16,
// user row 0) and that expert's gate column j and up column j (2 x kt tiles). The weights are DRAM width-sharded
// over the banks: bank b holds K slice b (ks = kt / banks tiles) of every (expert, pair) as [gate ks | up ks]
// contiguous, so a core reads one chunk per bank, starting at a bank rotated by its slot and pair (the compute
// kernel gets the same rotation) so the cores spread over the banks; every bank serves the same bytes for any
// routing.
// compile-time args: [cb_w, cb_route, kt, pairs, k, tile_bytes, route_page_bytes, banks, (unused)]
//                    + TensorAccessorArgs(route)
// runtime args: [w_addr, route_addr, slot, pair]
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
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t pair = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_route = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t pairs = get_compile_time_arg_val(3);
    constexpr uint32_t k = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t route_page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t banks = get_compile_time_arg_val(7);
    constexpr auto r_args = TensorAccessorArgs<9>();
    constexpr uint32_t ks = kt / banks;
    static_assert(ks * banks == kt, "K tiles must split evenly over the banks");
    const auto s_r = TensorAccessor(r_args, route_addr, route_page_bytes);

    cb_reserve_back(cb_route, 1);
    const uint32_t route_l1 = get_write_ptr(cb_route);
    noc_async_read(s_r.get_noc_addr(0), route_l1, route_page_bytes);
    noc_async_read_barrier();
    const uint32_t expert = bf16_to_uint(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(route_l1)[k + slot]);

    // per bank chunk, in issue order: [gate K slice | up K slice] of this (expert, pair), one transaction id each;
    // each chunk is pushed as soon as its reads land so compute overlaps the stream
    const uint32_t chunk_bytes = ks * tile_bytes;
    const uint32_t offset = (expert * pairs + pair) * 2 * chunk_bytes;
    cb_reserve_back(cb_w, 2 * kt);
    const uint32_t l1 = get_write_ptr(cb_w);
    const uint32_t first_bank = (slot * pairs + pair) % banks;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    for (uint32_t i = 0; i < banks; ++i) {
        const uint32_t b = (first_bank + i) % banks;
        const uint64_t src = get_noc_addr_from_bank_id<true>(b, w_addr + offset);
        noc_async_read_set_trid(i + 1);
        noc_async_read(src, l1 + 2 * i * chunk_bytes, 2 * chunk_bytes);
    }
    for (uint32_t i = 0; i < banks; ++i) {
        noc_async_read_barrier_with_trid(i + 1);
        cb_push_back(cb_w, 2 * ks);
    }
    noc_async_read_set_trid(0);
}
