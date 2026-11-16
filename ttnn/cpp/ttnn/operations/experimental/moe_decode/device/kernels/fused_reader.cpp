// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode batch-1 fused program, reader (RISCV_1, NOC_0). Every core of the grid runs it; runtime args give the
// core's roles:
//   gate|up unit (slot s, pair j): stream that expert's gate and up column j, one [gate ks | up ks] chunk per DRAM
//     bank (bank-sharded layout, see gate_up_reader.cpp), pushed per chunk as it lands;
//   down column c: prefetch the k routed experts' kts down tiles of column c from the column's bank (they do not
//     depend on h, so they stream while the gate|up units run), then wait until all k * pairs gate|up units have
//     signalled (semaphore) and pull their h tiles into cb_h.
// compile-time args: [cb_w, cb_dw, cb_h, cb_route, kt, pairs, k, kts, cols, tile_bytes, route_page_bytes, banks,
//                     h_sem_id, cb_hout] + TensorAccessorArgs(route)
// runtime args: [w_gu_addr, w_dn_addr, route_addr, is_gu, slot, pair, has_down, col,
//                gate|up core (x, y) NOC coordinates * (k * pairs), in h tile order]
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
    const uint32_t w_gu_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_dn_addr = get_arg_val<uint32_t>(1);
    const uint32_t route_addr = get_arg_val<uint32_t>(2);
    const uint32_t is_gu = get_arg_val<uint32_t>(3);
    const uint32_t slot = get_arg_val<uint32_t>(4);
    const uint32_t pair = get_arg_val<uint32_t>(5);
    const uint32_t has_down = get_arg_val<uint32_t>(6);
    const uint32_t col = get_arg_val<uint32_t>(7);
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dw = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h = get_compile_time_arg_val(2);
    constexpr uint32_t cb_route = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    constexpr uint32_t pairs = get_compile_time_arg_val(5);
    constexpr uint32_t k = get_compile_time_arg_val(6);
    constexpr uint32_t kts = get_compile_time_arg_val(7);
    constexpr uint32_t cols = get_compile_time_arg_val(8);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t route_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t banks = get_compile_time_arg_val(11);
    constexpr uint32_t h_sem_id = get_compile_time_arg_val(12);
    constexpr uint32_t cb_hout = get_compile_time_arg_val(13);
    constexpr auto r_args = TensorAccessorArgs<14>();
    constexpr uint32_t ks = kt / banks;
    constexpr uint32_t cpb = cols / banks;
    static_assert(ks * banks == kt && cpb * banks == cols, "K tiles and columns must split over the banks");
    if (!is_gu && !has_down) {
        return;
    }
    const auto s_r = TensorAccessor(r_args, route_addr, route_page_bytes);
    cb_reserve_back(cb_route, 1);
    const uint32_t route_l1 = get_write_ptr(cb_route);
    noc_async_read(s_r.get_noc_addr(0), route_l1, route_page_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(route_l1);

    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    uint32_t first_bank = 0;
    if (is_gu) {
        const uint32_t chunk_bytes = ks * tile_bytes;
        const uint32_t offset = (bf16_to_uint(row[k + slot]) * pairs + pair) * 2 * chunk_bytes;
        cb_reserve_back(cb_w, 2 * kt);
        const uint32_t l1 = get_write_ptr(cb_w);
        first_bank = (slot * pairs + pair) % banks;
        for (uint32_t i = 0; i < banks; ++i) {
            const uint32_t b = (first_bank + i) % banks;
            noc_async_read_set_trid(i + 1);
            noc_async_read(
                get_noc_addr_from_bank_id<true>(b, w_gu_addr + offset), l1 + 2 * i * chunk_bytes, 2 * chunk_bytes);
        }
    }
    if (has_down) {
        // down weights of the column: k contiguous kts-tile chunks from the column's bank (trid banks + 1)
        const uint32_t bank = col / cpb;
        const uint32_t chunk_bytes = kts * tile_bytes;
        cb_reserve_back(cb_dw, k * kts);
        uint32_t l1 = get_write_ptr(cb_dw);
        noc_async_read_set_trid(banks + 1);
        for (uint32_t s = 0; s < k; ++s) {
            const uint32_t offset = (bf16_to_uint(row[k + s]) * cpb + col % cpb) * chunk_bytes;
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, w_dn_addr + offset), l1, chunk_bytes);
            l1 += chunk_bytes;
        }
    }
    if (is_gu) {
        for (uint32_t i = 0; i < banks; ++i) {
            noc_async_read_barrier_with_trid(i + 1);
            cb_push_back(cb_w, 2 * ks);
        }
    }
    if (has_down) {
        noc_async_read_barrier_with_trid(banks + 1);
        cb_push_back(cb_dw, k * kts);
        // h: every gate|up unit increments this semaphore once its tile sits in its cb_hout; pull the k * pairs tiles
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(h_sem_id));
        cb_reserve_back(cb_h, k * kts);
#ifndef FUSED_NO_SYNC
        noc_semaphore_wait(sem, k * pairs);
#endif
        uint32_t h_l1 = get_write_ptr(cb_h);
        const uint32_t hout = get_read_ptr(cb_hout);  // the producers' cb_hout: same address on every core
        for (uint32_t u = 0; u < k * pairs; ++u) {
            noc_async_read(
                get_noc_addr(get_arg_val<uint32_t>(8 + 2 * u), get_arg_val<uint32_t>(9 + 2 * u), hout), h_l1, 64);
            h_l1 += 64;
        }
        noc_async_read_barrier();
        cb_push_back(cb_h, k * kts);
    }
    noc_async_read_set_trid(0);
}
