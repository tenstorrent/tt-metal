// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode gate|up, bank-local form (RISCV_1, NOC_0). Two roles (runtime arg):
//   1 bank reader (bank b, reader r; on the bank's row): streams, for each of the k routed experts, its nr gate|up
//     columns over K slice b (one contiguous chunk of nr * ks tiles in bank b, layout [expert][reader][n][k]); all
//     chunks in flight at once (one transaction id each), each pushed as it lands.
//   2 reducer (routing slot s, pair j): waits until the banks' readers have written their partial (g, u) tiles into
//     its cb_recv (semaphore = banks), then pushes them.
// compile-time args: [cb_w, cb_route, cb_recv, ks, nr, readers, k, tile_bytes, route_page_bytes, sem_id, banks]
//                    + TensorAccessorArgs(route)
// runtime args: [w_addr, route_addr, role, bank, reader]
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
    const uint32_t role = get_arg_val<uint32_t>(2);
    const uint32_t bank = get_arg_val<uint32_t>(3);
    const uint32_t reader = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_route = get_compile_time_arg_val(1);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(2);
    constexpr uint32_t ks = get_compile_time_arg_val(3);
    constexpr uint32_t nr = get_compile_time_arg_val(4);
    constexpr uint32_t readers = get_compile_time_arg_val(5);
    constexpr uint32_t k = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t route_page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t banks = get_compile_time_arg_val(10);
    constexpr auto r_args = TensorAccessorArgs<11>();
    if (role == 1) {
        const auto s_r = TensorAccessor(r_args, route_addr, route_page_bytes);
        cb_reserve_back(cb_route, 1);
        const uint32_t route_l1 = get_write_ptr(cb_route);
        noc_async_read(s_r.get_noc_addr(0), route_l1, route_page_bytes);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(route_l1);
        constexpr uint32_t chunk_tiles = nr * ks;
        constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;
        cb_reserve_back(cb_w, k * chunk_tiles);
        const uint32_t l1 = get_write_ptr(cb_w);
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
        for (uint32_t s = 0; s < k; ++s) {
            const uint32_t offset = (bf16_to_uint(row[k + s]) * readers + reader) * chunk_bytes;
            noc_async_read_set_trid(s + 1);
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, w_addr + offset), l1 + s * chunk_bytes, chunk_bytes);
        }
        for (uint32_t s = 0; s < k; ++s) {
            noc_async_read_barrier_with_trid(s + 1);
            cb_push_back(cb_w, chunk_tiles);
        }
        noc_async_read_set_trid(0);
    } else if (role == 2) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
        cb_reserve_back(cb_recv, 2 * banks);
        noc_semaphore_wait(sem, banks);
        cb_push_back(cb_recv, 2 * banks);
    }
}
