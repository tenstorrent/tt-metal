// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fused dispatch writer: copies activation tiles from hidden_states to pkt_buf in DRAM,
// then signals SEM_PKT_READY on the compute leader core.
//
// For N_tokens = P = 32, this is a tile-level copy (all tokens share one tile row).
// Runs on a single dispatch core. After copy, signals compute leader to begin processing.
//
// Semaphores:
//   SEM_PKT_READY (id=2): Incremented on compute leader core after pkt_buf is written.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t hs_addr = get_arg_val<uint32_t>(0);
    const uint32_t pkt_buf_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);  // D_tiles (e.g., 90)
    const uint32_t leader_phys_x = get_arg_val<uint32_t>(3);
    const uint32_t leader_phys_y = get_arg_val<uint32_t>(4);

    constexpr auto hs_args = TensorAccessorArgs<0>();
    constexpr auto pkt_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_temp = 3;
    constexpr uint32_t SEM_PKT_READY = 2;

    const uint32_t page_bytes = get_local_cb_interface(cb_temp).fifo_page_size;

    const auto hs_accessor = TensorAccessor(hs_args, hs_addr, page_bytes);
    const auto pkt_accessor = TensorAccessor(pkt_args, pkt_buf_addr, page_bytes);

    for (uint32_t d = 0; d < num_tiles; ++d) {
        cb_reserve_back(cb_temp, 1);
        uint32_t l1_addr = get_write_ptr(cb_temp);

        noc_async_read_page(d, hs_accessor, l1_addr);
        noc_async_read_barrier();

        noc_async_write_page(d, pkt_accessor, l1_addr);
        noc_async_write_barrier();

        cb_push_back(cb_temp, 1);
        cb_pop_front(cb_temp, 1);
    }

    // Signal compute leader that pkt_buf is ready
    uint64_t leader_sem_addr = get_noc_addr(leader_phys_x, leader_phys_y, get_semaphore(SEM_PKT_READY));
    noc_semaphore_inc(leader_sem_addr, 1);
}
