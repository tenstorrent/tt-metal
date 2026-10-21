// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

// Head core: hands each finished tile column to the head's sender, then writes the entry state (and, on the last
// rank, the final state) to DRAM; the first rank also writes the final state the last rank sends it.
void kernel_main() {
    constexpr uint32_t cb_eout = get_compile_time_arg_val(0);
    constexpr uint32_t cb_chron = get_compile_time_arg_val(1);
    constexpr uint32_t sp_rank = get_compile_time_arg_val(2);
    constexpr uint32_t sp_size = get_compile_time_arg_val(3);
    constexpr uint32_t local_rows = get_compile_time_arg_val(4);
    constexpr uint32_t kt = get_compile_time_arg_val(5);
    constexpr uint32_t ack_sem_id = get_compile_time_arg_val(6);
    constexpr auto entry_args = TensorAccessorArgs<7>();
    constexpr auto final_args = TensorAccessorArgs<entry_args.next_compile_time_args_offset()>();
    constexpr uint32_t state_tiles = kt * kt;

    const uint32_t entry_addr = get_arg_val<uint32_t>(0);
    const uint32_t final_addr = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2);
    const uint32_t ein_addr = get_arg_val<uint32_t>(3);       // this head's entry state (cb_ein storage)
    const uint32_t final_landing = get_arg_val<uint32_t>(4);  // the last rank's final state lands here
    const uint32_t final_sem = get_arg_val<uint32_t>(5);
    const uint32_t notify_sem_id = get_arg_val<uint32_t>(6);  // this head's column count on its sender core
    const uint32_t sender_x = get_arg_val<uint32_t>(7);
    const uint32_t sender_y = get_arg_val<uint32_t>(8);

    cb_wait_front(cb_chron, 1);
    const uint32_t actual_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_chron))[0];
    const auto t = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    const uint32_t position = (t.rank + sp_size - t.first_rank) % sp_size;
    cb_pop_front(cb_chron, 1);

    const auto entry = TensorAccessor(entry_args, entry_addr);
    const auto final_state = TensorAccessor(final_args, final_addr);
    const uint32_t tile_bytes = get_tile_size(cb_eout);
    // Hand each tile column to this head's sender as it is packed (a count on the sender's core): the DRAM copies
    // below are off the relay's path.
    const uint64_t notify = get_noc_addr(sender_x, sender_y, get_semaphore(notify_sem_id));
    for (uint32_t j = 0; j < kt; ++j) {
        cb_wait_front(cb_eout, (j + 1) * kt);
        noc_semaphore_inc(notify, 1);
    }
    // The states are stored column-major by tile; the DRAM tensors are row-major.
    const auto page = [&](uint32_t i) { return head * state_tiles + (i % kt) * kt + i / kt; };
    // Compute has consumed the entry state; it stays in place until the next call's predecessor writes it.
    for (uint32_t i = 0; i < state_tiles; ++i) {
        noc_async_write_page(page(i), entry, ein_addr + i * tile_bytes);
    }
    if (position == sp_size - 1) {
        for (uint32_t i = 0; i < state_tiles; ++i) {
            noc_async_write_page(page(i), final_state, get_read_ptr(cb_eout) + i * tile_bytes);
        }
    }
    noc_async_write_barrier();
    // Wait until the sender has read the state out of cb_eout.
    volatile tt_l1_ptr uint32_t* acked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ack_sem_id));
    noc_semaphore_wait_min(acked, 1);
    cb_pop_front(cb_eout, state_tiles);
    if (position == 0) {
        // The first rank's split tail enters at the final state, which the last rank sends here.
        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(final_sem);
        noc_semaphore_wait_min(landed, kt);
        noc_semaphore_set(landed, 0);
        for (uint32_t i = 0; i < state_tiles; ++i) {
            noc_async_write_page(page(i), final_state, final_landing + i * tile_bytes);
        }
        noc_async_write_barrier();
    }
    noc_async_atomic_barrier();
}
