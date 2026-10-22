// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

// Head core: hands each finished BF16 tile column to the head's sender, then writes the FP32 states compute converts
// to DRAM: the next state (the final state on the last rank), the entry state, and the final state the last rank
// sends the first rank.
void kernel_main() {
    constexpr uint32_t cb_eout = get_compile_time_arg_val(0);
    constexpr uint32_t cb_chron = get_compile_time_arg_val(1);
    constexpr uint32_t sp_rank = get_compile_time_arg_val(2);
    constexpr uint32_t sp_size = get_compile_time_arg_val(3);
    constexpr uint32_t local_rows = get_compile_time_arg_val(4);
    constexpr uint32_t kt = get_compile_time_arg_val(5);
    constexpr uint32_t ack_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_state = get_compile_time_arg_val(7);
    constexpr uint32_t cb_init = get_compile_time_arg_val(8);
    constexpr auto entry_args = TensorAccessorArgs<9>();
    constexpr auto final_args = TensorAccessorArgs<entry_args.next_compile_time_args_offset()>();
    constexpr uint32_t state_tiles = kt * kt;

    const uint32_t entry_addr = get_arg_val<uint32_t>(0);
    const uint32_t final_addr = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2);
    const uint32_t notify_sem_id = get_arg_val<uint32_t>(3);  // this head's column count on its sender core
    const uint32_t sender_x = get_arg_val<uint32_t>(4);
    const uint32_t sender_y = get_arg_val<uint32_t>(5);

    cb_wait_front(cb_chron, 1);
    const uint32_t actual_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_chron))[0];
    const auto t = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    const uint32_t position = (t.rank + sp_size - t.first_rank) % sp_size;
    cb_pop_front(cb_chron, 1);
    // cb_init is filled once, from its base: compute pops it, but nothing writes it again during the call.
    const uint32_t cb_init_address = get_read_ptr(cb_init);

    const auto entry = TensorAccessor(entry_args, entry_addr);
    const auto final_state = TensorAccessor(final_args, final_addr);
    // Hand each tile column to this head's sender as it is packed (a count on the sender's core): the DRAM copies
    // below are off the relay's path.
    const uint64_t notify = get_noc_addr(sender_x, sender_y, get_semaphore(notify_sem_id));
    for (uint32_t j = 0; j < kt; ++j) {
        cb_wait_front(cb_eout, (j + 1) * kt);
        noc_semaphore_inc(notify, 1);
    }
    // The states are stored column-major by tile; the DRAM tensors are row-major.
    const uint32_t tile_bytes = get_tile_size(cb_state);
    const auto page = [&](uint32_t i) { return head * state_tiles + (i % kt) * kt + i / kt; };
    const auto write_state = [&](const auto& tensor, bool keep, uint32_t source) {
        cb_wait_front(cb_state, state_tiles);
        if (keep) {
            for (uint32_t i = 0; i < state_tiles; ++i) {
                noc_async_write_page(page(i), tensor, source + i * tile_bytes);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_state, state_tiles);
    };
    write_state(final_state, position == sp_size - 1, get_read_ptr(cb_state));
    // The first rank's entry is the FP32 initial state itself, which compute's copy would round to TF32.
    write_state(entry, true, position == 0 ? cb_init_address : get_read_ptr(cb_state));
    // Wait until the sender has read the state out of cb_eout.
    volatile tt_l1_ptr uint32_t* acked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ack_sem_id));
    noc_semaphore_wait_min(acked, 1);
    cb_pop_front(cb_eout, state_tiles);
    write_state(final_state, position == 0, get_read_ptr(cb_state));
    noc_async_atomic_barrier();
}
