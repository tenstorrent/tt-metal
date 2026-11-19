// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

constexpr uint32_t cb_eout = get_compile_time_arg_val(0);
constexpr uint32_t cb_chron = get_compile_time_arg_val(1);
constexpr uint32_t sp_size = get_compile_time_arg_val(2);
constexpr uint32_t local_rows = get_compile_time_arg_val(3);
constexpr uint32_t kt = get_compile_time_arg_val(4);
constexpr uint32_t ack_semaphore_id = get_compile_time_arg_val(5);
constexpr auto entry_args = TensorAccessorArgs<6>();
constexpr auto final_args = TensorAccessorArgs<entry_args.next_compile_time_args_offset()>();
constexpr uint32_t state_tiles = kt * kt;
// The final carry: the first rank hands compute's final state to its senders column by column, and writes it to its
// own carry output.
constexpr auto carry_args = TensorAccessorArgs<final_args.next_compile_time_args_offset()>();
// Slab (slab_kernel_args): the carry output is the slab, written at entry `index * slab_scale + slab_offset`; nothing
// past slab_index_limit. The reader leaves the index in cb_chron's word slab_index_word.
constexpr uint32_t slab_ct = carry_args.next_compile_time_args_offset();
constexpr bool slab = get_compile_time_arg_val(slab_ct);
constexpr uint32_t slab_index_limit = get_compile_time_arg_val(slab_ct + 1);
constexpr uint32_t slab_scale = get_compile_time_arg_val(slab_ct + 2);
constexpr uint32_t slab_offset = get_compile_time_arg_val(slab_ct + 3);
constexpr uint32_t entry_tiles = get_compile_time_arg_val(slab_ct + 4);
constexpr uint32_t slab_index_word = get_compile_time_arg_val(slab_ct + 5);

// Head core: hands each finished tile column to the head's sender, then writes the entry state (and, on the last
// rank, the final state) to DRAM; the first rank also hands on and writes the final carry, and writes the final state
// the last rank sends it.
void kernel_main() {
    size_t arg = 0;
    const uint32_t entry_address = get_common_arg_val<uint32_t>(0);
    const uint32_t final_address = get_common_arg_val<uint32_t>(1);
    // The landing buffer's slots: this head's entry state (cb_ein storage), the last rank's final state.
    const uint32_t ein_address = get_common_arg_val<uint32_t>(2);
    const uint32_t final_landing = ein_address + state_tiles * get_tile_size(cb_eout);
    const uint32_t head = get_arg_val<uint32_t>(arg++);
    const uint32_t notify_semaphore_id = get_arg_val<uint32_t>(arg++);  // this head's column count on its sender
    const uint32_t sender_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_y = get_arg_val<uint32_t>(arg++);
    // The link's second carry sender, where this rank has one (the same count semaphore id there).
    const bool back_sender = get_arg_val<uint32_t>(arg++);
    const uint32_t back_sender_x = get_arg_val<uint32_t>(arg++);
    const uint32_t back_sender_y = get_arg_val<uint32_t>(arg++);
    const uint32_t sp_rank = get_arg_val<uint32_t>(arg++);

    cb_wait_front(cb_chron, 1);
    const auto* chron_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_chron));
    const uint32_t actual_start = chron_words[0];
    const uint32_t index = slab ? chron_words[slab_index_word] : 0;
    const bool writes_carry = !slab || index < slab_index_limit;
    const uint32_t carry_base = writes_carry ? (index * slab_scale + slab_offset) * entry_tiles : 0;
    const auto topology = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    const uint32_t position = (topology.rank + sp_size - topology.first_rank) % sp_size;
    cb_pop_front(cb_chron, 1);

    const auto entry = TensorAccessor(entry_args, entry_address);
    const auto final_state = TensorAccessor(final_args, final_address);
    const uint32_t tile_bytes = get_tile_size(cb_eout);
    // Hand each tile column to this head's sender as it is packed (a count on the sender's core): the DRAM copies
    // below are off the relay's path.
    const uint64_t notify = get_noc_addr(sender_x, sender_y, get_semaphore(notify_semaphore_id));
    for (uint32_t j = 0; j < kt; ++j) {
        cb_wait_front(cb_eout, (j + 1) * kt);
        noc_semaphore_inc(notify, 1);
    }
    // The states are stored column-major by tile; the DRAM tensors are row-major.
    const auto page = [&](uint32_t i) { return head * state_tiles + (i % kt) * kt + i / kt; };
    // Compute has consumed the entry state; it stays in place until the next call's predecessor writes it.
    for (uint32_t i = 0; i < state_tiles; ++i) {
        noc_async_write_page(page(i), entry, ein_address + i * tile_bytes);
    }
    if (position == sp_size - 1) {
        for (uint32_t i = 0; i < state_tiles; ++i) {
            noc_async_write_page(page(i), final_state, get_read_ptr(cb_eout) + i * tile_bytes);
        }
    }
    noc_async_write_barrier();
    // Wait until the sender has read the state out of cb_eout.
    volatile tt_l1_ptr uint32_t* acked =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ack_semaphore_id));
    noc_semaphore_wait_min(acked, 1);
    cb_pop_front(cb_eout, state_tiles);
    if (position == 0) {
        // Compute's second pass forms the final carry from the relayed final state: hand its columns to the senders,
        // then write it and the relayed state (all landed once the last column is formed) to DRAM.
        const uint64_t back_notify = get_noc_addr(back_sender_x, back_sender_y, get_semaphore(notify_semaphore_id));
        for (uint32_t j = 0; j < kt; ++j) {
            cb_wait_front(cb_eout, (j + 1) * kt);
            noc_semaphore_inc(notify, 1);
            if (back_sender) {
                noc_semaphore_inc(back_notify, 1);
            }
        }
        const auto carry = TensorAccessor(carry_args, get_common_arg_val<uint32_t>(3));
        for (uint32_t i = 0; i < state_tiles; ++i) {
            if (writes_carry) {
                noc_async_write_page(carry_base + page(i), carry, get_read_ptr(cb_eout) + i * tile_bytes);
            }
            noc_async_write_page(page(i), final_state, final_landing + i * tile_bytes);
        }
        noc_async_write_barrier();
        // The relay's release, then each carry sender's.
        noc_semaphore_wait_min(acked, back_sender ? 3 : 2);
        cb_pop_front(cb_eout, state_tiles);
    }
    noc_async_atomic_barrier();
}
