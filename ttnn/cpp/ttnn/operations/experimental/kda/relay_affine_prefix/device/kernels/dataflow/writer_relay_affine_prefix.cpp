// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_eout = get_compile_time_arg_val(0);
constexpr uint32_t cb_chron = get_compile_time_arg_val(1);
constexpr uint32_t cb_out = get_compile_time_arg_val(2);
constexpr uint32_t kt = get_compile_time_arg_val(3);
constexpr uint32_t num_chunks = get_compile_time_arg_val(4);
constexpr uint32_t ack_semaphore_id = get_compile_time_arg_val(5);
constexpr auto output_args = TensorAccessorArgs<6>();
constexpr auto carry_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
constexpr uint32_t state_tiles = kt * kt;

// Every core of a head writes its chunk outputs. The head's first core is also the relay's head core: it first hands
// each tile column of the next entry state to the head's sender, and on the first rank, between the chunks that start
// from the entry state and those of the separated tail, hands on and writes the final carry.
void kernel_main() {
    size_t arg = 0;
    const uint32_t head = get_arg_val<uint32_t>(arg++);
    const uint32_t first_chunk = get_arg_val<uint32_t>(arg++);
    const bool relay_head = get_arg_val<uint32_t>(arg++);
    const uint32_t notify_semaphore_id = get_arg_val<uint32_t>(arg++);  // this head's column count on its sender
    const uint32_t sender_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_y = get_arg_val<uint32_t>(arg++);
    // The link's second carry sender, where this rank has one (the same count semaphore id there).
    const bool back_sender = get_arg_val<uint32_t>(arg++);
    const uint32_t back_sender_x = get_arg_val<uint32_t>(arg++);
    const uint32_t back_sender_y = get_arg_val<uint32_t>(arg++);

    // The reader's chronology: this rank's position and this core's chunk counts.
    cb_wait_front(cb_chron, 1);
    const auto* chron = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_chron));
    const uint32_t position = chron[0];
    const uint32_t chunks = chron[1];
    const uint32_t group_chunks = chron[2];
    cb_pop_front(cb_chron, 1);

    volatile tt_l1_ptr uint32_t* acked =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ack_semaphore_id));
    const uint64_t notify = get_noc_addr(sender_x, sender_y, get_semaphore(notify_semaphore_id));
    // Hand each tile column of a state compute forms in cb_eout to the head's sender(s) as it is packed (a count on
    // the sender's core).
    const auto hand_on = [&](bool back) {
        const uint64_t back_notify = get_noc_addr(back_sender_x, back_sender_y, get_semaphore(notify_semaphore_id));
        for (uint32_t j = 0; j < kt; ++j) {
            cb_wait_front(cb_eout, (j + 1) * kt);
            noc_semaphore_inc(notify, 1);
            if (back) {
                noc_semaphore_inc(back_notify, 1);
            }
        }
    };
    if (relay_head) {
        hand_on(false);
        // The sender releases the head once it has read the last column.
        noc_semaphore_wait_min(acked, 1);
        cb_pop_front(cb_eout, state_tiles);
    }

    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(0));
    const uint32_t out_tile = get_tile_size(cb_out);
    const auto write_outputs = [&](uint32_t begin, uint32_t end) {
        for (uint32_t c = begin; c < end; ++c) {
            cb_wait_front(cb_out, kt);
            const uint32_t page = (head * num_chunks + first_chunk + c) * kt;
            for (uint32_t j = 0; j < kt; ++j) {
                noc_async_write_page(page + j, output, get_read_ptr(cb_out) + j * out_tile);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, kt);
        }
    };
    write_outputs(0, group_chunks);

    if (relay_head && position == 0) {
        // The final carry: hand its columns to the senders, then write it (stored column-major by tile) to this rank's
        // row-major carry output.
        hand_on(back_sender);
        const auto carry = TensorAccessor(carry_args, get_common_arg_val<uint32_t>(1));
        const uint32_t tile = get_tile_size(cb_eout);
        for (uint32_t i = 0; i < state_tiles; ++i) {
            noc_async_write_page(head * state_tiles + (i % kt) * kt + i / kt, carry, get_read_ptr(cb_eout) + i * tile);
        }
        noc_async_write_barrier();
        // The relay's release, then each carry sender's.
        noc_semaphore_wait_min(acked, back_sender ? 3 : 2);
        cb_pop_front(cb_eout, state_tiles);
    }

    write_outputs(group_chunks, chunks);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
