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
// The cores are heads x cores_per_head consecutive cores of the compute grid in row-major order, from first_core,
// head by head.
constexpr uint32_t cores_per_head = get_compile_time_arg_val(6);
constexpr uint32_t grid_x = get_compile_time_arg_val(7);
constexpr uint32_t core_chunks = get_compile_time_arg_val(8);
constexpr uint32_t num_links = get_compile_time_arg_val(9);
constexpr uint32_t first_core = get_compile_time_arg_val(10);
constexpr auto output_args = TensorAccessorArgs<11>();
constexpr auto carry_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
constexpr uint32_t state_tiles = kt * kt;

// Every core of a head writes its chunk outputs. The head's first core is also the relay's head core: it first hands
// each tile column of the next entry state to the head's sender, and on the first rank, between the chunks that start
// from the entry state and those of the separated tail, hands on and writes the final carry.
void kernel_main() {
    // Every core takes the same args (common, per device): its head and share follow from its place in the grid.
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x() - first_core;
    const uint32_t head = core / cores_per_head;
    const uint32_t first_chunk = core % cores_per_head * core_chunks;
    const bool relay_head = core % cores_per_head == 0;
    // Head h is served by link h % num_links: its relay sender, and the link's second carry sender where this rank has
    // one, count its columns in semaphore h / num_links. Their NoC x | y << 16 follow, relay senders first.
    const bool back_sender = get_common_arg_val<uint32_t>(2);
    const uint32_t sender_xy = get_common_arg_val<uint32_t>(3 + head % num_links);
    const uint32_t back_sender_xy = get_common_arg_val<uint32_t>(3 + num_links + head % num_links);
    const uint32_t notify_semaphore = get_semaphore(head / num_links);

    // The reader's chronology: this rank's position and this core's chunk counts.
    cb_wait_front(cb_chron, 1);
    const auto* chron = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_chron));
    const uint32_t position = chron[0];
    const uint32_t chunks = chron[1];
    const uint32_t group_chunks = chron[2];
    cb_pop_front(cb_chron, 1);

    volatile tt_l1_ptr uint32_t* acked =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ack_semaphore_id));
    const uint64_t notify = get_noc_addr(sender_xy & 0xFFFF, sender_xy >> 16, notify_semaphore);
    // Hand each tile column of a state compute forms in cb_eout to the head's sender(s) as it is packed (a count on
    // the sender's core).
    const auto hand_on = [&](bool back) {
        const uint64_t back_notify = get_noc_addr(back_sender_xy & 0xFFFF, back_sender_xy >> 16, notify_semaphore);
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
