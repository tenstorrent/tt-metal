// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/dataflow/gate_matmul_dataflow.hpp"

constexpr uint32_t cb_eout = get_compile_time_arg_val(0);
constexpr uint32_t cb_chron = get_compile_time_arg_val(1);
constexpr uint32_t cb_norm = get_compile_time_arg_val(2);
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
// The normalized outputs are token-major [B, T, H*V]: head h's tiles of chunk c at tile row (h / heads) * num_chunks +
// c, from tile column (h % heads) * kt.
constexpr uint32_t heads = get_compile_time_arg_val(11);
constexpr auto output_args = TensorAccessorArgs<12>();
constexpr auto carry_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
// Slab (slab_kernel_args): the carry output is the slab, written at entry `index * slab_scale + slab_offset`; nothing
// past slab_index_limit. The reader leaves the index in cb_chron's word slab_index_word.
constexpr uint32_t slab_ct = carry_args.next_compile_time_args_offset();
constexpr bool slab = get_compile_time_arg_val(slab_ct);
constexpr uint32_t slab_index_limit = get_compile_time_arg_val(slab_ct + 1);
constexpr uint32_t slab_scale = get_compile_time_arg_val(slab_ct + 2);
constexpr uint32_t slab_offset = get_compile_time_arg_val(slab_ct + 3);
constexpr uint32_t entry_tiles = get_compile_time_arg_val(slab_ct + 4);
constexpr uint32_t slab_index_word = get_compile_time_arg_val(slab_ct + 5);
// The output gate's operands (gate_matmul_dataflow.hpp), and where the common args hold a bit per head: whether its W
// moves north, on this kernel's NoC 1 (else x, see the reader).
constexpr uint32_t north_args = get_compile_time_arg_val(slab_ct + 6);
using Gate = kda_gate::Operands<slab_ct + 7>;
constexpr uint32_t state_tiles = kt * kt;

// Every core of a head writes its chunk outputs, and every other core of a head first moves one of the output gate's
// operands (the reader moves the other). The
// head's first core is also the relay's head core: it first hands each tile column of the next entry state to the
// head's sender, and on the first rank, before its chunk outputs, hands on and writes the final carry.
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

    // The output gate's operand this kernel moves for the core's matmul: the head's W, or x of its chunks (in the
    // group of the head batch's cores at this place).
    if (!relay_head) {
        if (get_common_arg_val<uint32_t>(north_args + head / 32) >> head % 32 & 1) {
            Gate::move_w(head);
        } else {
            Gate::move_x(
                (head / heads) * cores_per_head + core % cores_per_head,
                (head / heads) * num_chunks + first_chunk,
                head);
        }
    }

    // The reader's chronology: this rank's position and this core's chunk counts.
    cb_wait_front(cb_chron, 1);
    const auto* chron = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_chron));
    const uint32_t position = chron[0];
    const uint32_t chunks = chron[1];
    const uint32_t index = slab ? chron[slab_index_word] : 0;
    const bool writes_carry = !slab || index < slab_index_limit;
    const uint32_t carry_base = writes_carry ? (index * slab_scale + slab_offset) * entry_tiles : 0;
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
    const uint32_t out_tile = get_tile_size(cb_norm);
    const auto write_outputs = [&](uint32_t begin, uint32_t end) {
        for (uint32_t c = begin; c < end; ++c) {
            cb_wait_front(cb_norm, kt);
            const uint32_t page = (((head / heads) * num_chunks + first_chunk + c) * heads + head % heads) * kt;
            for (uint32_t j = 0; j < kt; ++j) {
                noc_async_write_page(page + j, output, get_read_ptr(cb_norm) + j * out_tile);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_norm, kt);
        }
    };
    if (relay_head && position == 0) {
        // The final carry: hand its columns to the senders, then write it (stored column-major by tile) to this rank's
        // row-major carry output.
        hand_on(back_sender);
        const auto carry = TensorAccessor(carry_args, get_common_arg_val<uint32_t>(1));
        const uint32_t tile = get_tile_size(cb_eout);
        for (uint32_t i = 0; i < state_tiles && writes_carry; ++i) {
            noc_async_write_page(
                carry_base + head * state_tiles + (i % kt) * kt + i / kt, carry, get_read_ptr(cb_eout) + i * tile);
        }
        noc_async_write_barrier();
        // The relay's release, then each carry sender's.
        noc_semaphore_wait_min(acked, back_sender ? 3 : 2);
        cb_pop_front(cb_eout, state_tiles);
    }

    write_outputs(0, chunks);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
