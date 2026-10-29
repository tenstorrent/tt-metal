// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

constexpr uint32_t cb_seed = get_compile_time_arg_val(0);
constexpr uint32_t cb_tail = get_compile_time_arg_val(1);
constexpr uint32_t cb_m = get_compile_time_arg_val(2);
constexpr uint32_t cb_base = get_compile_time_arg_val(3);
constexpr uint32_t cb_a = get_compile_time_arg_val(4);
constexpr uint32_t cb_b = get_compile_time_arg_val(5);
constexpr uint32_t cb_ctl = get_compile_time_arg_val(6);
constexpr uint32_t cb_ctl_writer = get_compile_time_arg_val(7);
constexpr uint32_t seed_semaphore_id = get_compile_time_arg_val(8);
constexpr uint32_t sp_rank = get_compile_time_arg_val(9);
constexpr uint32_t sp_size = get_compile_time_arg_val(10);
constexpr uint32_t local_rows = get_compile_time_arg_val(11);
constexpr uint32_t kt = get_compile_time_arg_val(12);
constexpr bool has_actual_end = get_compile_time_arg_val(13);
// A head's cores each form every final_stride-th tile row of the final state.
constexpr uint32_t final_stride = get_compile_time_arg_val(14);
constexpr auto a_args = TensorAccessorArgs<15>();
constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
constexpr auto entry_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
constexpr auto tail_args = TensorAccessorArgs<entry_args.next_compile_time_args_offset()>();
constexpr auto start_args = TensorAccessorArgs<tail_args.next_compile_time_args_offset()>();
constexpr auto end_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();
constexpr uint32_t state_tiles = kt * kt;

// Entry correction reader: the chronology (this core's valid chunks and the reset chunk), the head's seeds (read by the
// head's first core and written to the head's other cores), and the rows of A and B this core forms the final state of.
void kernel_main() {
    size_t arg = 0;
    const uint32_t a_address = get_arg_val<uint32_t>(arg++);
    const uint32_t b_address = get_arg_val<uint32_t>(arg++);
    const uint32_t entry_address = get_arg_val<uint32_t>(arg++);
    const uint32_t tail_address = get_arg_val<uint32_t>(arg++);
    const uint32_t start_address = get_arg_val<uint32_t>(arg++);
    const uint32_t end_address = get_arg_val<uint32_t>(arg++);
    const uint32_t head = get_arg_val<uint32_t>(arg++);
    const uint32_t first_chunk = get_arg_val<uint32_t>(arg++);
    const uint32_t last_chunk = get_arg_val<uint32_t>(arg++);
    // The first tile row of the final state this core forms (kt: none); it forms every final_stride-th row on.
    const uint32_t final_row = get_arg_val<uint32_t>(arg++);
    // The head's first core reads its seeds and writes them to the head's other cores (NoC x | y << 16 each).
    const bool seed_sender = get_arg_val<uint32_t>(arg++);
    const uint32_t seed_receivers = get_arg_val<uint32_t>(arg++);
    const size_t receiver_args = arg;
    const uint32_t seed_semaphore = get_semaphore(seed_semaphore_id);
    const bool owns_final = final_row < kt;

    // The chronology: valid chunks and the reset chunk (where the tail's seed takes over).
    cb_reserve_back(cb_ctl, 1);
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl));
    const auto start = TensorAccessor(start_args, start_address);
    noc_async_read(start.get_noc_addr(0), get_write_ptr(cb_ctl), sizeof(uint32_t));
    noc_async_read_barrier();
    const uint32_t actual_start = words[0];
    kda_chronology::Topology topology{};
    if constexpr (has_actual_end) {
        const auto end = TensorAccessor(end_args, end_address);
        noc_async_read(end.get_noc_addr(0), get_write_ptr(cb_ctl), sizeof(uint32_t));
        noc_async_read_barrier();
        topology = kda_chronology::derive_interval(actual_start, words[0], sp_rank, sp_size, local_rows);
    } else {
        topology = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    }
    const uint32_t valid = topology.valid_chunks(0, 1);
    const uint32_t reset = topology.reset_chunk(0, 1);
    const uint32_t end_chunk = last_chunk < valid ? last_chunk : valid;
    const uint32_t chunks = end_chunk > first_chunk ? end_chunk - first_chunk : 0;
    // Chunks before the reset take the group's seed, the rest the tail's.
    uint32_t group_chunks = chunks;
    if (reset != 0) {
        group_chunks = reset <= first_chunk ? 0 : (reset - first_chunk < chunks ? reset - first_chunk : chunks);
    }
    const uint32_t final = owns_final && valid != 0;
    const uint32_t final_tail = final && reset != 0;
    // Every core of the head takes the group's seed, and the tail's after a reset: the head's seeds are shared.
    const uint32_t need_group = valid != 0;
    const uint32_t need_tail = valid != 0 && reset != 0;
    words[0] = chunks;
    words[1] = group_chunks;
    words[2] = final;
    words[3] = final_tail;
    words[4] = need_group;
    words[5] = need_tail;
    words[6] = final_row;
    words[7] = final_stride;
    cb_reserve_back(cb_ctl_writer, 1);
    volatile tt_l1_ptr uint32_t* writer_words =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl_writer));
    writer_words[0] = chunks;
    writer_words[1] = final;
    writer_words[2] = final_row;
    writer_words[3] = final_stride;
    cb_push_back(cb_ctl_writer, 1);
    cb_push_back(cb_ctl, 1);

    const uint32_t seeds = need_group + need_tail;
    if (seed_sender) {
        const auto send = [&](uint32_t cb, uint32_t address, const auto& args) {
            cb_reserve_back(cb, state_tiles);
            const uint32_t local = get_write_ptr(cb);
            const auto states = TensorAccessor(args, address);
            const uint32_t tile = get_tile_size(cb);
            for (uint32_t i = 0; i < state_tiles; ++i) {
                noc_async_read_page(head * state_tiles + i, states, local + i * tile);
            }
            noc_async_read_barrier();
            for (uint32_t d = 0; d < seed_receivers; ++d) {
                const uint32_t xy = get_arg_val<uint32_t>(receiver_args + d);
                noc_async_write(local, get_noc_addr(xy & 0xFFFF, xy >> 16, local), state_tiles * tile);
            }
            cb_push_back(cb, state_tiles);
        };
        if (need_group) {
            send(cb_seed, entry_address, entry_args);
        }
        if (need_tail) {
            send(cb_tail, tail_address, tail_args);
        }
        if (seeds != 0) {
            noc_async_write_barrier();
            for (uint32_t d = 0; d < seed_receivers; ++d) {
                const uint32_t xy = get_arg_val<uint32_t>(receiver_args + d);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, seed_semaphore), seeds);
            }
            noc_async_atomic_barrier();
        }
    } else if (seeds != 0) {
        // The seeds land at the same buffer addresses on every core of the program.
        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seed_semaphore);
        noc_semaphore_wait_min(landed, seeds);
        if (need_group) {
            cb_reserve_back(cb_seed, state_tiles);
            cb_push_back(cb_seed, state_tiles);
        }
        if (need_tail) {
            cb_reserve_back(cb_tail, state_tiles);
            cb_push_back(cb_tail, state_tiles);
        }
    }
    // This core's chunk rows are in place in its shard of the emitted terms.
    if (chunks != 0) {
        cb_reserve_back(cb_m, chunks * kt);
        cb_reserve_back(cb_base, chunks * kt);
        cb_push_back(cb_m, chunks * kt);
        cb_push_back(cb_base, chunks * kt);
    }
    if (final) {
        // Tile rows of A and of B: each row of the final state is A[row] @ seed + B[row].
        const auto a = TensorAccessor(a_args, a_address);
        const auto b = TensorAccessor(b_args, b_address);
        const uint32_t tile = get_tile_size(cb_b);
        for (uint32_t row = final_row; row < kt; row += final_stride) {
            cb_reserve_back(cb_a, kt);
            cb_reserve_back(cb_b, kt);
            for (uint32_t c = 0; c < kt; ++c) {
                const uint32_t page = head * state_tiles + row * kt + c;
                noc_async_read_page(page, a, get_write_ptr(cb_a) + c * tile);
                noc_async_read_page(page, b, get_write_ptr(cb_b) + c * tile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, kt);
            cb_push_back(cb_b, kt);
        }
    }
}
