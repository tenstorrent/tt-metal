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
// The tail's seed is the relay's final landing on the first rank, possibly still arriving, instead of tail_entry.
constexpr bool tail_from_landing = get_compile_time_arg_val(15);
constexpr auto a_args = TensorAccessorArgs<16>();
constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
constexpr auto entry_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
constexpr auto tail_args = TensorAccessorArgs<entry_args.next_compile_time_args_offset()>();
constexpr auto start_args = TensorAccessorArgs<tail_args.next_compile_time_args_offset()>();
constexpr auto end_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();
constexpr uint32_t state_tiles = kt * kt;

namespace {

// A head's state in the relay's final landing: on the head's relay core, stored by tile column, its landed tiles
// counted into the arrival counter there.
struct TailLanding {
    uint32_t address;
    uint32_t x;
    uint32_t y;
    uint32_t counter;

    uint64_t counter_address() const { return get_noc_addr(x, y, counter); }
    // Poll the landed-tile count, through one word of this core's L1 at `scratch`, until the whole state is in.
    void wait(uint32_t scratch) const {
        volatile tt_l1_ptr uint32_t* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        do {
            noc_async_read(counter_address(), scratch, sizeof(uint32_t));
            noc_async_read_barrier();
            invalidate_l1_cache();
        } while (count[0] < state_tiles);
    }
    // The state, row-major by tile: tile (i / kt, i % kt) is column i % kt's tile i / kt.
    void read(uint32_t local, uint32_t tile) const {
        for (uint32_t i = 0; i < state_tiles; ++i) {
            const uint32_t stored = (i % kt) * kt + i / kt;
            noc_async_read(get_noc_addr(x, y, address + stored * tile), local + i * tile, tile);
        }
    }
    // Subtract instead of clearing (as the relay's writer did): a faster last rank may already be counting the next
    // call's tiles.
    void consume() const { noc_semaphore_inc(counter_address(), 0u - state_tiles); }
};

}  // namespace

// Entry correction reader: the chronology (this core's valid chunks and the reset chunk), the head's seeds (read by the
// head's first core and multicast to the head's other cores, the group's first so compute starts on it while the tail's
// streams in), and the rows of A and B this core forms the final state of.
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
    // The head's first core reads its seeds and writes them to the head's other cores.
    const bool seed_sender = get_arg_val<uint32_t>(arg++);
    TailLanding landing{};
    if constexpr (tail_from_landing) {
        landing.address = get_arg_val<uint32_t>(arg++);
        const uint32_t xy = get_arg_val<uint32_t>(arg++);
        landing.x = xy & 0xFFFF;
        landing.y = xy >> 16;
        landing.counter = get_arg_val<uint32_t>(arg++);
    }
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

    // This core's chunk rows are in place in its shard of the emitted terms: compute forms the differences (which need
    // no seed) while the seeds arrive.
    if (chunks != 0) {
        cb_reserve_back(cb_m, chunks * kt);
        cb_reserve_back(cb_base, chunks * kt);
        cb_push_back(cb_m, chunks * kt);
        cb_push_back(cb_base, chunks * kt);
    }
    // Tile rows of A and of B: each row of the final state is A[row] @ seed + B[row].
    const auto a = TensorAccessor(a_args, a_address);
    const auto b = TensorAccessor(b_args, b_address);
    uint32_t row = final_row;
    const auto read_final_row = [&]() {
        const uint32_t tile = get_tile_size(cb_b);
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
        row += final_stride;
    };

    // The head's seeds, counted on every other core of the head by its seed semaphore: 1 once the group's seed has
    // landed, 2 once the tail's has. Compute starts the group's chunks as soon as the group's seed is in.
    const uint32_t seeds = need_group + need_tail;
    if (seed_sender) {
        // The head's other cores: one multicast rectangle when they form one (start, end in NoC 0 order, x | y << 16),
        // then every core (x | y << 16) for the semaphore increments and the unicast fallback.
        const bool multicast = get_arg_val<uint32_t>(receiver_args) != 0;
        const size_t list_args = receiver_args + 3;
        uint32_t x0 = get_arg_val<uint32_t>(receiver_args + 1) & 0xFFFF;
        uint32_t y0 = get_arg_val<uint32_t>(receiver_args + 1) >> 16;
        uint32_t x1 = get_arg_val<uint32_t>(receiver_args + 2) & 0xFFFF;
        uint32_t y1 = get_arg_val<uint32_t>(receiver_args + 2) >> 16;
        if (noc_index == 1) {
            // NoC 1 runs the other way: its rectangle starts at the far corner.
            const uint32_t x = x0;
            const uint32_t y = y0;
            x0 = x1;
            y0 = y1;
            x1 = x;
            y1 = y;
        }
        const uint32_t tile = get_tile_size(cb_seed);
        const auto read = [&](uint32_t cb, uint32_t address, const auto& args) {
            cb_reserve_back(cb, state_tiles);
            const uint32_t local = get_write_ptr(cb);
            const auto states = TensorAccessor(args, address);
            for (uint32_t i = 0; i < state_tiles; ++i) {
                noc_async_read_page(head * state_tiles + i, states, local + i * tile);
            }
            return local;
        };
        // A landed seed: this core's compute takes it, the head's other cores get it and then one increment each.
        const auto send = [&](uint32_t cb, uint32_t local) {
            cb_push_back(cb, state_tiles);
            if (seed_receivers == 0) {
                return;
            }
            if (multicast) {
                noc_async_write_multicast(
                    local, get_noc_multicast_addr(x0, y0, x1, y1, local), state_tiles * tile, seed_receivers);
            } else {
                for (uint32_t d = 0; d < seed_receivers; ++d) {
                    const uint32_t xy = get_arg_val<uint32_t>(list_args + d);
                    noc_async_write(local, get_noc_addr(xy & 0xFFFF, xy >> 16, local), state_tiles * tile);
                }
            }
            noc_async_write_barrier();
            for (uint32_t d = 0; d < seed_receivers; ++d) {
                const uint32_t xy = get_arg_val<uint32_t>(list_args + d);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, seed_semaphore), 1);
            }
        };
        if (need_group) {
            const uint32_t group = read(cb_seed, entry_address, entry_args);
            noc_async_read_barrier();
            if constexpr (tail_from_landing) {
                // The last rank's final state may still be landing: the head's cores start on the group's chunks
                // meanwhile.
                send(cb_seed, group);
                if (need_tail) {
                    cb_reserve_back(cb_tail, state_tiles);
                    const uint32_t tail = get_write_ptr(cb_tail);
                    landing.wait(tail);
                    landing.read(tail, tile);
                    noc_async_read_barrier();
                    send(cb_tail, tail);
                }
            } else {
                // The tail's seed streams in while the group's goes out.
                const uint32_t tail = need_tail ? read(cb_tail, tail_address, tail_args) : 0;
                send(cb_seed, group);
                if (need_tail) {
                    noc_async_read_barrier();
                    send(cb_tail, tail);
                }
            }
            noc_async_atomic_barrier();
        }
    } else if (seeds != 0) {
        // This core's first final row needs no seed: it lands while the seeds are on their way.
        if (final) {
            read_final_row();
        }
        // The seeds land at the same buffer addresses on every core of the program.
        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seed_semaphore);
        noc_semaphore_wait_min(landed, 1);
        cb_reserve_back(cb_seed, state_tiles);
        cb_push_back(cb_seed, state_tiles);
        if (need_tail) {
            noc_semaphore_wait_min(landed, 2);
            cb_reserve_back(cb_tail, state_tiles);
            cb_push_back(cb_tail, state_tiles);
        }
    }
    if (final) {
        while (row < kt) {
            read_final_row();
        }
    }
    if constexpr (tail_from_landing) {
        // The last rank sends its final state to the first rank every call, whether or not a tail reads it: the
        // head's first core consumes its landed count (after reading it, when the tail did).
        if (seed_sender && topology.rank == topology.first_rank) {
            if (!need_tail) {
                // cb_tail is unused without a tail: one word of it holds the polled count.
                cb_reserve_back(cb_tail, state_tiles);
                landing.wait(get_write_ptr(cb_tail));
            }
            landing.consume();
            noc_async_atomic_barrier();
        }
    }
}
