// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

constexpr uint32_t cb_a = get_compile_time_arg_val(0);
constexpr uint32_t cb_b = get_compile_time_arg_val(1);
constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
constexpr uint32_t cb_chron = get_compile_time_arg_val(3);
constexpr uint32_t sp_rank = get_compile_time_arg_val(4);
constexpr uint32_t sp_size = get_compile_time_arg_val(5);
constexpr uint32_t local_rows = get_compile_time_arg_val(6);
constexpr uint32_t kt = get_compile_time_arg_val(7);  // key tiles; the state is kt x kt tiles
constexpr uint32_t read_turn_semaphore_id = get_compile_time_arg_val(8);
constexpr auto packed_args = TensorAccessorArgs<9>();
constexpr auto init_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
constexpr auto start_args = TensorAccessorArgs<init_args.next_compile_time_args_offset()>();
constexpr uint32_t state_tiles = kt * kt;

// Read transaction ids: the chronology word, A, then one per tile column of B (and of the initial state).
constexpr uint32_t chron_trid = 1;
constexpr uint32_t a_trid = 2;
constexpr uint32_t first_column_trid = 3;
static_assert(first_column_trid + kt <= 16, "one read transaction id per tile column");

// Head core: this rank's A and B blocks of the head, then its entry state column by column (the initial state on the
// chronologically first rank, else the predecessor's landed columns).
//
// On the first rank these DRAM reads start the whole relay. Every head core reading at once would share the DRAM
// bandwidth evenly, so the first head of the stream would get its data only when nearly all of it has landed. Instead
// the head cores take turns in stream order: each reads A (all of which its first column needs), then passes the turn
// to the next head on its link while its own B and initial state land column by column, each column handed to compute
// as soon as its own reads land. Measured on the first rank, this lets the first column out ~6 us sooner.
void kernel_main() {
    size_t arg = 0;
    const uint32_t packed_address = get_arg_val<uint32_t>(arg++);
    const uint32_t init_address = get_arg_val<uint32_t>(arg++);
    const uint32_t start_address = get_arg_val<uint32_t>(arg++);
    const uint32_t head = get_arg_val<uint32_t>(arg++);
    const uint32_t arrival_semaphore = get_arg_val<uint32_t>(arg++);
    const bool waits_turn = get_arg_val<uint32_t>(arg++) != 0;  // a head before it on its link reads first
    const bool passes_turn = get_arg_val<uint32_t>(arg++) != 0;
    const uint32_t next_x = get_arg_val<uint32_t>(arg++);  // the next head core on its link
    const uint32_t next_y = get_arg_val<uint32_t>(arg++);

    const uint32_t read_turn = get_semaphore(read_turn_semaphore_id);
    if (waits_turn) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(read_turn), 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(read_turn), 0);
    }

    cb_reserve_back(cb_chron, 1);
    const auto start = TensorAccessor(start_args, start_address);
    noc_async_read_set_trid(chron_trid);
    noc_async_read(start.get_noc_addr(0), get_write_ptr(cb_chron), sizeof(uint32_t));
    const auto packed = TensorAccessor(packed_args, packed_address);
    const auto init = TensorAccessor(init_args, init_address);
    cb_reserve_back(cb_a, state_tiles);
    cb_reserve_back(cb_b, state_tiles);
    // The entry state is stored column-major by tile, so each tile column (all the next state's column needs) can
    // land, be computed and leave on its own.
    cb_reserve_back(cb_ein, state_tiles);
    const uint32_t ab_tile = get_tile_size(cb_a);
    const uint32_t state_tile = get_tile_size(cb_ein);
    // [A | B] rows of this head: kt tile rows of 2 kt tiles. A is kept row-major by tile, B column-major.
    const auto packed_page = [&](uint32_t r, uint32_t c) { return (head * kt + r) * 2 * kt + c; };
    noc_async_read_set_trid(a_trid);
    for (uint32_t r = 0; r < kt; ++r) {
        for (uint32_t c = 0; c < kt; ++c) {
            noc_async_read_page(packed_page(r, c), packed, get_write_ptr(cb_a) + (r * kt + c) * ab_tile);
        }
    }
    // The chronology says whether this is the first rank, the only one that reads the initial state.
    noc_async_read_barrier_with_trid(chron_trid);
    const uint32_t actual_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_chron))[0];
    const auto topology = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    const uint32_t position = (topology.rank + sp_size - topology.first_rank) % sp_size;
    cb_push_back(cb_chron, 1);
    for (uint32_t j = 0; j < kt; ++j) {
        noc_async_read_set_trid(first_column_trid + j);
        for (uint32_t r = 0; r < kt; ++r) {
            noc_async_read_page(packed_page(r, kt + j), packed, get_write_ptr(cb_b) + (j * kt + r) * ab_tile);
        }
        if (position == 0) {
            for (uint32_t r = 0; r < kt; ++r) {
                noc_async_read_page(
                    head * state_tiles + r * kt + j, init, get_write_ptr(cb_ein) + (j * kt + r) * state_tile);
            }
        }
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier_with_trid(a_trid);
    cb_push_back(cb_a, state_tiles);
    if (passes_turn) {
        noc_semaphore_inc(get_noc_addr(next_x, next_y, read_turn), 1);
    }
    for (uint32_t j = 0; j < kt; ++j) {
        noc_async_read_barrier_with_trid(first_column_trid + j);
        cb_push_back(cb_b, kt);
        if (position == 0) {
            cb_push_back(cb_ein, kt);
        }
    }

    if (position != 0) {
        // The predecessor's sender writes this head's entry state into cb_ein's storage in order, counting the tiles
        // that have landed; each is handed to compute as soon as it is counted.
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_semaphore);
        for (uint32_t pushed = 0; pushed < state_tiles;) {
            noc_semaphore_wait_min(arrived, pushed + 1);
            // A faster predecessor may already be counting the next call's tiles.
            const uint32_t count = *arrived < state_tiles ? *arrived : state_tiles;
            cb_push_back(cb_ein, count - pushed);
            pushed = count;
        }
        // Subtract instead of clearing, for the same reason.
        noc_semaphore_inc(get_noc_addr(arrival_semaphore), 0u - state_tiles);
    }
    noc_async_atomic_barrier();
}
