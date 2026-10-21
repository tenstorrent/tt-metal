// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

// Head core: this rank's [A | B] rows of the head, then its entry state column by column (the initial state on the
// chronologically first rank, else the predecessor's landed columns).
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
    constexpr uint32_t cb_chron = get_compile_time_arg_val(3);
    constexpr uint32_t sp_rank = get_compile_time_arg_val(4);
    constexpr uint32_t sp_size = get_compile_time_arg_val(5);
    constexpr uint32_t local_rows = get_compile_time_arg_val(6);
    constexpr uint32_t kt = get_compile_time_arg_val(7);  // key tiles; the state is kt x kt tiles
    constexpr auto packed_args = TensorAccessorArgs<8>();
    constexpr auto init_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    constexpr auto start_args = TensorAccessorArgs<init_args.next_compile_time_args_offset()>();
    constexpr uint32_t state_tiles = kt * kt;

    const uint32_t packed_addr = get_arg_val<uint32_t>(0);
    const uint32_t init_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_addr = get_arg_val<uint32_t>(2);
    const uint32_t head = get_arg_val<uint32_t>(3);
    const uint32_t arrival_sem = get_arg_val<uint32_t>(4);

    cb_reserve_back(cb_chron, 1);
    const auto start = TensorAccessor(start_args, start_addr);
    noc_async_read(start.get_noc_addr(0), get_write_ptr(cb_chron), sizeof(uint32_t));
    const auto packed = TensorAccessor(packed_args, packed_addr);
    cb_reserve_back(cb_a, state_tiles);
    cb_reserve_back(cb_b, state_tiles);
    const uint32_t ab_tile = get_tile_size(cb_a);
    // [A | B] rows of this head: kt tile rows of 2 kt tiles.
    for (uint32_t r = 0; r < kt; ++r) {
        for (uint32_t c = 0; c < kt; ++c) {
            const uint32_t page = (head * kt + r) * 2 * kt + c;
            noc_async_read_page(page, packed, get_write_ptr(cb_a) + (r * kt + c) * ab_tile);
            noc_async_read_page(page + kt, packed, get_write_ptr(cb_b) + (r * kt + c) * ab_tile);
        }
    }
    noc_async_read_barrier();
    const uint32_t actual_start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_chron))[0];
    const auto t = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    const uint32_t position = (t.rank + sp_size - t.first_rank) % sp_size;
    cb_push_back(cb_chron, 1);
    cb_push_back(cb_a, state_tiles);
    cb_push_back(cb_b, state_tiles);

    // The entry state is stored column-major by tile, so each tile column (all the next state's column needs) can
    // land, be computed and leave on its own.
    cb_reserve_back(cb_ein, state_tiles);
    if (position == 0) {
        const auto init = TensorAccessor(init_args, init_addr);
        const uint32_t state_tile = get_tile_size(cb_ein);
        for (uint32_t j = 0; j < kt; ++j) {
            for (uint32_t r = 0; r < kt; ++r) {
                noc_async_read_page(
                    head * state_tiles + r * kt + j, init, get_write_ptr(cb_ein) + (j * kt + r) * state_tile);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_ein, state_tiles);
    } else {
        // The predecessor's sender writes each tile column of this head's entry state into cb_ein's storage, then
        // increments.
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_sem);
        for (uint32_t j = 0; j < kt; ++j) {
            noc_semaphore_wait_min(arrived, j + 1);
            cb_push_back(cb_ein, kt);
        }
        noc_semaphore_set(arrived, 0);
    }
}
