// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

// Head core: this rank's [A | B] rows of the head, then its entry state column by column (the FP32 initial state on
// the chronologically first rank, else the predecessor's landed BF16 columns), then the final landing (on the first
// rank once the last rank's final state has landed).
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
    constexpr uint32_t cb_chron = get_compile_time_arg_val(3);
    constexpr uint32_t sp_rank = get_compile_time_arg_val(4);
    constexpr uint32_t sp_size = get_compile_time_arg_val(5);
    constexpr uint32_t local_rows = get_compile_time_arg_val(6);
    constexpr uint32_t kt = get_compile_time_arg_val(7);  // key tiles; the state is kt x kt tiles
    constexpr uint32_t cb_init = get_compile_time_arg_val(8);
    constexpr uint32_t cb_final = get_compile_time_arg_val(9);
    constexpr uint32_t cb_first = get_compile_time_arg_val(10);
    constexpr auto packed_args = TensorAccessorArgs<11>();
    constexpr auto init_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    constexpr auto start_args = TensorAccessorArgs<init_args.next_compile_time_args_offset()>();
    constexpr uint32_t state_tiles = kt * kt;

    const uint32_t packed_addr = get_arg_val<uint32_t>(0);
    const uint32_t init_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_addr = get_arg_val<uint32_t>(2);
    const uint32_t head = get_arg_val<uint32_t>(3);
    const uint32_t arrival_sem = get_arg_val<uint32_t>(4);
    const uint32_t final_sem = get_arg_val<uint32_t>(5);

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
    const bool first = t.rank == t.first_rank;
    cb_reserve_back(cb_first, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_first))[0] = first;
    cb_push_back(cb_first, 1);
    cb_push_back(cb_chron, 1);
    cb_push_back(cb_a, state_tiles);
    cb_push_back(cb_b, state_tiles);

    // The entry state is stored column-major by tile, so each tile column (all the next state's column needs) can
    // land, be computed and leave on its own.
    if (first) {
        cb_reserve_back(cb_init, state_tiles);
        const auto init = TensorAccessor(init_args, init_addr);
        const uint32_t state_tile = get_tile_size(cb_init);
        for (uint32_t j = 0; j < kt; ++j) {
            for (uint32_t r = 0; r < kt; ++r) {
                noc_async_read_page(
                    head * state_tiles + r * kt + j, init, get_write_ptr(cb_init) + (j * kt + r) * state_tile);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_init, state_tiles);
    } else {
        // The predecessor's sender writes each tile column of this head's entry state into cb_ein's storage, then
        // increments.
        cb_reserve_back(cb_ein, state_tiles);
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_sem);
        for (uint32_t j = 0; j < kt; ++j) {
            noc_semaphore_wait_min(arrived, j + 1);
            cb_push_back(cb_ein, kt);
        }
        noc_semaphore_set(arrived, 0);
    }
    cb_reserve_back(cb_final, state_tiles);
    if (first) {
        // The first rank's split tail enters at the final state, which the last rank sends here.
        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(final_sem);
        noc_semaphore_wait_min(landed, kt);
        noc_semaphore_set(landed, 0);
    }
    cb_push_back(cb_final, state_tiles);
}
