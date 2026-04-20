// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t num_trids = 3;
uint32_t get_next_trid(uint32_t trid) { return trid == num_trids ? 1 : (trid + 1); }

void kernel_main() {
    constexpr auto cb_a = get_compile_time_arg_val(0);
    constexpr auto cb_b = get_compile_time_arg_val(1);

    constexpr auto src_a_args = TensorAccessorArgs<2>();
    constexpr auto src_b_args = TensorAccessorArgs<src_a_args.next_compile_time_args_offset()>();

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_ofs = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t num_batches = get_arg_val<uint32_t>(4);
    const uint32_t num_tiles_per_batch = get_arg_val<uint32_t>(5);

    if (num_tiles == 0) {
        return;
    }

    const uint32_t a_tile_size = get_tile_size(cb_a);
    const uint32_t b_tile_size = get_tile_size(cb_b);

    const auto a_tensor = TensorAccessor(src_a_args, a_addr, a_tile_size);
    const auto b_tensor = TensorAccessor(src_b_args, b_addr, b_tile_size);

    uint64_t a_noc_addr = a_tensor.get_noc_addr(tile_ofs);
    uint64_t b_noc_addr = b_tensor.get_noc_addr(tile_ofs);

    uint32_t a_addr_ofs = 0;
    uint32_t b_addr_ofs = 0;
    uint32_t trid = 1u;  // MUST START WITH ONE
    uint32_t trid_to_wait = trid;

    const uint32_t large_chunk = num_batches * num_tiles_per_batch;
    uint32_t remaining_tiles = num_tiles;
    bool first_iter = true;
    uint32_t prev_chunk = 0;

    auto get_num_tiles = [&](uint32_t remaining_tiles) {
        if (remaining_tiles >= large_chunk) {
            return large_chunk;
        } else if (remaining_tiles >= num_tiles_per_batch) {
            return num_tiles_per_batch;
        }
        return remaining_tiles;
    };
    {
        uint32_t chunk = get_num_tiles(remaining_tiles);
        uint32_t chunk2 = get_num_tiles(remaining_tiles - chunk);
        cb_reserve_back(cb_a, chunk + chunk2);
        cb_reserve_back(cb_b, chunk + chunk2);
    }

    // get_write_ptr must be called after reserve_back
    uint32_t a_write_base_ptr = get_write_ptr(cb_a);
    uint32_t b_write_base_ptr = get_write_ptr(cb_b);
    uint32_t a_write_end_ptr = a_write_base_ptr + CB_PAGE_COUNT(cb_a) * a_tile_size;
    uint32_t b_write_end_ptr = b_write_base_ptr + CB_PAGE_COUNT(cb_b) * b_tile_size;

    uint32_t a_cb_ptr = a_write_base_ptr;
    uint32_t b_cb_ptr = b_write_base_ptr;

    auto next_a_cb_addr = [&](uint32_t addr, uint32_t num_tiles = 1) {
        for (auto j = 0u; j < num_tiles; j++) {
            addr += a_tile_size;
            if (addr >= a_write_end_ptr) {
                addr = a_write_base_ptr;
            }
        }

        return addr;
    };

    auto next_b_cb_addr = [&](uint32_t addr, uint32_t num_tiles = 1) {
        for (auto j = 0u; j < num_tiles; j++) {
            addr += b_tile_size;
            if (addr >= b_write_end_ptr) {
                addr = b_write_base_ptr;
            }
        }
        return addr;
    };

    while (remaining_tiles > 0) {
        uint32_t chunk = get_num_tiles(remaining_tiles);
        uint32_t transfer_sz = chunk * a_tile_size;

        noc_async_read_set_trid(trid);

        noc_async_read_one_packet_set_state<true>(a_noc_addr, transfer_sz);
        noc_async_read_one_packet_with_state_with_trid(a_noc_addr, a_addr_ofs, a_cb_ptr, trid);
        a_addr_ofs += transfer_sz;
        a_cb_ptr = next_a_cb_addr(a_cb_ptr, chunk);

        noc_async_read_one_packet_set_state<true>(b_noc_addr, transfer_sz);
        noc_async_read_one_packet_with_state_with_trid(b_noc_addr, b_addr_ofs, b_cb_ptr, trid);
        b_addr_ofs += transfer_sz;
        b_cb_ptr = next_b_cb_addr(b_cb_ptr, chunk);

        if (!first_iter) {
            noc_async_read_barrier_with_trid(trid_to_wait);

            trid_to_wait = get_next_trid(trid_to_wait);
            cb_push_back(cb_a, prev_chunk);
            cb_push_back(cb_b, prev_chunk);

            // Reserve two blocks of space to issue multiple block reads in parallel
            cb_reserve_back(cb_a, 2 * prev_chunk);
            cb_reserve_back(cb_b, 2 * prev_chunk);
        }

        trid = get_next_trid(trid);
        first_iter = false;
        prev_chunk = chunk;
        remaining_tiles -= chunk;
    }

    noc_async_read_barrier_with_trid(trid_to_wait);
    cb_push_back(cb_a, prev_chunk);
    cb_push_back(cb_b, prev_chunk);
}
