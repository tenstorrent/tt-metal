// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t ELEMENT_SIZE_BYTES = 2;
constexpr uint32_t STICK_SIZE = TILE_SIZE * ELEMENT_SIZE_BYTES;

void kernel_main() {
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t BATCH_SIZE = 8;
    const uint32_t num_batches = total_tiles / BATCH_SIZE;
    const uint32_t leftover = total_tiles % BATCH_SIZE;

    constexpr uint32_t cb_in_transpose = get_compile_time_arg_val(0);
    constexpr uint32_t in_transpose_tile_size = get_tile_size(cb_in_transpose);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);

    experimental::Noc noc;
    experimental::CB cb_transpose(cb_in_transpose);
    experimental::CB cb_out_obj(cb_out);

    cb_out_obj.reserve_back(1);
    const uint32_t base_l1_write_addr = cb_out_obj.get_write_ptr();
    experimental::set_read_state<STICK_SIZE>(noc, cb_transpose.get_read_ptr());

    const uint32_t channel_size = total_tiles * STICK_SIZE;

    int tile_index = 0;
    for (uint32_t i = 0; i < num_batches; i++) {
        cb_transpose.wait_front(BATCH_SIZE);
        uint32_t l1_read_addr_tile = cb_transpose.get_read_ptr();
        for (uint32_t b = 0; b < BATCH_SIZE; b++) {
            uint32_t l1_read_addr = l1_read_addr_tile;
            for (uint32_t j = 0; j < C; j++) {
                const uint32_t l1_write_addr = base_l1_write_addr + (j * channel_size) + (tile_index * STICK_SIZE);
                experimental::read_with_state(noc, l1_write_addr, l1_read_addr);
                l1_read_addr += STICK_SIZE;
            }
            tile_index++;
            l1_read_addr_tile += in_transpose_tile_size;
        }
        noc.async_read_barrier();
        cb_transpose.pop_front(BATCH_SIZE);
    }

    for (uint32_t i = 0; i < leftover; i++) {
        cb_transpose.wait_front(1);
        uint32_t l1_read_addr = cb_transpose.get_read_ptr();
        for (uint32_t j = 0; j < C; j++) {
            const uint32_t l1_write_addr = base_l1_write_addr + (j * channel_size) + (tile_index * STICK_SIZE);
            experimental::read_with_state(noc, l1_write_addr, l1_read_addr);
            l1_read_addr += STICK_SIZE;
        }
        tile_index++;
        noc.async_read_barrier();
        cb_transpose.pop_front(1);
    }
    noc.async_read_barrier();
    cb_out_obj.push_back(1);
}
