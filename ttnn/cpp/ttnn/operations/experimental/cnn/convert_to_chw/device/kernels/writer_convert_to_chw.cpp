// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t ELEMENT_SIZE_BYTES = 2;
constexpr uint32_t STICK_SIZE = TILE_SIZE * ELEMENT_SIZE_BYTES;

template <uint32_t channels>
TT_KERNEL void writer(uint32_t tiles_per_core) {
    constexpr uint32_t BATCH_SIZE = 8;
    const uint32_t num_batches = tiles_per_core / BATCH_SIZE;
    const uint32_t leftover = tiles_per_core % BATCH_SIZE;

    Noc noc;
    DataflowBuffer transpose_input(dfb::transpose);
    DataflowBuffer output(dfb::output);
    constexpr uint32_t in_transpose_tile_size = get_tile_size(dfb::transpose);

    output.reserve_back(1);
    const uint32_t base_l1_write_addr = output.get_write_ptr();
    experimental::set_read_state<STICK_SIZE>(noc, transpose_input.get_read_ptr());

    const uint32_t channel_size = tiles_per_core * STICK_SIZE;

    int tile_index = 0;
    for (uint32_t i = 0; i < num_batches; i++) {
        transpose_input.wait_front(BATCH_SIZE);
        uint32_t l1_read_addr_tile = transpose_input.get_read_ptr();
        for (uint32_t b = 0; b < BATCH_SIZE; b++) {
            uint32_t l1_read_addr = l1_read_addr_tile;
            for (uint32_t j = 0; j < channels; j++) {
                const uint32_t l1_write_addr = base_l1_write_addr + (j * channel_size) + (tile_index * STICK_SIZE);
                experimental::read_with_state(noc, l1_write_addr, l1_read_addr);
                l1_read_addr += STICK_SIZE;
            }
            tile_index++;
            l1_read_addr_tile += in_transpose_tile_size;
        }
        noc.async_read_barrier();
        transpose_input.pop_front(BATCH_SIZE);
    }

    for (uint32_t i = 0; i < leftover; i++) {
        transpose_input.wait_front(1);
        uint32_t l1_read_addr = transpose_input.get_read_ptr();
        for (uint32_t j = 0; j < channels; j++) {
            const uint32_t l1_write_addr = base_l1_write_addr + (j * channel_size) + (tile_index * STICK_SIZE);
            experimental::read_with_state(noc, l1_write_addr, l1_read_addr);
            l1_read_addr += STICK_SIZE;
        }
        tile_index++;
        noc.async_read_barrier();
        transpose_input.pop_front(1);
    }
    noc.async_read_barrier();
    output.push_back(1);
}
