// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_output = tt::CBIndex::c_16;

constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;
constexpr uint32_t num_tokens = get_compile_time_arg_val(1);
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(2);

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);

    constexpr uint32_t tile_size = 2048;

    // For TILE layout, page size is one tile
    const InterleavedAddrGen<output_is_dram> output_addrg = {.bank_base_address = output_addr, .page_size = tile_size};

    // Each core produces 224 tiles from 32 tokens (32 rows × 224 tile-columns)
    constexpr uint32_t tiles_total = 224;
    cb_wait_front(cb_output, tiles_total);

    uint32_t cb_read_addr = get_read_ptr(cb_output);

    // Calculate starting tile index in output tensor
    // Output tensor: [3200, 7168] in TILE layout = [100 tile-rows, 224 tile-columns]
    // token_start_idx is the starting row (0, 32, 64, ..., 3168)
    // Divide by 32 to get tile-row index (0, 1, 2, ..., 99)
    uint32_t tile_row = token_start_idx / 32;
    uint32_t start_tile_idx = tile_row * 224;

    // Write all 224 tiles contiguously (each core writes one tile-row)
    for (uint32_t tile_idx = 0; tile_idx < 224; ++tile_idx) {
        noc_async_write_page(start_tile_idx + tile_idx, output_addrg, cb_read_addr);
        cb_read_addr += tile_size;
    }

    noc_async_write_barrier();

    cb_pop_front(cb_output, tiles_total);
}
