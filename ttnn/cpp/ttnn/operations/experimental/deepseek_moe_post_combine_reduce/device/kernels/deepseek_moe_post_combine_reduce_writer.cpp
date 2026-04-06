// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_output = tt::CBIndex::c_16;

constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(0);
constexpr auto output_accessor_args = TensorAccessorArgs<2>();

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);

    constexpr uint32_t tile_size = 2048;

    const auto output_addrg = TensorAccessor(output_accessor_args, output_addr, tile_size);

    constexpr uint32_t tiles_total = 224;
    cb_wait_front(cb_output, tiles_total);

    uint32_t cb_read_addr = get_read_ptr(cb_output);

    uint32_t tile_row = token_start_idx / 32;
    uint32_t start_tile_idx = tile_row * 224;

    for (uint32_t tile_idx = 0; tile_idx < 224; ++tile_idx) {
        noc_async_write_page(start_tile_idx + tile_idx, output_addrg, cb_read_addr);
        cb_read_addr += tile_size;
    }

    noc_async_write_barrier();

    cb_pop_front(cb_output, tiles_total);
}
