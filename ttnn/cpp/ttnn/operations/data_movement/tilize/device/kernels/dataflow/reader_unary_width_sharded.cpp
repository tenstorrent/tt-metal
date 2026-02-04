// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor_args.h"
#include "api/tensor/tensor_accessor.h"
#include "api/debug/dprint_pages.h"

void kernel_main() {
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t row_width = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores_total = get_compile_time_arg_val(2);
    constexpr uint32_t datum_size = get_compile_time_arg_val(3);
    constexpr auto src_tensor_args = TensorAccessorArgs<4>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t height_per_core = get_arg_val<uint32_t>(1);
    const uint32_t responsibility = get_arg_val<uint32_t>(2);
    const uint32_t start_core_number = get_arg_val<uint32_t>(3);

    const auto src_accessor = TensorAccessor(src_tensor_args, src_addr, row_width);

    // Calculate element size in bytes and tile width in bytes
    constexpr uint32_t tile_width_bytes = tt::constants::TILE_WIDTH * datum_size;

    uint32_t core_number = start_core_number;

    uint64_t noc_addrs[tt::constants::TILE_HEIGHT];

    // Get NOC addresses (one for each row in a tile)
    for (uint32_t i = 0; i < tt::constants::TILE_HEIGHT; i++) {
        noc_addrs[i] = src_accessor.get_noc_addr(core_number + num_cores_total * i, 0);
    }

    for (uint32_t tile_col = 0; tile_col < responsibility; tile_col++) {
        // Reserve space in circular buffer for 1024 datums (one tile)
        cb_reserve_back(cb_id_out, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_out);
        uint64_t offset = tile_col * tile_width_bytes;

        // Inner loop for 32 rows (tile_height)
        for (uint32_t tile_row = 0; tile_row < tt::constants::TILE_HEIGHT; tile_row++) {
            noc_async_read(noc_addrs[tile_row] + offset, l1_write_addr, tile_width_bytes);

            l1_write_addr += tile_width_bytes;
        }

        noc_async_read_barrier();
        cb_push_back(cb_id_out, 1);
    }
}
