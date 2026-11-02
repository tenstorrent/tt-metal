// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compact interleaved-RM tilize reader for rows that fit one full-width block: one tile row of
// `tile_height` sticks is read per iteration and pushed as `num_tiles_per_block` tiles.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t tile_height = get_compile_time_arg_val(1);
    constexpr uint32_t num_pages_in_row = get_compile_time_arg_val(2);
    constexpr uint32_t page_width_size = get_compile_time_arg_val(3);
    constexpr uint32_t size_of_valid_data_in_last_page_in_row = get_compile_time_arg_val(4);
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(5);
    constexpr auto src_tensor_args = TensorAccessorArgs<6>();

    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t start_page_id = get_arg_val<uint32_t>(1);

    const auto s = TensorAccessor(src_tensor_args, src_addr);
    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);

    uint32_t page_id = start_page_id;
    for (uint32_t i = 0; i < num_tile_rows; i++) {
        cb_in0.reserve_back(num_tiles_per_block);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();
        for (uint32_t k = 0; k < tile_height; k++) {
            for (uint32_t l = 0; l < num_pages_in_row; l++) {
                uint32_t width_size =
                    (l == num_pages_in_row - 1) ? size_of_valid_data_in_last_page_in_row : page_width_size;
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(s, dst, width_size, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = 0});
                page_id++;
                l1_write_addr += width_size;
            }
        }
        noc.async_read_barrier();
        cb_in0.push_back(num_tiles_per_block);
    }
}
