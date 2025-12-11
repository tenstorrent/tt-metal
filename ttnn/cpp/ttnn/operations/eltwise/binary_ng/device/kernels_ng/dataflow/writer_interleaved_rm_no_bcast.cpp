// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
// #include "debug/dprint.h"

void kernel_main() {
    uint32_t index = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(index++);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(index++);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(index++);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(index++);

    const uint32_t D = get_arg_val<uint32_t>(index++);
    const uint32_t N = get_arg_val<uint32_t>(index++);
    const uint32_t C = get_arg_val<uint32_t>(index++);
    const uint32_t Ht = get_arg_val<uint32_t>(index++);
    const uint32_t Wt = get_arg_val<uint32_t>(index++);
    const uint32_t cND = get_arg_val<uint32_t>(index++);

    const uint32_t current_row_from_host = get_arg_val<uint32_t>(index++); // not now
    const uint32_t num_rows = get_arg_val<uint32_t>(index++);
    uint32_t page_size = get_arg_val<uint32_t>(index++);

    constexpr auto cb_id_dst = tt::CBIndex::c_2;
    const uint32_t tile_hw = get_tile_hw(cb_id_dst);

    const uint32_t aligned_page_size = ((page_size + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

#if !DST_SHARDED
    constexpr auto dst_args = TensorAccessorArgs<0>();
    const uint32_t dst_tile_bytes = get_tile_size(cb_id_dst);
    const uint32_t element_size = dst_tile_bytes / tile_hw;

    const auto dst = TensorAccessor(dst_args, dst_addr, aligned_page_size);
#endif

    constexpr bool has_sharding = get_compile_time_arg_val(dst_args.next_compile_time_args_offset()) == 1;

    auto row_width = aligned_page_size / element_size;
    const uint32_t div = (row_width + tile_hw - 1) / tile_hw;

    // Calculate batches based on total tiles assigned to this core
    const uint32_t num_batches = dst_num_tiles / div;

    for (uint32_t b = 0; b < num_batches; ++b) {
        // Reset byte counters for every batch/row
        uint32_t bytes_left = aligned_page_size;
        uint32_t bytes_to_write = aligned_page_size > dst_tile_bytes ? dst_tile_bytes : aligned_page_size;

        uint32_t current_calculated_row_start = 0;
        if (div > 1) {
            // Wide row: multiple tiles per row. Row index increases every 'div' tiles.
            current_calculated_row_start = (start_tile_id / div) + b;
        } else {
            // Narrow row: multiple rows per tile. Row index increases by 'num_rows' every tile.
            current_calculated_row_start = (start_tile_id + b) * num_rows;
        }

        for (uint32_t t = 0; t < div; t++) {
            // Handle partial tiles logic
            bytes_to_write = bytes_to_write < bytes_left ? bytes_to_write : bytes_left;

            cb_wait_front(cb_id_dst, 1);
            uint32_t l1_read_addr_src = get_read_ptr(cb_id_dst);

            for (uint32_t i = 0; i < num_rows; i++) {
                // Use calculated row instead of host argument
                uint64_t src_noc_addr = get_noc_addr(current_calculated_row_start + i, dst) + dst_tile_bytes * t;

                noc_async_write(l1_read_addr_src, src_noc_addr, bytes_to_write);

                l1_read_addr_src += bytes_to_write;
            }
            noc_async_write_barrier();

            cb_pop_front(cb_id_dst, 1);

            // Update remaining bytes
            bytes_left -= bytes_to_write;
            bytes_left = ((bytes_left + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;
        }
    }

    // DPRINT << "Writer Exit" << ENDL();
}
