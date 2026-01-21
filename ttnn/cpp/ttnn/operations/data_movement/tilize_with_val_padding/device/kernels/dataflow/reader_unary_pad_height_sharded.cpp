// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t cb_id_0 = get_compile_time_arg_val(0);
    constexpr uint32_t element_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t tile_width = get_compile_time_arg_val(3);
    constexpr uint32_t tile_hw = tile_height * tile_width;

    // Runtime args
    uint32_t rt_arg_ind = 0;
    uint32_t src_addr = get_arg_val<uint32_t>(rt_arg_ind++);  // Buffer base address for ShardedAddrGen
    uint32_t logical_width = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t padded_width = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t logical_height = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t padded_height = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t tile_rows = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t num_batches = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t packed_pad_value = get_arg_val<uint32_t>(rt_arg_ind++);

#ifdef SHARDED
    // ShardedAddrGen setup
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(4),    // Memory layout
        get_compile_time_arg_val(5),    // Number of sharding cores
        get_compile_time_arg_val(6),    // Page size
        get_compile_time_arg_val(7),    // Pages per shard row
        get_compile_time_arg_val(8),    // Contiguous pages flag
        get_compile_time_arg_val(9),    // pages_per_shard_x
        get_compile_time_arg_val(10)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(rt_arg_ind));
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {.bank_base_address = src_addr, .shard_array = mapping_table};
#endif

    uint32_t row_bytes = logical_width * element_size;
    uint32_t padded_row_bytes = padded_width * element_size;
    uint32_t width_padding_bytes = padded_row_bytes - row_bytes;

    // Process each batch
    for (uint32_t batch = 0; batch < num_batches; batch++) {
        uint32_t base_row = batch * logical_height;

        // Process tile rows
        for (uint32_t tile_row = 0; tile_row < tile_rows; tile_row++) {
            cb_reserve_back(cb_id_0, tiles_per_row);
            uint32_t l1_write_addr = get_write_ptr(cb_id_0);

            // Process rows within this tile row
            for (uint32_t row_in_tile = 0; row_in_tile < tile_height; row_in_tile++) {
                uint32_t global_row = base_row + tile_row * tile_height + row_in_tile;

                if (global_row < base_row + logical_height) {
                    // Read actual data row
                    uint64_t src_row_addr = get_noc_addr(global_row, s0);
                    noc_async_read(src_row_addr, l1_write_addr, row_bytes);
                    noc_async_read_barrier();  // Barrier per row to make sure data arrives before padding
                    l1_write_addr += row_bytes;

                    // Add width padding
                    if (width_padding_bytes > 0) {
                        volatile tt_l1_ptr uint32_t* pad_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
                        for (uint32_t i = 0; i < width_padding_bytes / 4; i++) {
                            pad_ptr[i] = packed_pad_value;
                        }
                        l1_write_addr += width_padding_bytes;
                    }
                } else {
                    // Height padding row - fill entire row with pad value
                    volatile tt_l1_ptr uint32_t* pad_ptr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
                    for (uint32_t i = 0; i < padded_row_bytes / 4; i++) {
                        pad_ptr[i] = packed_pad_value;
                    }
                    l1_write_addr += padded_row_bytes;
                }
            }

            cb_push_back(cb_id_0, tiles_per_row);
        }
    }
}
