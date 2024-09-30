// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

uint32_t next_index(uint32_t cur_index, uint32_t start_index, uint32_t num_cores_y, uint32_t block_height, uint32_t block_width) {
    if (cur_index % (num_cores_y * block_width) - start_index % (num_cores_y * block_width) + 1 == block_width) { // go to the next row
        return cur_index - block_width + num_cores_y * block_width + 1;
    } else {
        return cur_index + 1; // also this will be always hit in the interleaved and height sharded cases, because block_width from condition is going to be 0
    }
}

void kernel_main() {
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src1_addr  = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t block_height = get_arg_val<uint32_t>(4);
    uint32_t block_width = get_arg_val<uint32_t>(5);
    uint32_t num_cores_y = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    #ifdef IN0_SHARDED
        cb_reserve_back(cb_id_in0, num_tiles);
        cb_push_back(cb_id_in0, num_tiles);
    #else
    uint32_t l1_write_addr_in0;
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    DataFormat src0_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr,
        .page_size = src0_tile_bytes,
        .data_format = src0_data_format
    };
    #endif
    #ifdef IN1_SHARDED
        cb_reserve_back(cb_id_in1, num_tiles);
        cb_push_back(cb_id_in1, num_tiles);
    #else
    uint32_t l1_write_addr_in1;
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    DataFormat src1_data_format = get_dataformat(cb_id_in1);
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;
    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr,
        .page_size = src1_tile_bytes,
        .data_format = src1_data_format
    };
    #endif

    #if !(defined IN0_SHARDED && defined IN1_SHARDED)

    constexpr uint32_t onetile = 1;
    uint32_t i = start_id;

    while (num_tiles--){
        #ifndef IN0_SHARDED
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(i, s0, l1_write_addr_in0);
        #endif

        #ifndef IN1_SHARDED
        cb_reserve_back(cb_id_in1, onetile);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(i, s1, l1_write_addr_in1);
        #endif

        noc_async_read_barrier();

        #ifndef IN0_SHARDED
        cb_push_back(cb_id_in0, onetile);
        #endif

        #ifndef IN1_SHARDED
        cb_push_back(cb_id_in1, onetile);
        #endif

        i = next_index(i, start_id, num_cores_y, block_height, block_width);
    }
    #endif
}
