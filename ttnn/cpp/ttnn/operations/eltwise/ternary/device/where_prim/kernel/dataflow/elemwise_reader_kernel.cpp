// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    DPRINT << "read kernel start" << ENDL();
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t src2_addr = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);
    uint32_t block_height = get_arg_val<uint32_t>(5);
    uint32_t block_width = get_arg_val<uint32_t>(6);
    uint32_t num_cores_y = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_2;

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    DataFormat src0_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    DataFormat src1_data_format = get_dataformat(cb_id_in1);
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;
    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = src1_tile_bytes, .data_format = src1_data_format};

    uint32_t src2_tile_bytes = get_tile_size(cb_id_in2);
    DataFormat src2_data_format = get_dataformat(cb_id_in2);
    constexpr bool src2_is_dram = get_compile_time_arg_val(2) == 1;
    const InterleavedAddrGenFast<src2_is_dram> s2 = {
        .bank_base_address = src2_addr, .page_size = src2_tile_bytes, .data_format = src2_data_format};

    constexpr uint32_t onetile = 1;
    constexpr bool block_or_width_sharded = get_compile_time_arg_val(3) == 1;

    // DPRINT << "copy data to CB ..." << ENDL();
    for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
        cb_reserve_back(cb_id_in0, onetile);
        noc_async_read_tile(tile_id, s0, get_write_ptr(cb_id_in0));

        cb_reserve_back(cb_id_in1, onetile);
        noc_async_read_tile(tile_id, s1, get_write_ptr(cb_id_in1));

        cb_reserve_back(cb_id_in2, onetile);
        noc_async_read_tile(tile_id, s2, get_write_ptr(cb_id_in2));

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, onetile);
        cb_push_back(cb_id_in1, onetile);
        cb_push_back(cb_id_in2, onetile);
    }
    // DPRINT << "copy data to CB DONE" << ENDL();
}
