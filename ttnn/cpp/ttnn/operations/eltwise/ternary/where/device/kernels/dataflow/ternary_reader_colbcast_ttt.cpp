// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t src2_addr = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(5);

    // Broadcast flags - compile time args
    constexpr bool bcast_in1 = get_compile_time_arg_val(6) == 1;  // value_true broadcast
    constexpr bool bcast_in2 = get_compile_time_arg_val(7) == 1;  // value_false broadcast

    uint32_t l1_write_addr_in0;
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    DataFormat src0_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    uint32_t l1_write_addr_in1;
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    DataFormat src1_data_format = get_dataformat(cb_id_in1);
    constexpr bool src1_is_dram = get_compile_time_arg_val(2) == 1;
    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = src1_tile_bytes, .data_format = src1_data_format};

    uint32_t l1_write_addr_in2;
    uint32_t src2_tile_bytes = get_tile_size(cb_id_in2);
    DataFormat src2_data_format = get_dataformat(cb_id_in2);
    constexpr bool src2_is_dram = get_compile_time_arg_val(4) == 1;
    const InterleavedAddrGenFast<src2_is_dram> s2 = {
        .bank_base_address = src2_addr, .page_size = src2_tile_bytes, .data_format = src2_data_format};

    constexpr uint32_t onetile = 1;

    for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
        // Read predicate (always full)
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(tile_id, s0, l1_write_addr_in0);

        // Read value_true
        cb_reserve_back(cb_id_in1, onetile);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        if constexpr (bcast_in1) {
            // For broadcast, read only the first column of this tile
            uint32_t col_tile_id = (tile_id / 32) * 32;  // Get tile ID for first column in row
            noc_async_read_tile(col_tile_id, s1, l1_write_addr_in1);
        } else {
            noc_async_read_tile(tile_id, s1, l1_write_addr_in1);
        }

        // Read value_false
        cb_reserve_back(cb_id_in2, onetile);
        l1_write_addr_in2 = get_write_ptr(cb_id_in2);
        if constexpr (bcast_in2) {
            // For broadcast, read only the first column of this tile
            uint32_t col_tile_id = (tile_id / 32) * 32;  // Get tile ID for first column in row
            noc_async_read_tile(col_tile_id, s2, l1_write_addr_in2);
        } else {
            noc_async_read_tile(tile_id, s2, l1_write_addr_in2);
        }

        noc_async_read_barrier();

        // Apply broadcast fill if needed
        if constexpr (bcast_in1) {
            FILL_TILE_WITH_FIRST_COLUMN(cb_id_in1);
        }
        if constexpr (bcast_in2) {
            FILL_TILE_WITH_FIRST_COLUMN_B(cb_id_in2);
        }

        cb_push_back(cb_id_in0, onetile);
        cb_push_back(cb_id_in1, onetile);
        cb_push_back(cb_id_in2, onetile);
    }
}
