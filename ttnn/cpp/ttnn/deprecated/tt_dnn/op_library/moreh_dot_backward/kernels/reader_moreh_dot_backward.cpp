// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t has_input_grad = get_arg_val<uint32_t>(0);
    uint32_t has_other_grad = get_arg_val<uint32_t>(1);
    uint32_t src0_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_addr = get_arg_val<uint32_t>(3);
    uint32_t src2_addr = get_arg_val<uint32_t>(4);
    uint32_t num_tiles = get_arg_val<uint32_t>(5);
    uint32_t start_id = get_arg_val<uint32_t>(6);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool src2_is_dram = get_compile_time_arg_val(2) == 1;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2;
    constexpr uint32_t onetile = 1;

    uint32_t l1_write_addr_in0;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    DataFormat src0_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};
    uint32_t l1_write_addr_in1;
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    DataFormat src1_data_format = get_dataformat(cb_id_in1);
    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = src1_tile_bytes, .data_format = src1_data_format};

    uint32_t l1_write_addr_in2;
    uint32_t src2_tile_bytes = get_tile_size(cb_id_in2);
    DataFormat src2_data_format = get_dataformat(cb_id_in2);
    const InterleavedAddrGenFast<src2_is_dram> s2 = {
        .bank_base_address = src2_addr, .page_size = src2_tile_bytes, .data_format = src2_data_format};

    cb_reserve_back(cb_id_in0, onetile);
    l1_write_addr_in0 = get_write_ptr(cb_id_in0);
    noc_async_read_tile(0, s0, l1_write_addr_in0);
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, onetile);

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        if (has_input_grad) {
            cb_reserve_back(cb_id_in2, onetile);
            l1_write_addr_in2 = get_write_ptr(cb_id_in2);
            noc_async_read_tile(i, s2, l1_write_addr_in2);
            noc_async_read_barrier();
            cb_push_back(cb_id_in2, onetile);
        }

        if (has_other_grad) {
            cb_reserve_back(cb_id_in1, onetile);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read_tile(i, s1, l1_write_addr_in1);
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, onetile);
        }
    }
}
