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

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto src2_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2;
    constexpr uint32_t onetile = 1;

    uint32_t l1_write_addr_in0;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const auto s0 = TensorAccessor(src0_args, src0_addr, src0_tile_bytes);
    uint32_t l1_write_addr_in1;
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const auto s1 = TensorAccessor(src1_args, src1_addr, src1_tile_bytes);

    uint32_t l1_write_addr_in2;
    uint32_t src2_tile_bytes = get_tile_size(cb_id_in2);
    const auto s2 = TensorAccessor(src2_args, src2_addr, src2_tile_bytes);

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
