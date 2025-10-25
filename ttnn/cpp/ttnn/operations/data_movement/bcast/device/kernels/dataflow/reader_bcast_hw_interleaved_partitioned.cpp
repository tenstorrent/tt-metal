// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t HtWt = get_arg_val<uint32_t>(3);
    uint32_t base_start_id_HtWt = get_arg_val<uint32_t>(4);
    uint32_t curr_id_from_base = get_arg_val<uint32_t>(5);
    uint32_t bcast_id = get_arg_val<uint32_t>(6);

#ifndef IN0_SHARDED
    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
#else
    constexpr auto src1_args = TensorAccessorArgs<0>();
#endif

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t onetile = 1;

    // single-tile ublocks
    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

#ifndef IN0_SHARDED
    const auto s0 = TensorAccessor(src0_args, src0_addr, in0_tile_bytes);
#else
    cb_reserve_back(cb_id_in0, num_tiles);
    cb_push_back(cb_id_in0, num_tiles);
#endif

    const auto s1 = TensorAccessor(src1_args, src1_addr, in1_tile_bytes);

#ifdef BCAST_SCALAR
    cb_reserve_back(cb_id_in1, onetile);
    l1_write_addr_in1 = get_write_ptr(cb_id_in1);
    noc_async_read_tile(bcast_id, s1, l1_write_addr_in1);
    noc_async_read_barrier();
    cb_push_back(cb_id_in1, onetile);
#endif

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t curr_id = base_start_id_HtWt + curr_id_from_base;

#ifndef IN0_SHARDED
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(curr_id, s0, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
#endif

        curr_id_from_base++;

#ifndef BCAST_SCALAR
        cb_reserve_back(cb_id_in1, onetile);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(bcast_id, s1, l1_write_addr_in1);
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, onetile);

        if (curr_id_from_base == HtWt) {
            bcast_id++;
#else
        if (curr_id_from_base == HtWt) {
#endif
            base_start_id_HtWt += HtWt;
            curr_id_from_base = 0;
        }
    }
}
