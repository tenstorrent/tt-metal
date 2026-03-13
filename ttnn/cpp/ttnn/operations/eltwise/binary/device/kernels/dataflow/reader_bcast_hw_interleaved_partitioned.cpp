// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This code is temporarily copied from ttnn/operations/datamovement/binary/device/ to demonstrate
// the new ability to keep the CircularBufferConfigs continuous during dispatching.  See the use of CBIndex::c_2 below.
// When broadcating is properly supported we expect this code to be deleted or refactored substantially.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

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

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_id_in0);
    experimental::CircularBuffer cb1(cb_id_in1);

#ifndef IN0_SHARDED
    const auto s0 = TensorAccessor(src0_args, src0_addr, in0_tile_bytes);
#else
    cb0.reserve_back(num_tiles);
    cb0.push_back(num_tiles);
#endif

    const auto s1 = TensorAccessor(src1_args, src1_addr, in1_tile_bytes);

#ifdef BCAST_SCALAR
    cb1.reserve_back(onetile);
    noc.async_read(s1, cb1, in1_tile_bytes, {.page_id = bcast_id}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb1.push_back(onetile);
#endif

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t curr_id = base_start_id_HtWt + curr_id_from_base;

#ifndef IN0_SHARDED
        cb0.reserve_back(onetile);
        noc.async_read(s0, cb0, in0_tile_bytes, {.page_id = curr_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb0.push_back(onetile);
#endif

        curr_id_from_base++;

#ifndef BCAST_SCALAR
        cb1.reserve_back(onetile);
        noc.async_read(s1, cb1, in1_tile_bytes, {.page_id = bcast_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb1.push_back(onetile);

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
