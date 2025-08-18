// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This code is temporarily copied from ttnn/cpp/ttnn/operations/datamovement/binary/device/ to demonstrate
// the new ability to keep the CircularBufferConfigs continuous during dispatching.  See the use of CBIndex::c_2 below.
// When broadcating is properly supported we expect this code to be deleted or refactored substantially.

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lengths for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t block_height = get_arg_val<uint32_t>(4);
    uint32_t block_width = get_arg_val<uint32_t>(5);
    uint32_t num_cores_y = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr bool block_or_width_sharded = get_compile_time_arg_val(0) == 1;
#if !defined(IN0_SHARDED) && !defined(IN1_SHARDED)
    constexpr auto src0_args = TensorAccessorArgs<1>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
#elif !defined(IN0_SHARDED)
    constexpr auto src0_args = TensorAccessorArgs<1>();
#elif !defined(IN1_SHARDED)
    constexpr auto src1_args = TensorAccessorArgs<1>();
#endif

#ifdef IN0_SHARDED
    cb_reserve_back(cb_id_in0, num_tiles);
    cb_push_back(cb_id_in0, num_tiles);
#else
    uint32_t l1_write_addr_in0;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const auto s0 = TensorAccessor(src0_args, src0_addr, src0_tile_bytes);
#endif
#ifdef IN1_SHARDED
    cb_reserve_back(cb_id_in1, num_tiles);
    cb_push_back(cb_id_in1, num_tiles);
#else
    uint32_t l1_write_addr_in1;
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const auto s1 = TensorAccessor(src1_args, src1_addr, src1_tile_bytes);
#endif

#if !(defined IN0_SHARDED && defined IN1_SHARDED)

    constexpr uint32_t onetile = 1;

    if constexpr (block_or_width_sharded) {
        uint32_t row_start_tile_id = start_id;
        for (uint32_t h = 0; h < block_height; h++) {
            uint32_t tile_id = row_start_tile_id;
            for (uint32_t w = 0; w < block_width; w++) {
#ifndef IN0_SHARDED
                cb_reserve_back(cb_id_in0, onetile);
                l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(tile_id, s0, l1_write_addr_in0);
#endif

#ifndef IN1_SHARDED
                cb_reserve_back(cb_id_in1, onetile);
                l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(tile_id, s1, l1_write_addr_in1);
#endif

                tile_id++;
                noc_async_read_barrier();

#ifndef IN0_SHARDED
                cb_push_back(cb_id_in0, onetile);
#endif

#ifndef IN1_SHARDED
                cb_push_back(cb_id_in1, onetile);
#endif
            }
            row_start_tile_id += num_cores_y * block_width;
        }
    } else {
        for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
#ifndef IN0_SHARDED
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(tile_id, s0, l1_write_addr_in0);
#endif

#ifndef IN1_SHARDED
            cb_reserve_back(cb_id_in1, onetile);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read_tile(tile_id, s1, l1_write_addr_in1);
#endif

            noc_async_read_barrier();

#ifndef IN0_SHARDED
            cb_push_back(cb_id_in0, onetile);
#endif

#ifndef IN1_SHARDED
            cb_push_back(cb_id_in1, onetile);
#endif
        }
    }
#endif
}
