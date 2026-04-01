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

    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_id_in0);
    experimental::CircularBuffer cb1(cb_id_in1);

#ifdef IN0_SHARDED
    cb0.reserve_back(num_tiles);
    cb0.push_back(num_tiles);
#else
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const auto s0 = TensorAccessor(src0_args, src0_addr, src0_tile_bytes);
#endif
#ifdef IN1_SHARDED
    cb1.reserve_back(num_tiles);
    cb1.push_back(num_tiles);
#else
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
                cb0.reserve_back(onetile);
                noc.async_read(s0, cb0, src0_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
#endif

#ifndef IN1_SHARDED
                cb1.reserve_back(onetile);
                noc.async_read(s1, cb1, src1_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
#endif

                tile_id++;
                noc.async_read_barrier();

#ifndef IN0_SHARDED
                cb0.push_back(onetile);
#endif

#ifndef IN1_SHARDED
                cb1.push_back(onetile);
#endif
            }
            row_start_tile_id += num_cores_y * block_width;
        }
    } else {
        for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
#ifndef IN0_SHARDED
            cb0.reserve_back(onetile);
            noc.async_read(s0, cb0, src0_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
#endif

#ifndef IN1_SHARDED
            cb1.reserve_back(onetile);
            noc.async_read(s1, cb1, src1_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
#endif

            noc.async_read_barrier();

#ifndef IN0_SHARDED
            cb0.push_back(onetile);
#endif

#ifndef IN1_SHARDED
            cb1.push_back(onetile);
#endif
        }
    }
#endif
}
