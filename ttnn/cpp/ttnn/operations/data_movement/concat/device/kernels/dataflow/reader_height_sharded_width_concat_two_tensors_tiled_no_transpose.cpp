// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Grouped width-concat for two height-sharded, TILE-layout inputs whose tiles have a single row of
// faces (i.e. face_height == tile_height, which holds for all width-32 tiny tiles: 8x32, 16x32, ...).
//
// For such tiles a column slice that is a multiple of the face width is contiguous in L1, so the
// grouped concat is just an interleaved sequence of contiguous copies and needs no tile transpose
// (unlike the 32x32 path, where a 2x2 face layout makes column slices non-contiguous). Each output
// stick (tile-row) is built as [in0 group 0][in1 group 0][in0 group 1][in1 group 1]... directly from
// the resident input shards into the resident output shard.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t input0_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input1_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(2);

    constexpr uint32_t input0_num_tiles_height = get_compile_time_arg_val(3);
    constexpr uint32_t input0_num_tiles_width = get_compile_time_arg_val(4);
    constexpr uint32_t input1_num_tiles_width = get_compile_time_arg_val(5);

    constexpr uint32_t tile_size = get_compile_time_arg_val(6);
    constexpr uint32_t groups = get_compile_time_arg_val(7);

    constexpr uint32_t output_num_tiles_width = input0_num_tiles_width + input1_num_tiles_width;
    constexpr uint32_t input0_row_bytes = tile_size * input0_num_tiles_width;
    constexpr uint32_t input1_row_bytes = tile_size * input1_num_tiles_width;
    constexpr uint32_t output_row_bytes = tile_size * output_num_tiles_width;
    constexpr uint32_t input0_group_stride = input0_row_bytes / groups;
    constexpr uint32_t input1_group_stride = input1_row_bytes / groups;

    Noc noc;
    CircularBuffer input0_cb(input0_cb_id);
    CircularBuffer input1_cb(input1_cb_id);
    CircularBuffer output_cb(output_cb_id);

    const uint32_t in0_base = input0_cb.get_read_ptr();
    const uint32_t in1_base = input1_cb.get_read_ptr();

    output_cb.reserve_back(output_num_tiles_width * input0_num_tiles_height);
    const uint32_t out_base = output_cb.get_write_ptr();

    const uint32_t noc_x = (uint32_t)my_x[noc.get_noc_id()];
    const uint32_t noc_y = (uint32_t)my_y[noc.get_noc_id()];

    for (uint32_t i = 0; i < input0_num_tiles_height; i++) {
        uint32_t in0_read_addr = in0_base + i * input0_row_bytes;
        uint32_t in1_read_addr = in1_base + i * input1_row_bytes;
        uint32_t l1_write_addr = out_base + i * output_row_bytes;
        for (uint32_t g = 0; g < groups; g++) {
            {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(
                    UnicastEndpoint{},
                    dst,
                    input0_group_stride,
                    {.noc_x = noc_x, .noc_y = noc_y, .addr = in0_read_addr},
                    {.offset_bytes = 0});
                in0_read_addr += input0_group_stride;
                l1_write_addr += input0_group_stride;
            }
            {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(
                    UnicastEndpoint{},
                    dst,
                    input1_group_stride,
                    {.noc_x = noc_x, .noc_y = noc_y, .addr = in1_read_addr},
                    {.offset_bytes = 0});
                in1_read_addr += input1_group_stride;
                l1_write_addr += input1_group_stride;
            }
        }
    }

    noc.async_read_barrier();
    output_cb.push_back(output_num_tiles_width * input0_num_tiles_height);
}
