// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t in1_num_blocks = get_arg_val<uint32_t>(2);
    uint32_t in1_start_id = get_arg_val<uint32_t>(3);
    uint32_t in1_num_blocks_h = get_arg_val<uint32_t>(4);
    uint32_t in1_num_blocks_w = get_arg_val<uint32_t>(5);
    uint32_t in0_num_blocks_w = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in1_transposed = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in1_bcast_row = get_compile_time_arg_val(3);

    constexpr auto src0_args = TensorAccessorArgs<4>();
    const auto s0 = TensorAccessor(src0_args, src0_addr);

    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(src1_args, src1_addr);

    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_in1(cb_id_in1);
    CircularBuffer cb_in1_transposed_buf(cb_in1_transposed);
    CircularBuffer cb_in1_bcast_row_buf(cb_in1_bcast_row);

#ifdef REPEAT_INTERLEAVE_IN1
    // The transposed rows are copied L1->L1 via a NoC loopback to this core's own coordinates.
    UnicastEndpoint local_src;
    const uint32_t local_noc_x = my_x[noc.get_noc_id()];
    const uint32_t local_noc_y = my_y[noc.get_noc_id()];
#endif

    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_face = 16;
    constexpr uint32_t bfloat16_one_face_bytes = 512;
    constexpr uint32_t bfloat16_one_row_in_face_bytes = 32;
    constexpr uint32_t in0_blocks_per_in1_block = 32;

    for (uint32_t block_h_id = 0; block_h_id < in1_num_blocks_h; block_h_id++) {
#ifdef REPEAT_IN0
        // in0 only has one tile and read in only once
        cb_in0.reserve_back(onetile);
        noc.async_read(s0, cb_in0, in0_tile_bytes, {.page_id = block_h_id, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(onetile);
#endif

        for (uint32_t i = in1_start_id; i < in1_start_id + in1_num_blocks; i++) {
            cb_in1.reserve_back(onetile);
            noc.async_read(
                s1,
                cb_in1,
                in1_tile_bytes,
                {.page_id = block_h_id * in1_num_blocks_w + i, .offset_bytes = 0},
                {.offset_bytes = 0});

            noc.async_read_barrier();
            cb_in1.push_back(onetile);

#ifdef REPEAT_INTERLEAVE_IN1
            cb_in1_transposed_buf.wait_front(onetile);
            uint32_t cb_in1_transposed_read_addr = cb_in1_transposed_buf.get_read_ptr();

            // Manually unroll iterating across the tile to eliminate unnecessary conditional checking
            // First + second face
            for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_face; tile_row_id++) {
                cb_in1_bcast_row_buf.reserve_back(onetile);

#ifndef REPEAT_IN0
                cb_in0.reserve_back(onetile);
                noc.async_read(
                    s0,
                    cb_in0,
                    in0_tile_bytes,
                    {.page_id = block_h_id * in0_num_blocks_w + (i * in0_blocks_per_in1_block + tile_row_id),
                     .offset_bytes = 0},
                    {.offset_bytes = 0});
#endif
                noc.async_read(
                    local_src,
                    cb_in1_bcast_row_buf,
                    bfloat16_one_row_in_face_bytes,
                    {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = cb_in1_transposed_read_addr},
                    {.offset_bytes = 0});
                noc.async_read(
                    local_src,
                    cb_in1_bcast_row_buf,
                    bfloat16_one_row_in_face_bytes,
                    {.noc_x = local_noc_x,
                     .noc_y = local_noc_y,
                     .addr = cb_in1_transposed_read_addr + bfloat16_one_face_bytes},
                    {.offset_bytes = bfloat16_one_face_bytes});
                noc.async_read_barrier();

#ifndef REPEAT_IN0
                cb_in0.push_back(onetile);
#endif
                cb_in1_bcast_row_buf.push_back(onetile);

                cb_in1_transposed_read_addr += bfloat16_one_row_in_face_bytes;
            }

            cb_in1_transposed_read_addr += bfloat16_one_face_bytes;
            // Third + fourth face
            for (uint32_t tile_row_id = num_rows_in_face; tile_row_id < 2 * num_rows_in_face; tile_row_id++) {
                cb_in1_bcast_row_buf.reserve_back(onetile);

#ifndef REPEAT_IN0
                cb_in0.reserve_back(onetile);
                noc.async_read(
                    s0,
                    cb_in0,
                    in0_tile_bytes,
                    {.page_id = block_h_id * 5120 + (i * in0_blocks_per_in1_block + tile_row_id), .offset_bytes = 0},
                    {.offset_bytes = 0});
#endif
                noc.async_read(
                    local_src,
                    cb_in1_bcast_row_buf,
                    bfloat16_one_row_in_face_bytes,
                    {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = cb_in1_transposed_read_addr},
                    {.offset_bytes = 0});
                noc.async_read(
                    local_src,
                    cb_in1_bcast_row_buf,
                    bfloat16_one_row_in_face_bytes,
                    {.noc_x = local_noc_x,
                     .noc_y = local_noc_y,
                     .addr = cb_in1_transposed_read_addr + bfloat16_one_face_bytes},
                    {.offset_bytes = bfloat16_one_face_bytes});
                noc.async_read_barrier();

#ifndef REPEAT_IN0
                cb_in0.push_back(onetile);
#endif
                cb_in1_bcast_row_buf.push_back(onetile);

                cb_in1_transposed_read_addr += bfloat16_one_row_in_face_bytes;
            }
            cb_in1_transposed_buf.pop_front(onetile);

#endif
        }
    }
}
