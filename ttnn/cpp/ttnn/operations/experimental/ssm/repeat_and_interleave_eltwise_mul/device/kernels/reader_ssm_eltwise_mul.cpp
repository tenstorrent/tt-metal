// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t in1_num_blocks = get_arg(args::in1_num_blocks);
    uint32_t in1_start_id = get_arg(args::in1_start_id);
    uint32_t in1_num_blocks_h = get_arg(args::in1_num_blocks_h);
    uint32_t in1_num_blocks_w = get_arg(args::in1_num_blocks_w);
    uint32_t in0_num_blocks_w = get_arg(args::in0_num_blocks_w);

    const auto s0 = TensorAccessor(tensor::src0);
    const auto s1 = TensorAccessor(tensor::src1);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_in1_transposed(dfb::in1_transposed);
    DataflowBuffer dfb_in1_bcast_row(dfb::in1_bcast_row);

#ifdef REPEAT_INTERLEAVE_IN1
    // The transposed rows are copied L1->L1 via a NoC loopback to this core's own coordinates.
    UnicastEndpoint local_src;
    const uint32_t local_noc_x = my_x[noc.get_noc_id()];
    const uint32_t local_noc_y = my_y[noc.get_noc_id()];
#endif

    const uint32_t in0_tile_bytes = dfb_in0.get_tile_size();
    const uint32_t in1_tile_bytes = dfb_in1.get_tile_size();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_face = 16;
    constexpr uint32_t bfloat16_one_face_bytes = 512;
    constexpr uint32_t bfloat16_one_row_in_face_bytes = 32;
    constexpr uint32_t in0_blocks_per_in1_block = 32;

    for (uint32_t block_h_id = 0; block_h_id < in1_num_blocks_h; block_h_id++) {
#ifdef REPEAT_IN0
        // in0 only has one tile and read in only once
        dfb_in0.reserve_back(onetile);
        noc.async_read(s0, dfb_in0, in0_tile_bytes, {.page_id = block_h_id, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(onetile);
#endif

        for (uint32_t i = in1_start_id; i < in1_start_id + in1_num_blocks; i++) {
            dfb_in1.reserve_back(onetile);
            noc.async_read(
                s1,
                dfb_in1,
                in1_tile_bytes,
                {.page_id = block_h_id * in1_num_blocks_w + i, .offset_bytes = 0},
                {.offset_bytes = 0});

            noc.async_read_barrier();
            dfb_in1.push_back(onetile);

#ifdef REPEAT_INTERLEAVE_IN1
            dfb_in1_transposed.wait_front(onetile);
            uint32_t in1_transposed_read_addr = dfb_in1_transposed.get_read_ptr();

            // Manually unroll iterating across the tile to eliminate unnecessary conditional checking
            // First + second face
            for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_face; tile_row_id++) {
                dfb_in1_bcast_row.reserve_back(onetile);

#ifndef REPEAT_IN0
                dfb_in0.reserve_back(onetile);
                noc.async_read(
                    s0,
                    dfb_in0,
                    in0_tile_bytes,
                    {.page_id = block_h_id * in0_num_blocks_w + (i * in0_blocks_per_in1_block + tile_row_id),
                     .offset_bytes = 0},
                    {.offset_bytes = 0});
#endif
                noc.async_read(
                    local_src,
                    dfb_in1_bcast_row,
                    bfloat16_one_row_in_face_bytes,
                    {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = in1_transposed_read_addr},
                    {.offset_bytes = 0});
                noc.async_read(
                    local_src,
                    dfb_in1_bcast_row,
                    bfloat16_one_row_in_face_bytes,
                    {.noc_x = local_noc_x,
                     .noc_y = local_noc_y,
                     .addr = in1_transposed_read_addr + bfloat16_one_face_bytes},
                    {.offset_bytes = bfloat16_one_face_bytes});
                noc.async_read_barrier();

#ifndef REPEAT_IN0
                dfb_in0.push_back(onetile);
#endif
                dfb_in1_bcast_row.push_back(onetile);

                in1_transposed_read_addr += bfloat16_one_row_in_face_bytes;
            }

            in1_transposed_read_addr += bfloat16_one_face_bytes;
            // Third + fourth face
            for (uint32_t tile_row_id = num_rows_in_face; tile_row_id < 2 * num_rows_in_face; tile_row_id++) {
                dfb_in1_bcast_row.reserve_back(onetile);

#ifndef REPEAT_IN0
                dfb_in0.reserve_back(onetile);
                noc.async_read(
                    s0,
                    dfb_in0,
                    in0_tile_bytes,
                    {.page_id = block_h_id * 5120 + (i * in0_blocks_per_in1_block + tile_row_id), .offset_bytes = 0},
                    {.offset_bytes = 0});
#endif
                noc.async_read(
                    local_src,
                    dfb_in1_bcast_row,
                    bfloat16_one_row_in_face_bytes,
                    {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = in1_transposed_read_addr},
                    {.offset_bytes = 0});
                noc.async_read(
                    local_src,
                    dfb_in1_bcast_row,
                    bfloat16_one_row_in_face_bytes,
                    {.noc_x = local_noc_x,
                     .noc_y = local_noc_y,
                     .addr = in1_transposed_read_addr + bfloat16_one_face_bytes},
                    {.offset_bytes = bfloat16_one_face_bytes});
                noc.async_read_barrier();

#ifndef REPEAT_IN0
                dfb_in0.push_back(onetile);
#endif
                dfb_in1_bcast_row.push_back(onetile);

                in1_transposed_read_addr += bfloat16_one_row_in_face_bytes;
            }
            dfb_in1_transposed.pop_front(onetile);

#endif
        }
    }
}
