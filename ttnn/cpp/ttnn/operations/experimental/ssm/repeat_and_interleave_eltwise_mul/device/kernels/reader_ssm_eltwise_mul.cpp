// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
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
    constexpr bool src0_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(5) == 1;

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

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_face = 16;
    constexpr uint32_t bfloat16_one_face_bytes = 512;
    constexpr uint32_t bfloat16_one_row_in_face_bytes = 32;
    constexpr uint32_t in0_blocks_per_in1_block = 32;

    for (uint32_t block_h_id = 0; block_h_id < in1_num_blocks_h; block_h_id++) {
#ifdef REPEAT_IN0
        // in0 only has one tile and read in only once
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(block_h_id, s0, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
#endif

        for (uint32_t i = in1_start_id; i < in1_start_id + in1_num_blocks; i++) {
            cb_reserve_back(cb_id_in1, onetile);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read_tile(block_h_id * in1_num_blocks_w + i, s1, l1_write_addr_in1);

            noc_async_read_barrier();
            cb_push_back(cb_id_in1, onetile);

#ifdef REPEAT_INTERLEAVE_IN1
            cb_wait_front(cb_in1_transposed, onetile);
            uint64_t cb_in1_transposed_read_ptr = get_noc_addr(get_read_ptr(cb_in1_transposed));

            // Manually unroll iterating across the tile to eliminate unncessary conditional checking
            // First + second face
            for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_face; tile_row_id++) {
                cb_reserve_back(cb_in1_bcast_row, onetile);
                uint32_t cb_in1_bcast_row_write_ptr = get_write_ptr(cb_in1_bcast_row);

#ifndef REPEAT_IN0
                cb_reserve_back(cb_id_in0, onetile);
                l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(block_h_id * in0_num_blocks_w + (i * in0_blocks_per_in1_block + tile_row_id),
                                    s0,
                                    l1_write_addr_in0);
#endif
                noc_async_read(cb_in1_transposed_read_ptr, cb_in1_bcast_row_write_ptr, bfloat16_one_row_in_face_bytes);
                noc_async_read(cb_in1_transposed_read_ptr + bfloat16_one_face_bytes,
                               cb_in1_bcast_row_write_ptr + bfloat16_one_face_bytes,
                               bfloat16_one_row_in_face_bytes);
                noc_async_read_barrier();

#ifndef REPEAT_IN0
                cb_push_back(cb_id_in0, onetile);
#endif
                cb_push_back(cb_in1_bcast_row, onetile);

                cb_in1_transposed_read_ptr += bfloat16_one_row_in_face_bytes;
            }

            cb_in1_transposed_read_ptr += bfloat16_one_face_bytes;
            // Third + fourth face
            for (uint32_t tile_row_id = num_rows_in_face; tile_row_id < 2 * num_rows_in_face; tile_row_id++) {
                cb_reserve_back(cb_in1_bcast_row, onetile);
                uint32_t cb_in1_bcast_row_write_ptr = get_write_ptr(cb_in1_bcast_row);

#ifndef REPEAT_IN0
                cb_reserve_back(cb_id_in0, onetile);
                l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(
                    block_h_id * 5120 + (i * in0_blocks_per_in1_block + tile_row_id), s0, l1_write_addr_in0);
#endif
                noc_async_read(cb_in1_transposed_read_ptr, cb_in1_bcast_row_write_ptr, bfloat16_one_row_in_face_bytes);
                noc_async_read(cb_in1_transposed_read_ptr + bfloat16_one_face_bytes,
                               cb_in1_bcast_row_write_ptr + bfloat16_one_face_bytes,
                               bfloat16_one_row_in_face_bytes);
                noc_async_read_barrier();

#ifndef REPEAT_IN0
                cb_push_back(cb_id_in0, onetile);
#endif
                cb_push_back(cb_in1_bcast_row, onetile);

                cb_in1_transposed_read_ptr += bfloat16_one_row_in_face_bytes;
            }
            cb_pop_front(cb_in1_transposed, onetile);

#endif
        }
    }
}
