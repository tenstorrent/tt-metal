// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t DATA_FORMAT_BYTES = 2;
    constexpr uint32_t FACE_WIDTH = 16;
    constexpr uint32_t FACE_WIDTH_BYTES = FACE_WIDTH * DATA_FORMAT_BYTES;
    constexpr uint32_t FACE_SIZE = 16 * 16;
    constexpr uint32_t FACE_SIZE_BYTES = FACE_SIZE * DATA_FORMAT_BYTES;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_output_blocks_w_per_core = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t out_num_blocks_h = get_arg_val<uint32_t>(3);
    uint32_t out_num_blocks_w = get_arg_val<uint32_t>(4);

    constexpr uint32_t intermed_cb_id1 = get_compile_time_arg_val(0);
    constexpr uint32_t intermed_cb_id2 = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(2);
    constexpr bool output_is_dram = get_compile_time_arg_val(3) == 1;

    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(output_cb_id);
    const DataFormat data_format = get_dataformat(output_cb_id);

    const InterleavedAddrGenFast<output_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    const uint32_t end_id = start_id + num_output_blocks_w_per_core;

    for (uint32_t block_h_id = 0; block_h_id < out_num_blocks_h; block_h_id++) {
        for (uint32_t i = start_id; i < end_id; ++i) {
            cb_reserve_back(intermed_cb_id2, onetile);
            uint32_t dst = get_write_ptr(intermed_cb_id2);

            // Manually unroll copying into destination face 1+2 and 3+4 to avoid conditional inside loop
            for (uint32_t j = 0; j < FACE_WIDTH; ++j) {
                cb_wait_front(intermed_cb_id1, onetile);
                uint64_t src = get_noc_addr(get_read_ptr(intermed_cb_id1));

                noc_async_read(src, dst, FACE_WIDTH_BYTES);
                noc_async_read(src + FACE_SIZE_BYTES, dst + FACE_SIZE_BYTES, FACE_WIDTH_BYTES);
                noc_async_read_barrier();

                cb_pop_front(intermed_cb_id1, onetile);

                dst += FACE_WIDTH_BYTES;
            }
            dst += FACE_SIZE_BYTES;

            // Copy face 3/4 into the destination
            for (uint32_t j = 0; j < FACE_WIDTH; ++j) {
                cb_wait_front(intermed_cb_id1, onetile);
                uint64_t src = get_noc_addr(get_read_ptr(intermed_cb_id1));

                noc_async_read(src, dst, FACE_WIDTH_BYTES);
                noc_async_read(src + FACE_SIZE_BYTES, dst + FACE_SIZE_BYTES, FACE_WIDTH_BYTES);
                noc_async_read_barrier();

                cb_pop_front(intermed_cb_id1, onetile);

                dst += FACE_WIDTH_BYTES;
            }
            cb_push_back(intermed_cb_id2, onetile);

            cb_wait_front(output_cb_id, onetile);
            uint32_t l1_read_addr = get_read_ptr(output_cb_id);
            noc_async_write_tile((block_h_id * out_num_blocks_w) + i, s, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(output_cb_id, onetile);
        }
    }
}
