// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

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

    CircularBuffer cb_intermed1(intermed_cb_id1);
    CircularBuffer cb_intermed2(intermed_cb_id2);
    CircularBuffer cb_output(output_cb_id);

    constexpr uint32_t onetile = 1;
    constexpr auto dst_args = TensorAccessorArgs<3>();
    const auto s = TensorAccessor(dst_args, dst_addr);

    const uint32_t end_id = start_id + num_output_blocks_w_per_core;

    for (uint32_t block_h_id = 0; block_h_id < out_num_blocks_h; block_h_id++) {
        for (uint32_t i = start_id; i < end_id; ++i) {
            cb_intermed2.reserve_back(onetile);
            uint32_t dst = cb_intermed2.get_write_ptr();

            // Manually unroll copying into destination face 1+2 and 3+4 to avoid conditional inside loop
            for (uint32_t j = 0; j < FACE_WIDTH; ++j) {
                cb_intermed1.wait_front(onetile);
                uint64_t src = get_noc_addr(cb_intermed1.get_read_ptr());

                // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
                noc_async_read(src, dst, FACE_WIDTH_BYTES);
                // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
                noc_async_read(src + FACE_SIZE_BYTES, dst + FACE_SIZE_BYTES, FACE_WIDTH_BYTES);
                noc.async_read_barrier();

                cb_intermed1.pop_front(onetile);

                dst += FACE_WIDTH_BYTES;
            }
            dst += FACE_SIZE_BYTES;

            // Copy face 3/4 into the destination
            for (uint32_t j = 0; j < FACE_WIDTH; ++j) {
                cb_intermed1.wait_front(onetile);
                uint64_t src = get_noc_addr(cb_intermed1.get_read_ptr());

                // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
                noc_async_read(src, dst, FACE_WIDTH_BYTES);
                // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
                noc_async_read(src + FACE_SIZE_BYTES, dst + FACE_SIZE_BYTES, FACE_WIDTH_BYTES);
                noc.async_read_barrier();

                cb_intermed1.pop_front(onetile);

                dst += FACE_WIDTH_BYTES;
            }
            cb_intermed2.push_back(onetile);

            cb_output.wait_front(onetile);
            uint32_t l1_read_addr = cb_output.get_read_ptr();
            noc.async_write(
                CoreLocalMem<uint32_t>(l1_read_addr),
                s,
                get_tile_size(output_cb_id),
                {},
                {.page_id = (block_h_id * out_num_blocks_w) + i});
            noc.async_write_barrier();
            cb_output.pop_front(onetile);
        }
    }
}
