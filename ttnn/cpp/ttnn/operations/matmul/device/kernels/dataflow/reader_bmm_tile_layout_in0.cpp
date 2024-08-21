// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    // in0/in1 common args
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // batch args
    const uint32_t batch = get_arg_val<uint32_t>(1);
    const uint32_t bcast_B = get_arg_val<uint32_t>(2);
    const uint32_t MtKt = get_arg_val<uint32_t>(3);  // if 0

    // in0 tensor args
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(4);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(5);
    const uint32_t in0_tensor_stride_w = get_arg_val<uint32_t>(6);
    const uint32_t in0_tensor_stride_h = get_arg_val<uint32_t>(7);
    const uint32_t in0_tensor_next_block_stride = get_arg_val<uint32_t>(8);

    // in0 block args
    const uint32_t in0_block_w = get_arg_val<uint32_t>(9);
    const uint32_t in0_block_h = get_arg_val<uint32_t>(10);
    const uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(11);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr bool in0_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_in0 = 0;

#ifdef IN0_SHARDED
    const uint32_t in0_num_tiles = batch * num_blocks * in0_block_h * in0_block_w;
    cb_reserve_back(cb_id_in0, in0_num_tiles);
    cb_push_back(cb_id_in0, in0_num_tiles);
#else

    const uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);

    uint32_t l1_write_addr_in0;

    const InterleavedAddrGenFast<in0_is_dram> s0 = {
        .bank_base_address = in0_tensor_addr, .page_size = in0_single_tile_size_bytes, .data_format = in0_data_format};

    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);

            l1_write_addr_in0 = get_write_ptr(cb_id_in0);

            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; ++h) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; ++w) {
                    noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);

                    l1_write_addr_in0 += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            noc_async_read_barrier();

            cb_push_back(cb_id_in0, in0_block_num_tiles);
        }
        in0_tensor_start_tile_id += MtKt;
    }
#endif
}
