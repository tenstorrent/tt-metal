// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    constexpr uint32_t cb_id_in0            =  get_compile_time_arg_val(0);
    constexpr uint32_t num_dims             =  get_compile_time_arg_val(1);
    const uint32_t src_addr                 = get_common_arg_val<uint32_t>(0);

    volatile tt_l1_ptr uint32_t * num_unpadded_tiles = (volatile tt_l1_ptr uint32_t*)(get_common_arg_addr(1));
    volatile tt_l1_ptr uint32_t * num_padded_tiles = num_unpadded_tiles + num_dims;

    const uint32_t start_id                 = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles                = get_arg_val<uint32_t>(1);

    tt_l1_ptr uint32_t * id_per_dim = (tt_l1_ptr uint32_t*)(get_arg_addr(2));

    constexpr uint32_t tile_size = get_tile_size(cb_id_in0);
    constexpr DataFormat data_format = get_dataformat(cb_id_in0);

    constexpr bool src0_is_dram                           = get_compile_time_arg_val(2) == 1;
    // In and out are assumed to be same dataformat
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = tile_size,
        .data_format = data_format
    };

    uint32_t src_tile_id = start_id;

    for(uint32_t i = 0; i < num_tiles; ++i) {
        // Copy Input
        cb_reserve_back(cb_id_in0, 1);
        uint32_t src_buffer_l1_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(src_tile_id, s0, src_buffer_l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
        src_tile_id++;
        for(uint32_t j = 0; j < num_dims; ++j) {
            id_per_dim[j]++;
            if (id_per_dim[j] == num_unpadded_tiles[j]) {
                id_per_dim[j] = 0;
                src_tile_id += num_padded_tiles[j];
            } else {
                break;
            }
        }
    }
}
