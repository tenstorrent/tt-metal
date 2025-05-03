// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_unpadded_tiles_head_dim = get_compile_time_arg_val(2);
    constexpr uint32_t num_unpadded_tiles_seqlen_dim = get_compile_time_arg_val(3);
    constexpr uint32_t num_padded_tiles_seqlen_dim = get_compile_time_arg_val(4);
    constexpr uint32_t num_readers = get_compile_time_arg_val(5);

    constexpr uint32_t cb_id_in0 = 0;

    constexpr uint32_t tile_size = get_tile_size(cb_id_in0);
    constexpr DataFormat data_format = get_dataformat(cb_id_in0);

    // In and out are assumed to be same dataformat
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src_addr, .page_size = tile_size, .data_format = data_format};

    uint32_t src_tile_id = start_id;
    cb_reserve_back(cb_id_in0, num_tiles);
    uint32_t src_buffer_l1_addr = get_write_ptr(cb_id_in0);
    uint32_t seqlen_dim_id = 0;

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_size, num_readers>();

    uint32_t num_iterations = num_tiles / num_unpadded_tiles_head_dim;
    uint32_t barrier_count = 0;
    for (uint32_t i = 0; i < num_iterations; i++) {
        // Copy Input
        for (uint32_t j = 0; j < num_unpadded_tiles_head_dim; j++) {
            noc_async_read_tile(src_tile_id, s0, src_buffer_l1_addr);
            src_buffer_l1_addr += tile_size;
            src_tile_id++;
            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
        seqlen_dim_id++;
        if (seqlen_dim_id == num_unpadded_tiles_seqlen_dim) {
            seqlen_dim_id = 0;
            src_tile_id += num_padded_tiles_seqlen_dim;
        }
    }

    noc_async_read_barrier();
    cb_push_back(cb_id_in0, num_tiles);
}
