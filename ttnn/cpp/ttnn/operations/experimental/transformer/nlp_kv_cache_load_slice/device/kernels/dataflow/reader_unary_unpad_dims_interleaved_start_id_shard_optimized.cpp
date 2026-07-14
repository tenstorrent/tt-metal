// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

void kernel_main() {
    Noc noc;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_unpadded_tiles_head_dim = get_compile_time_arg_val(1);
    constexpr uint32_t num_unpadded_tiles_seqlen_dim = get_compile_time_arg_val(2);
    constexpr uint32_t num_padded_tiles_seqlen_dim = get_compile_time_arg_val(3);
    constexpr uint32_t num_readers = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();

    constexpr uint32_t cb_id_in0 = 0;

    constexpr uint32_t tile_size = get_tile_size(cb_id_in0);
    const auto s0 = TensorAccessor(src_args, src_addr);

    CircularBuffer cb_in0(cb_id_in0);

    uint32_t src_tile_id = start_id;
    cb_in0.reserve_back(num_tiles);
    uint32_t src_buffer_l1_addr = cb_in0.get_write_ptr();
    uint32_t seqlen_dim_id = 0;

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_size, num_readers>();

    uint32_t num_iterations = num_tiles / num_unpadded_tiles_head_dim;
    uint32_t barrier_count = 0;
    for (uint32_t i = 0; i < num_iterations; i++) {
        // Copy Input
        for (uint32_t j = 0; j < num_unpadded_tiles_head_dim; j++) {
            noc.async_read(s0, CoreLocalMem<uint32_t>(src_buffer_l1_addr), tile_size, {.page_id = src_tile_id}, {});
            src_buffer_l1_addr += tile_size;
            src_tile_id++;
            if (++barrier_count == barrier_threshold) {
                noc.async_read_barrier();
                barrier_count = 0;
            }
        }
        seqlen_dim_id++;
        if (seqlen_dim_id == num_unpadded_tiles_seqlen_dim) {
            seqlen_dim_id = 0;
            src_tile_id += num_padded_tiles_seqlen_dim;
        }
    }

    noc.async_read_barrier();
    cb_in0.push_back(num_tiles);
}
