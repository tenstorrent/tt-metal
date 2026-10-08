// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the modulated deformable im2col: the channel tiles of (block,
// kernel position kk) are tile columns [kk * C/32, (kk + 1) * C/32) of tile row
// `block` in the (rows, K*K*C) output.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t output_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t channel_tiles = get_compile_time_arg_val(1);
constexpr uint32_t kernel_positions = get_compile_time_arg_val(2);
constexpr uint32_t tile_nbytes = get_compile_time_arg_val(3);
constexpr auto output_args = TensorAccessorArgs<4>();

constexpr uint32_t TILES_PER_ROW = kernel_positions * channel_tiles;

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);

    const auto output_acc = TensorAccessor(output_args, output_addr, tile_nbytes);
    Noc noc;
    CircularBuffer output_cb(output_cb_index);

    for (uint32_t block = start_block; block < start_block + num_blocks; ++block) {
        for (uint32_t kk = 0; kk < kernel_positions; ++kk) {
            output_cb.wait_front(channel_tiles);
            const uint32_t tiles_l1 = output_cb.get_read_ptr();
            const uint32_t first_tile = block * TILES_PER_ROW + kk * channel_tiles;
            for (uint32_t t = 0; t < channel_tiles; ++t) {
                CoreLocalMem<uint32_t> src(tiles_l1 + t * tile_nbytes);
                noc.async_write(src, output_acc, tile_nbytes, {.offset_bytes = 0}, {.page_id = first_tile + t});
            }
            noc.async_write_barrier();
            output_cb.pop_front(channel_tiles);
        }
    }
}
