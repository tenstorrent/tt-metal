// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm writer (RISCV_0, NoC1). store_block: R_b x Wt output tiles in C-tile chunks,
// one flush per chunk, one write barrier at kernel end.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t io_chunk_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t output_tile_bytes = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(2);

    static_assert(tensor_w_tiles % io_chunk_tiles == 0, "every block pop must be a whole number of chunks");

    DataflowBuffer output_tiles(cb_output_tiles);
    const auto output_accessor = TensorAccessor(output_args, output_addr, output_tile_bytes);

    for (uint32_t block_row = 0; block_row < core_row_tiles; block_row += block_row_tiles) {
        const uint32_t rows =
            (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
        const uint32_t first_page = (row_tile_start + block_row) * tensor_w_tiles;
        const uint32_t block_tiles = rows * tensor_w_tiles;

        for (uint32_t chunk_start = 0; chunk_start < block_tiles; chunk_start += io_chunk_tiles) {
            output_tiles.wait_front(io_chunk_tiles);
            uint32_t l1_read_addr = output_tiles.get_read_ptr();
            for (uint32_t tile = 0; tile < io_chunk_tiles; ++tile) {
                noc_async_write(
                    l1_read_addr, output_accessor.get_noc_addr(first_page + chunk_start + tile), output_tile_bytes);
                l1_read_addr += output_tile_bytes;
            }
            noc_async_writes_flushed();
            output_tiles.pop_front(io_chunk_tiles);
        }
    }
    noc_async_write_barrier();
}
