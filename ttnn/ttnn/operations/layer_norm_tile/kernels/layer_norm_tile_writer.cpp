// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm writer (RISCV_0; NoC chosen per core by the host, see _core_nocs).
//   store_block   : R_b x Wt output tiles in C-tile chunks; one flush + pop per `write_chunks_per_flush`
//                   chunks, one write barrier at kernel end.
//   load_residual : (residual_on_writer, split_reader) the residual stream on this RISC and NoC, kept
//                   `in_depth - 1` blocks ahead of store_block so compute sees the reader's prefetch depth.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "layer_norm_tile_stream.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t io_chunk_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t output_tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t write_chunks_per_flush = get_compile_time_arg_val(5);
    constexpr bool residual_on_writer = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t in_depth = get_compile_time_arg_val(8);
    constexpr uint32_t read_trid_depth = get_compile_time_arg_val(9);
    constexpr uint32_t input_tile_bytes = get_compile_time_arg_val(10);
    constexpr bool half_band = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t image_row_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t image_bands = get_compile_time_arg_val(13);
    constexpr auto output_args = TensorAccessorArgs<14>();
    [[maybe_unused]] constexpr auto residual_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    const uint32_t rotation = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t residual_addr = get_arg_val<uint32_t>(4);

    static_assert(tensor_w_tiles % io_chunk_tiles == 0, "every block pop must be a whole number of chunks");
    static_assert(write_chunks_per_flush >= 1, "flush group must hold at least one chunk");

    DataflowBuffer output_tiles(cb_output_tiles);
    [[maybe_unused]] DataflowBuffer residual_tiles(cb_residual_tiles);
    const auto output_accessor = TensorAccessor(output_args, output_addr, output_tile_bytes);
    [[maybe_unused]] const auto residual_accessor = TensorAccessor(residual_args, residual_addr, input_tile_bytes);
    const Noc noc;

    static_assert(input_tile_bytes == output_tile_bytes, "residual and output share one band layout");
    using Sources =
        layer_norm_tile::RowSources<half_band, tensor_w_tiles, image_row_tiles, image_bands, output_tile_bytes / 2>;
    // Separate cursors: the residual stream runs `in_depth - 1` blocks ahead of the output.
    Sources output_sources(unit_start, unit_count);
    [[maybe_unused]] Sources residual_sources(unit_start, unit_count);
    const uint32_t core_row_tiles = Sources::compute_rows(unit_count);
    constexpr uint32_t row_chunks = tensor_w_tiles / io_chunk_tiles;

    auto block_rows = [&](uint32_t block_row) {
        return (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
    };

    auto load_residual = [&](uint32_t block_row) {
        MaybeDeviceZoneScope("writer_residual_block");
        const uint32_t rows = block_rows(block_row);
        residual_tiles.reserve_back(rows * tensor_w_tiles);
        const uint32_t residual_base = residual_tiles.get_write_ptr();
        layer_norm_tile::RowSource src{};
        layer_norm_tile::read_block_chunks<read_trid_depth>(
            noc,
            rows * row_chunks,
            [&](uint32_t chunk) {
                const uint32_t col = (chunk % row_chunks) * io_chunk_tiles;
                if (col == 0) {
                    src = residual_sources.next();
                }
                layer_norm_tile::read_row_chunk<io_chunk_tiles, input_tile_bytes>(
                    residual_accessor, src, col, residual_base + chunk * io_chunk_tiles * input_tile_bytes, rotation);
            },
            [&]() { residual_tiles.push_back(io_chunk_tiles); });
    };

    auto store_block = [&](uint32_t block_row) {
        MaybeDeviceZoneScope("writer_store_block");
        const uint32_t num_chunks = block_rows(block_row) * row_chunks;
        layer_norm_tile::RowSource src{};
        for (uint32_t group = 0; group < num_chunks; group += write_chunks_per_flush) {
            const uint32_t group_chunks =
                (num_chunks - group) < write_chunks_per_flush ? (num_chunks - group) : write_chunks_per_flush;
            for (uint32_t chunk = 0; chunk < group_chunks; ++chunk) {
                const uint32_t col = ((group + chunk) % row_chunks) * io_chunk_tiles;
                if (col == 0) {
                    src = output_sources.next();
                }
                output_tiles.wait_front((chunk + 1) * io_chunk_tiles);
                layer_norm_tile::write_row_chunk<io_chunk_tiles, output_tile_bytes>(
                    output_accessor,
                    src,
                    col,
                    output_tiles.get_read_ptr() + chunk * io_chunk_tiles * output_tile_bytes,
                    rotation);
            }
            noc_async_writes_flushed();
            output_tiles.pop_front(group_chunks * io_chunk_tiles);
        }
    };

    constexpr uint32_t prefetch_rows = (in_depth - 1) * block_row_tiles;
    if constexpr (residual_on_writer) {
        for (uint32_t block_row = 0; block_row < prefetch_rows && block_row < core_row_tiles;
             block_row += block_row_tiles) {
            load_residual(block_row);
        }
    }
    for (uint32_t block_row = 0; block_row < core_row_tiles; block_row += block_row_tiles) {
        if constexpr (residual_on_writer) {
            if (block_row + prefetch_rows < core_row_tiles) {
                load_residual(block_row + prefetch_rows);
            }
        }
        store_block(block_row);
    }
    {
        MaybeDeviceZoneScope("writer_final_barrier");
        noc_async_write_barrier();
    }
}
