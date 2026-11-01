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
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_affine_scratch = get_named_compile_time_arg_val("affine_scratch_cb");
    constexpr uint32_t affine_group_cols = get_named_compile_time_arg_val("affine_group_cols");
    constexpr bool beta_on_writer = get_named_compile_time_arg_val("affine_beta_on_writer") != 0;
    constexpr bool affine_is_tile = get_named_compile_time_arg_val("affine_is_tile") != 0;
    constexpr uint32_t affine_page_elems = get_named_compile_time_arg_val("affine_page_elems");
    constexpr uint32_t affine_page_bytes = get_named_compile_time_arg_val("affine_page_bytes");
    constexpr uint32_t affine_elem_bytes = get_named_compile_time_arg_val("affine_elem_bytes");
    constexpr bool fold = get_named_compile_time_arg_val("fold_lone") != 0;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    const uint32_t rotation = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t residual_addr = get_arg_val<uint32_t>(4);
    [[maybe_unused]] const uint32_t beta_addr = get_arg_val<uint32_t>(5);
    const bool affine_injector = get_arg_val<uint32_t>(6) != 0;

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
    // lone_band_fold: the lone row's chunks carry C / 2 folded tiles; its page slot is padded to Wt pages.
    const uint32_t lone_row = fold ? output_sources.lone_row() : Sources::NO_LONE_ROW;
    constexpr uint32_t folded_chunk_tiles = io_chunk_tiles / 2;
    constexpr uint32_t fold_pad_tiles = tensor_w_tiles - tensor_w_tiles / 2;

    auto block_rows = [&](uint32_t block_row) {
        return (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
    };

    auto load_residual = [&](uint32_t block_row) {
        MaybeDeviceZoneScope("writer_residual_block");
        const uint32_t rows = block_rows(block_row);
        residual_tiles.reserve_back(rows * tensor_w_tiles);
        const uint32_t residual_base = residual_tiles.get_write_ptr();
        layer_norm_tile::RowSource src{};
        uint32_t retired = 0;
        layer_norm_tile::read_block_chunks<read_trid_depth>(
            noc,
            rows * row_chunks,
            [&](uint32_t chunk) {
                const uint32_t col = (chunk % row_chunks) * io_chunk_tiles;
                if (col == 0) {
                    src = residual_sources.next();
                }
                const uint32_t row = chunk / row_chunks;
                if (block_row + row == lone_row) {
                    layer_norm_tile::read_folded_chunk<io_chunk_tiles, input_tile_bytes>(
                        residual_accessor,
                        src,
                        col,
                        residual_base + (row * tensor_w_tiles + col / 2) * input_tile_bytes);
                } else {
                    layer_norm_tile::read_row_chunk<io_chunk_tiles, input_tile_bytes>(
                        residual_accessor,
                        src,
                        col,
                        residual_base + chunk * io_chunk_tiles * input_tile_bytes,
                        rotation);
                }
            },
            [&]() {
                uint32_t tiles = io_chunk_tiles;
                if (block_row + retired / row_chunks == lone_row) {
                    tiles = folded_chunk_tiles + (retired % row_chunks == row_chunks - 1 ? fold_pad_tiles : 0);
                }
                ++retired;
                residual_tiles.push_back(tiles);
            });
    };

    auto store_block = [&](uint32_t block_row) {
        MaybeDeviceZoneScope("writer_store_block");
        const uint32_t num_chunks = block_rows(block_row) * row_chunks;
        layer_norm_tile::RowSource src{};
        for (uint32_t group = 0; group < num_chunks; group += write_chunks_per_flush) {
            const uint32_t group_chunks =
                (num_chunks - group) < write_chunks_per_flush ? (num_chunks - group) : write_chunks_per_flush;
            uint32_t group_tiles = 0;
            for (uint32_t chunk = 0; chunk < group_chunks; ++chunk) {
                const uint32_t col = ((group + chunk) % row_chunks) * io_chunk_tiles;
                if (col == 0) {
                    src = output_sources.next();
                }
                const uint32_t l1 = output_tiles.get_read_ptr() + group_tiles * output_tile_bytes;
                if (block_row + (group + chunk) / row_chunks == lone_row) {
                    output_tiles.wait_front(group_tiles + folded_chunk_tiles);
                    layer_norm_tile::write_folded_chunk<io_chunk_tiles, output_tile_bytes>(
                        output_accessor, src, col, l1);
                    group_tiles += folded_chunk_tiles;
                    if (col + io_chunk_tiles == tensor_w_tiles) {
                        // The row's unwritten pad pages, pushed by compute right after the row.
                        group_tiles += fold_pad_tiles;
                        output_tiles.wait_front(group_tiles);
                    }
                } else {
                    output_tiles.wait_front(group_tiles + io_chunk_tiles);
                    layer_norm_tile::write_row_chunk<io_chunk_tiles, output_tile_bytes>(
                        output_accessor, src, col, l1, rotation);
                    group_tiles += io_chunk_tiles;
                }
            }
            noc_async_writes_flushed();
            output_tiles.pop_front(group_tiles);
        }
    };

    // Injector only (beta_on_writer): beta into the scratch block's second half, handed to the reader by the push.
    // Runs before any other read of this RISC, so the plain read barrier (software-counted) is exact; a trid
    // barrier right after a single read can pass before the NIU counts it outstanding (seen with one group).
    auto load_beta = [&]() {
        using Affine = layer_norm_tile::
            AffineBlock<affine_is_tile, tensor_w_tiles, affine_group_cols, affine_page_elems, affine_elem_bytes>;
        DataflowBuffer scratch(cb_affine_scratch);
        scratch.reserve_back(1);
        const uint32_t beta_block = scratch.get_write_ptr() + Affine::tensor_bytes;
        Affine::issue_reads(TensorAccessor(beta_args, beta_addr, affine_page_bytes), beta_block);
        noc_async_read_barrier();
        if constexpr (!affine_is_tile) {
            Affine::shift_face1_rows(beta_block);
            noc_async_read_barrier();
        }
        scratch.push_back(1);
    };

    constexpr uint32_t prefetch_rows = (in_depth - 1) * block_row_tiles;
    // Before the residual prefetch: the multicast waits on beta, and the injector's first residual block can
    // land late under all-core load.
    if constexpr (beta_on_writer) {
        if (affine_injector) {
            load_beta();
        }
    }
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
