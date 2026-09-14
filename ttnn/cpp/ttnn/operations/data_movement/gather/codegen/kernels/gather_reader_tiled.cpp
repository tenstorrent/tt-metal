// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather reader -- TILED mode (NCRISC).
//
// High-parallelism decomposition for gathers with FEW tile-rows (small Ht) but
// many index columns (Wt_index > 1). The row-buffered kernel splits work by Ht,
// so a 2-tile-row gather only ever lights up 2 cores. This kernel instead splits
// by OUTPUT TILE: total_work = Ht * Wt_index, so a 2x64 gather fans out to 128
// cores.
//
// Each core owns a CONTIGUOUS range of output-tile ordinals [start, start+n).
// Ordinal g maps to (h, w): h = g / Wt_index, w = g % Wt_index. Because index/
// output tiles are laid out as tile(h,w) -> DRAM ordinal h*Wt_index + w, and
// input tiles as tile(h,c) -> h*Wt_input + c, a core's range is processed as a
// sequence of (row h, contiguous column sub-range [w0, w0+w_count)) chunks.
//
// RESIDENCY INVARIANT (the thing two prior rewrites got wrong): to gather ANY
// output tile in row h the reader needs the ENTIRE input row h resident, because
// a gather index can point at any input column. This kernel satisfies it BY
// CONSTRUCTION: the paired writer (gather_writer_tiled.cpp) walks the identical
// [start, n) range with the identical row-chunking arithmetic and, on every row
// change, loads the FULL input row h (all Wt_input tiles) into THIS core's
// cb_input before the reader consumes it. The reader's
// input_buffer.wait_front(Wt_input) blocks until exactly those Wt_input tiles
// are present. No core ever reads a cb_input slot it did not itself populate
// for the current row.

#include "codegen_gather_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include <cstdint>

void kernel_main() {
    // Runtime args: per-core contiguous output-tile range [start, start+n)
    const uint32_t index_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    const uint32_t tile_width = get_arg_val<uint32_t>(3);
    const uint32_t tile_height = get_arg_val<uint32_t>(4);

    // Compile-time args. No core count: this factory hands each core a contiguous output-tile
    // range, so no row is ever derived from a core ordinal.
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output = get_compile_time_arg_val(2);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(3);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(4);
    constexpr uint32_t index_valid_h_last = get_compile_time_arg_val(5);
    constexpr uint32_t index_valid_w_last = get_compile_time_arg_val(6);
    constexpr uint32_t index_ht_per_batch = get_compile_time_arg_val(7);
    constexpr auto index_ta_args = TensorAccessorArgs<8>();

    constexpr uint32_t one_tile = 1;
    const uint32_t tile_width_mask = tile_width - 1;

    constexpr uint32_t index_tile_bytes = get_tile_size(cb_index);
    const auto index_accessor = TensorAccessor(index_ta_args, index_addr, index_tile_bytes);

    Noc noc;
    CircularBuffer input_buffer(cb_input);
    CircularBuffer index_buffer(cb_index);
    CircularBuffer output_buffer(cb_output);

    constexpr uint32_t input_df_size = get_tile_size(cb_input) / get_tile_hw(cb_input);
    constexpr uint32_t index_df_size = index_tile_bytes / get_tile_hw(cb_index);
    constexpr uint32_t output_df_size = get_tile_size(cb_output) / get_tile_hw(cb_output);

    constexpr uint32_t face_size = 16;
    constexpr uint32_t FACE_SIZE_MASK = face_size - 1;
    constexpr uint32_t tile_faces = 2;

    uint32_t g = start;
    uint32_t remaining = n;
    while (remaining > 0) {
        const uint32_t h = g / Wt_index;
        const uint32_t w0 = g % Wt_index;
        const uint32_t h_in_batch = h % index_ht_per_batch;
        const uint32_t valid_h = (h_in_batch == index_ht_per_batch - 1) ? index_valid_h_last : tile_height;
        // Columns of row h that this core owns, clamped to the row boundary.
        uint32_t w_count = Wt_index - w0;
        if (w_count > remaining) {
            w_count = remaining;
        }

        // Wait for the writer to make the FULL input row h resident in cb_input.
        // These Wt_input tiles stay resident for every column w in this chunk.
        input_buffer.wait_front(Wt_input);
        const uint32_t input_l1 = input_buffer.get_read_ptr();

        for (uint32_t jw = 0; jw < w_count; jw++) {
            const uint32_t w = w0 + jw;
            const uint32_t valid_w = (w == Wt_index - 1) ? index_valid_w_last : tile_width;
            // Index tile (h, w0+jw) lives at DRAM ordinal g + jw.
            index_buffer.reserve_back(one_tile);
            noc.async_read(
                index_accessor,
                index_buffer,
                index_tile_bytes,
                {.page_id = g + jw, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            index_buffer.push_back(one_tile);

            index_buffer.wait_front(one_tile);
            output_buffer.reserve_back(one_tile);

            const uint32_t index_l1 = index_buffer.get_read_ptr();
            const uint32_t output_l1 = output_buffer.get_write_ptr();

            // Element-level gather over the index tile (IDENTICAL to the proven
            // gather_reader.cpp inner loop -- face addressing unchanged).
            uint32_t count = 0;
            for (uint32_t i = 0; i < tile_faces; ++i) {
                for (uint32_t j = 0; j < tile_faces; ++j) {
                    for (uint32_t k = 0; k < face_size; ++k) {
                        for (uint32_t l = 0; l < face_size; l++) {
                            const uint32_t row_in_tile = i * face_size + k;
                            const uint32_t col_in_tile = j * face_size + l;
                            const uint32_t global_index = (row_in_tile < valid_h && col_in_tile < valid_w)
                                                              ? get_value_from_tile(index_l1, count, index_df_size)
                                                              : 0;

                            const uint32_t tile_idx = global_index >> __builtin_ctz(tile_width);
                            const uint32_t index_in_local_tile = global_index & tile_width_mask;
                            const uint32_t which_row = index_in_local_tile >> __builtin_ctz(face_size);
                            const uint32_t which_col = index_in_local_tile & FACE_SIZE_MASK;

                            const uint32_t local_index = tile_idx * (tile_width * tile_height) +
                                                         which_row * (face_size * face_size) + k * face_size +
                                                         which_col + i * (tile_width * face_size);

                            const uint32_t value = get_value_from_tile(input_l1, local_index, input_df_size);
                            write_value_to_tile(output_l1, count, output_df_size, value);
                            count++;
                        }
                    }
                }
            }

            output_buffer.push_back(one_tile);
            index_buffer.pop_front(one_tile);
        }  // column sub-range

        // Release this row's input tiles so the writer can load the next row.
        input_buffer.pop_front(Wt_input);

        g += w_count;
        remaining -= w_count;
    }  // row chunks
}
