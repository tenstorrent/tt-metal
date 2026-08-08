// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather reader (NCRISC): reads index tiles from DRAM, performs element-level
// L1 gather from input data (already loaded by writer into cb_input),
// and pushes output tiles to cb_output for the writer to flush to DRAM.
//
// dim=-1 path: each "row" is one tile-row (Ht index). For each row the writer
// has loaded Wt_input data tiles into cb_input. We iterate over Wt_index index
// tiles, and for each element in the index tile we look up the corresponding
// element in the data tiles and write it to the output tile.
//
// Multicore: work is split by Ht rows. Each core processes core_loop_count
// rows using a strided pattern: row = core_loop * num_cores + core_id.

#include "codegen_gather_common.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t index_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(1);
    const uint32_t tile_width = get_arg_val<uint32_t>(2);
    const uint32_t tile_height = get_arg_val<uint32_t>(3);
    const uint32_t core_id = get_arg_val<uint32_t>(4);

    // Compile-time args
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output = get_compile_time_arg_val(2);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(3);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(4);
    constexpr uint32_t num_cores = get_compile_time_arg_val(5);
    constexpr uint32_t index_valid_h_last = get_compile_time_arg_val(6);
    constexpr uint32_t index_valid_w_last = get_compile_time_arg_val(7);
    constexpr uint32_t index_ht_per_batch = get_compile_time_arg_val(8);
    constexpr auto index_ta_args = TensorAccessorArgs<9>();

    constexpr uint32_t one_tile = 1;
    const uint32_t tile_width_mask = tile_width - 1;

    // Index tensor accessor
    constexpr uint32_t index_tile_bytes = get_tile_size(cb_index);
    const auto index_accessor = TensorAccessor(index_ta_args, index_addr, index_tile_bytes);

    Noc noc;
    CircularBuffer input_buffer(cb_input);
    CircularBuffer index_buffer(cb_index);
    CircularBuffer output_buffer(cb_output);

    // Data format sizes for element-level access
    constexpr uint32_t input_df_size = get_tile_size(cb_input) / get_tile_hw(cb_input);
    constexpr uint32_t index_df_size = index_tile_bytes / get_tile_hw(cb_index);
    constexpr uint32_t output_df_size = get_tile_size(cb_output) / get_tile_hw(cb_output);

    constexpr uint32_t face_size = 16;
    constexpr uint32_t FACE_SIZE_MASK = face_size - 1;
    constexpr uint32_t tile_faces = 2;

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * num_cores + core_id;
        const uint32_t h_in_batch = h % index_ht_per_batch;
        const uint32_t valid_h = (h_in_batch == index_ht_per_batch - 1) ? index_valid_h_last : tile_height;

        for (uint32_t w = 0; w < Wt_index; w++) {
            const uint32_t valid_w = (w == Wt_index - 1) ? index_valid_w_last : tile_width;
            // Read one index tile from DRAM
            index_buffer.reserve_back(one_tile);
            noc.async_read(
                index_accessor,
                index_buffer,
                index_tile_bytes,
                {.page_id = h * Wt_index + w, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            index_buffer.push_back(one_tile);

            // Wait for input data row (loaded by writer) and index tile
            input_buffer.wait_front(Wt_input);
            index_buffer.wait_front(one_tile);
            output_buffer.reserve_back(one_tile);

            const uint32_t input_l1 = input_buffer.get_read_ptr();
            const uint32_t index_l1 = index_buffer.get_read_ptr();
            const uint32_t output_l1 = output_buffer.get_write_ptr();

            // Element-level gather: iterate over all elements in the index tile.
            // The valid_h/valid_w bounds check is hoisted out of the element loop:
            // on the in-order baby RISC the per-element compare+select is a
            // measurable fraction of the gather-address math it sits inside, and
            // nearly every tile is full.
            const bool full_tile = (valid_h == tile_height) && (valid_w == tile_width);
            uint32_t count = 0;
            if (full_tile) {
                for (uint32_t i = 0; i < tile_faces; ++i) {
                    for (uint32_t j = 0; j < tile_faces; ++j) {
                        for (uint32_t k = 0; k < face_size; ++k) {
                            for (uint32_t l = 0; l < face_size; l++) {
                                const uint32_t global_index = get_value_from_tile(index_l1, count, index_df_size);

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
            } else {
                for (uint32_t i = 0; i < tile_faces; ++i) {
                    for (uint32_t j = 0; j < tile_faces; ++j) {
                        for (uint32_t k = 0; k < face_size; ++k) {
                            for (uint32_t l = 0; l < face_size; l++) {
                                // Read the global index value from the index tile
                                const uint32_t row_in_tile = i * face_size + k;
                                const uint32_t col_in_tile = j * face_size + l;
                                const uint32_t global_index = (row_in_tile < valid_h && col_in_tile < valid_w)
                                                                  ? get_value_from_tile(index_l1, count, index_df_size)
                                                                  : 0;

                                // Map global_index to local_index in tiled layout:
                                // tile_idx = which input tile along W
                                // index_in_local_tile = position within that tile
                                const uint32_t tile_idx = global_index >> __builtin_ctz(tile_width);
                                const uint32_t index_in_local_tile = global_index & tile_width_mask;
                                const uint32_t which_row = index_in_local_tile >> __builtin_ctz(face_size);
                                const uint32_t which_col = index_in_local_tile & FACE_SIZE_MASK;

                                const uint32_t local_index = tile_idx * (tile_width * tile_height) +
                                                             which_row * (face_size * face_size) + k * face_size +
                                                             which_col + i * (tile_width * face_size);

                                // Gather: read from input, write to output
                                const uint32_t value = get_value_from_tile(input_l1, local_index, input_df_size);
                                write_value_to_tile(output_l1, count, output_df_size, value);
                                count++;
                            }
                        }
                    }
                }
            }

            output_buffer.push_back(one_tile);
            index_buffer.pop_front(one_tile);
        }  // Wt_index loop

        // Done with this row's input data
        input_buffer.pop_front(Wt_input);
    }  // core_loop loop
}
