// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather reader – streaming mode (NCRISC).
// For large Wt_input (>60 tiles) where the full input row cannot fit in L1.
//
// Work is split by Wt_index across cores (not Ht like the row-buffered mode).
// Each core processes a strided subset of output columns across ALL Ht rows.
//
// For each assigned index tile:
//   1. Read the index tile from DRAM
//   2. For each of Wt_input input tiles (streamed one at a time by writer):
//      - Scan all 1024 index elements
//      - If index points to the current input tile, copy the value to output
//      - Otherwise skip
//   3. Push completed output tile
//
// This trades bandwidth (re-reading input per index tile) for L1 budget.

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
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(4);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr uint32_t index_valid_h_last = get_compile_time_arg_val(7);
    constexpr uint32_t index_valid_w_last = get_compile_time_arg_val(8);
    constexpr uint32_t index_ht_per_batch = get_compile_time_arg_val(9);
    constexpr auto index_ta_args = TensorAccessorArgs<10>();

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

    // Column-tile id must restart at core_id for EACH tile-row h: this core owns
    // the strided columns {core_id, core_id+num_cores, ...} within every row, and
    // the DRAM tile id is h*Wt_* + column. A single running counter (not reset per
    // h) drifts past Wt_index after the first row, reading/writing the wrong row's
    // tiles — garbage for Ht>1 (root cause of the >60-tile gather corruption).
    for (uint32_t h = 0; h < Ht; h++) {
        const uint32_t h_in_batch = h % index_ht_per_batch;
        const uint32_t valid_h = (h_in_batch == index_ht_per_batch - 1) ? index_valid_h_last : tile_height;
        uint32_t current_index_tile_id = core_id;
        for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
            const uint32_t valid_w = (current_index_tile_id == Wt_index - 1) ? index_valid_w_last : tile_width;
            // Read one index tile from DRAM
            index_buffer.reserve_back(one_tile);
            noc.async_read(
                index_accessor,
                index_buffer,
                index_tile_bytes,
                {.page_id = h * Wt_index + current_index_tile_id, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            index_buffer.push_back(one_tile);
            index_buffer.wait_front(one_tile);

            output_buffer.reserve_back(one_tile);

            // Stream through all Wt_input tiles, gathering matching elements
            for (uint32_t wi = 0; wi < Wt_input; wi++) {
                // Wait for writer to provide next input tile
                input_buffer.wait_front(one_tile);

                const uint32_t input_l1 = input_buffer.get_read_ptr();
                const uint32_t index_l1 = index_buffer.get_read_ptr();
                const uint32_t output_l1 = output_buffer.get_write_ptr();

                // Scan all elements in index tile, copy those pointing to tile wi
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

                                if (tile_idx != wi) {
                                    count++;
                                    continue;
                                }

                                const uint32_t index_in_local_tile = global_index & tile_width_mask;
                                const uint32_t which_row = index_in_local_tile >> __builtin_ctz(face_size);
                                const uint32_t which_col = index_in_local_tile & FACE_SIZE_MASK;

                                const uint16_t local_index = which_row * (face_size * face_size) + k * face_size +
                                                             which_col + i * (tile_width * face_size);

                                const uint32_t value = get_value_from_tile(input_l1, local_index, input_df_size);
                                write_value_to_tile(output_l1, count, output_df_size, value);
                                count++;
                            }
                        }
                    }
                }
                // Release input tile so writer can reuse the CB slot
                input_buffer.pop_front(one_tile);
            }  // Wt_input loop

            output_buffer.push_back(one_tile);
            index_buffer.pop_front(one_tile);
            current_index_tile_id += num_cores;
        }  // core_loop
    }  // Ht loop
}
