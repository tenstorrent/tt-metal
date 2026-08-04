// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather writer – streaming mode (BRISC).
// For large Wt_input (>60 tiles) where the full input row cannot fit in L1.
//
// Work is split by Wt_index across cores. Each core processes a strided
// subset of output columns across ALL Ht rows.
//
// For each assigned index tile:
//   1. Read all Wt_input tiles one at a time into cb_input (double-buffered)
//      so the reader can stream through them
//   2. Wait for the completed output tile from the reader
//   3. Write the output tile to DRAM

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(2);
    const uint32_t core_id = get_arg_val<uint32_t>(3);

    // Compile-time args
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(3);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(4);
    constexpr uint32_t num_cores = get_compile_time_arg_val(5);
    constexpr auto input_ta_args = TensorAccessorArgs<6>();
    constexpr auto output_ta_args = TensorAccessorArgs<input_ta_args.next_compile_time_args_offset()>();

    constexpr uint32_t one_tile = 1;

    // Input tensor accessor (for DRAM reads)
    constexpr uint32_t input_tile_bytes = get_tile_size(cb_input);
    const auto input_accessor = TensorAccessor(input_ta_args, input_addr, input_tile_bytes);

    // Output tensor accessor (for DRAM writes)
    constexpr uint32_t output_tile_bytes = get_tile_size(cb_output);
    const auto output_accessor = TensorAccessor(output_ta_args, output_addr, output_tile_bytes);

    Noc noc;
    CircularBuffer input_buffer(cb_input);
    CircularBuffer output_buffer(cb_output);

    // Reset the column-tile id per row h (see gather_reader_streaming.cpp): the
    // DRAM tile id is h*Wt_* + column, and this core owns the same strided columns
    // in every row. A counter carried across h drifts and streams the wrong row.
    for (uint32_t h = 0; h < Ht; h++) {
        uint32_t current_index_tile_id = core_id;
        for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
            // Stream all Wt_input tiles one at a time for the reader to process
            for (uint32_t w = 0; w < Wt_input; w++) {
                input_buffer.reserve_back(one_tile);
                noc.async_read(
                    input_accessor,
                    input_buffer,
                    input_tile_bytes,
                    {.page_id = h * Wt_input + w, .offset_bytes = 0},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                input_buffer.push_back(one_tile);
            }

            // Wait for completed output tile and write to DRAM
            output_buffer.wait_front(one_tile);
            noc.async_write(
                output_buffer,
                output_accessor,
                output_tile_bytes,
                {.offset_bytes = 0},
                {.page_id = h * Wt_index + current_index_tile_id, .offset_bytes = 0});
            noc.async_write_barrier();
            output_buffer.pop_front(one_tile);

            current_index_tile_id += num_cores;
        }  // core_loop
    }  // Ht loop
}
