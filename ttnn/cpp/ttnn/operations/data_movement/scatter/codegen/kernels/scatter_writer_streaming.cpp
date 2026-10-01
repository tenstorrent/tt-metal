// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scatter writer – streaming mode (BRISC).
// Dual-RISC: loads input tiles into cb_input, writes output tiles to DRAM.
//
// Work is split by Wt_output across cores. Each core handles its assigned
// output columns across ALL Ht rows.

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
    constexpr uint32_t Wt_output = get_compile_time_arg_val(3);
    constexpr uint32_t num_cores = get_compile_time_arg_val(4);
    constexpr auto input_ta_args = TensorAccessorArgs<5>();
    constexpr auto output_ta_args = TensorAccessorArgs<input_ta_args.next_compile_time_args_offset()>();

    constexpr uint32_t one_tile = 1;

    // Tensor accessors
    constexpr uint32_t input_tile_bytes = get_tile_size(cb_input);
    const auto input_accessor = TensorAccessor(input_ta_args, input_addr, input_tile_bytes);

    constexpr uint32_t output_tile_bytes = get_tile_size(cb_output);
    const auto output_accessor = TensorAccessor(output_ta_args, output_addr, output_tile_bytes);

    Noc noc;
    CircularBuffer in_cb(cb_input);
    CircularBuffer out_cb(cb_output);

    // Reset the output-column tile id per row h (see scatter_reader_streaming.cpp):
    // the DRAM tile id is h*Wt_output + column and this core owns the same strided
    // columns in every row. A counter carried across h drifts and streams/writes
    // the wrong row's tiles.
    for (uint32_t h = 0; h < Ht; h++) {
        uint32_t current_output_tile_id = core_id;
        for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
            // Phase 1: Load one input tile into cb_input
            in_cb.reserve_back(one_tile);
            noc.async_read(
                input_accessor,
                in_cb,
                input_tile_bytes,
                {.page_id = h * Wt_output + current_output_tile_id, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            in_cb.push_back(one_tile);

            // Phase 2: Write one completed output tile to DRAM
            out_cb.wait_front(one_tile);
            noc.async_write(
                out_cb,
                output_accessor,
                output_tile_bytes,
                {.offset_bytes = 0},
                {.page_id = h * Wt_output + current_output_tile_id, .offset_bytes = 0});
            noc.async_writes_flushed();
            out_cb.pop_front(one_tile);

            current_output_tile_id += num_cores;
        }  // core_loop
    }  // Ht loop
    noc.async_write_barrier();
}
