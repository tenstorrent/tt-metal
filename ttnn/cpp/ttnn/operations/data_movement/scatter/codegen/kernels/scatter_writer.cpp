// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scatter writer (BRISC): dual-RISC pattern (mirrors gather_writer).
// 1) Reads full input data row (Wt_output tiles) from DRAM into cb_input.
// 2) Writes completed output tiles from cb_output to DRAM.
//
// This runs on BRISC while the reader (NCRISC) reads index/src tiles and does
// the element-level scatter. Both RISCs do DRAM ops concurrently.
//
// Multicore: strided row assignment, same as reader.

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
    constexpr uint32_t Wt_output = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores = get_compile_time_arg_val(3);
    constexpr auto input_ta_args = TensorAccessorArgs<4>();
    constexpr auto output_ta_args = TensorAccessorArgs<input_ta_args.next_compile_time_args_offset()>();

    constexpr uint32_t READ_BATCH = 4;
    constexpr uint32_t WRITE_BATCH = 4;

    // Input tensor accessor (for DRAM reads)
    constexpr uint32_t input_tile_bytes = get_tile_size(cb_input);
    const auto input_accessor = TensorAccessor(input_ta_args, input_addr, input_tile_bytes);

    // Output tensor accessor (for DRAM writes)
    constexpr uint32_t output_tile_bytes = get_tile_size(cb_output);
    const auto output_accessor = TensorAccessor(output_ta_args, output_addr, output_tile_bytes);

    Noc noc;
    CircularBuffer in_cb(cb_input);
    CircularBuffer out_cb(cb_output);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * num_cores + core_id;

        // --- Phase 1: Read full input data row (Wt_output tiles) from DRAM ---
        uint32_t tiles_read = 0;
        while (tiles_read < Wt_output) {
            uint32_t batch = (Wt_output - tiles_read < READ_BATCH) ? (Wt_output - tiles_read) : READ_BATCH;
            in_cb.reserve_back(batch);
            uint32_t l1_offset = 0;
            for (uint32_t b = 0; b < batch; b++) {
                noc.async_read(
                    input_accessor,
                    in_cb,
                    input_tile_bytes,
                    {.page_id = h * Wt_output + tiles_read + b, .offset_bytes = 0},
                    {.offset_bytes = l1_offset});
                l1_offset += input_tile_bytes;
            }
            noc.async_read_barrier();
            in_cb.push_back(batch);
            tiles_read += batch;
        }

        // --- Phase 2: Write completed output tiles to DRAM ---
        uint32_t tiles_written = 0;
        while (tiles_written < Wt_output) {
            uint32_t batch = (Wt_output - tiles_written < WRITE_BATCH) ? (Wt_output - tiles_written) : WRITE_BATCH;
            out_cb.wait_front(batch);
            uint32_t l1_offset = 0;
            for (uint32_t b = 0; b < batch; b++) {
                noc.async_write(
                    out_cb,
                    output_accessor,
                    output_tile_bytes,
                    {.offset_bytes = l1_offset},
                    {.page_id = h * Wt_output + tiles_written + b, .offset_bytes = 0});
                l1_offset += output_tile_bytes;
            }
            noc.async_writes_flushed();
            out_cb.pop_front(batch);
            tiles_written += batch;
        }
    }
    noc.async_write_barrier();
}
