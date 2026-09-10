// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gather writer (BRISC): dual-RISC pattern.
// 1) Reads full input data row (Wt_input tiles) from DRAM into cb_input.
// 2) Writes completed output tiles from cb_output to DRAM.
//
// This runs on BRISC while the reader (NCRISC) reads index tiles and does
// the element-level gather. Both RISCs do DRAM reads concurrently.
//
// Pipelined: batches tile reads and writes for NOC overlap.
// Multicore: strided row assignment, same as reader.

#include "codegen_gather_common.hpp"
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
    constexpr uint32_t Wt_input = get_compile_time_arg_val(2);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(3);
    constexpr uint32_t num_cores = get_compile_time_arg_val(4);
    // Both come from gather_output_cb_tiles()/kGatherWriteBatchTiles, the same values the factory
    // sizes cb_output with -- the Phase 2 wrap clamp below is only correct against the real depth.
    constexpr uint32_t OUT_CB_DEPTH = get_compile_time_arg_val(5);
    constexpr uint32_t WRITE_BATCH = get_compile_time_arg_val(6);
    constexpr auto input_ta_args = TensorAccessorArgs<7>();
    constexpr auto output_ta_args = TensorAccessorArgs<input_ta_args.next_compile_time_args_offset()>();

    constexpr uint32_t READ_BATCH = kGatherReadBatchTiles;
    // Read-pointer position within the output CB ring (see the Phase 2 clamp below).
    uint32_t out_cb_pos = 0;

    // Input tensor accessor (for DRAM reads)
    constexpr uint32_t input_tile_bytes = get_tile_size(cb_input);
    const auto input_accessor = TensorAccessor(input_ta_args, input_addr, input_tile_bytes);

    // Output tensor accessor (for DRAM writes)
    constexpr uint32_t output_tile_bytes = get_tile_size(cb_output);
    const auto output_accessor = TensorAccessor(output_ta_args, output_addr, output_tile_bytes);

    Noc noc;
    CircularBuffer input_buffer(cb_input);
    CircularBuffer output_buffer(cb_output);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * num_cores + core_id;

        // --- Phase 1: Read full input data row (Wt_input tiles) from DRAM ---
        // Batched: issue READ_BATCH reads before barrier
        uint32_t tiles_read = 0;
        while (tiles_read < Wt_input) {
            uint32_t batch = (Wt_input - tiles_read < READ_BATCH) ? (Wt_input - tiles_read) : READ_BATCH;
            input_buffer.reserve_back(batch);
            uint32_t l1_offset = 0;
            for (uint32_t b = 0; b < batch; b++) {
                noc.async_read(
                    input_accessor,
                    input_buffer,
                    input_tile_bytes,
                    {.page_id = h * Wt_input + tiles_read + b, .offset_bytes = 0},
                    {.offset_bytes = l1_offset});
                l1_offset += input_tile_bytes;
            }
            noc.async_read_barrier();
            input_buffer.push_back(batch);
            tiles_read += batch;
        }

        // --- Phase 2: Write completed output tiles to DRAM ---
        // Batched: issue WRITE_BATCH writes before flush
        uint32_t tiles_written = 0;
        while (tiles_written < Wt_index) {
            uint32_t batch = (Wt_index - tiles_written < WRITE_BATCH) ? (Wt_index - tiles_written) : WRITE_BATCH;
            // cb_pop_front wraps the read pointer only when a pop lands exactly on the
            // ring end (dataflow_api.h pop contract), and a flat multi-tile read must
            // not cross it: clamp each batch to the remaining distance to the wrap.
            const uint32_t to_wrap = OUT_CB_DEPTH - out_cb_pos;
            if (batch > to_wrap) {
                batch = to_wrap;
            }
            output_buffer.wait_front(batch);
            uint32_t l1_offset = 0;
            for (uint32_t b = 0; b < batch; b++) {
                noc.async_write(
                    output_buffer,
                    output_accessor,
                    output_tile_bytes,
                    {.offset_bytes = l1_offset},
                    {.page_id = h * Wt_index + tiles_written + b, .offset_bytes = 0});
                l1_offset += output_tile_bytes;
            }
            // Popping the slot only needs the write off the local NoC, not landed at the
            // destination; completion is claimed once for the whole kernel below.
            noc.async_writes_flushed();
            output_buffer.pop_front(batch);
            tiles_written += batch;
            out_cb_pos += batch;
            if (out_cb_pos == OUT_CB_DEPTH) {
                out_cb_pos = 0;
            }
        }
    }

    noc.async_write_barrier();
}
