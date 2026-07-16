// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"

// Compute kernel for tilizing incoming tokens from the reader.
// Waits for writer to pass total_chunks via CB, then processes each chunk:
// - Wait for reader to push tokens_per_chunk tokens
// - Tilize the block
// - Push tilized output to writer
void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t tilize_input_cb_id = get_named_compile_time_arg_val("tilize_input_cb_id");
    constexpr uint32_t tilize_output_cb_id = get_named_compile_time_arg_val("tilize_output_cb_id");
    constexpr uint32_t total_chunks_cb_id = get_named_compile_time_arg_val("total_chunks_cb_id");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");
    constexpr uint32_t shared_cb_num_pages = get_named_compile_time_arg_val("shared_cb_num_pages");

    // Runtime arguments
    uint32_t rt_args_idx = 0;
    uint32_t tiles_per_local_chunk = get_arg_val<uint32_t>(rt_args_idx++);

    // Constants
    constexpr uint32_t one_page = 1;

    DataflowBuffer cb_tilize_input(tilize_input_cb_id);
    DataflowBuffer cb_tilize_output(tilize_output_cb_id);
    DataflowBuffer cb_total_chunks(total_chunks_cb_id);

    // Setup
    compute_kernel_hw_startup(cb_tilize_input.get_id(), cb_tilize_output.get_id());
    fast_tilize_init(cb_tilize_input.get_id(), tiles_per_local_chunk, cb_tilize_output.get_id());

    // Wait for writer to push total_chunks via CB
    cb_total_chunks.wait_front(one_page);

    // Read total_chunks from the CB
    uint32_t total_chunks =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_total_chunks.get_id(), 0));

    // Process each chunk
    for (uint32_t chunk = 0; chunk < total_chunks; chunk++) {
        // Wait for reader to push tokens_per_chunk pages (row-major data)
        // Reader always reserves/pushes tokens_per_chunk for consistent synchronization
        cb_tilize_input.wait_front(tokens_per_chunk);

        // we reserve the entire CB so that we treat it as single buffered
        // this is to allow us to gather into it before mcasting to the MM cores
        cb_tilize_output.reserve_back(shared_cb_num_pages);

        fast_tilize_block(cb_tilize_input.get_id(), tiles_per_local_chunk, cb_tilize_output.get_id());

        cb_tilize_output.push_back(shared_cb_num_pages);

        // Pop input from reader (tokens_per_chunk pages)
        cb_tilize_input.pop_front(tokens_per_chunk);
    }

    fast_tilize_uninit(cb_tilize_input.get_id(), cb_tilize_output.get_id(), tiles_per_local_chunk);
    cb_total_chunks.pop_front(one_page);
}
