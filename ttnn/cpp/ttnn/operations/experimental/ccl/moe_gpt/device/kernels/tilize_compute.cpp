// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/tilize.h"

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
    constexpr uint32_t max_tiles_per_local_chunk = get_named_compile_time_arg_val("max_tiles_per_local_chunk");
    constexpr uint32_t shared_cb_num_pages = get_named_compile_time_arg_val("shared_cb_num_pages");

    // Runtime arguments
    uint32_t rt_args_idx = 0;
    uint32_t tiles_per_local_chunk = get_arg_val<uint32_t>(rt_args_idx++);

    // Constants
    constexpr uint32_t one_page = 1;

    // Setup
    compute_kernel_hw_startup(tilize_input_cb_id, tilize_output_cb_id);
    fast_tilize_init(tilize_input_cb_id, max_tiles_per_local_chunk, tilize_output_cb_id);

    // Wait for writer to push total_chunks via CB
    cb_wait_front(total_chunks_cb_id, one_page);

    // Read total_chunks from the CB
    uint32_t total_chunks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(total_chunks_cb_id, 0));

    // Process each chunk
    for (uint32_t chunk = 0; chunk < total_chunks; chunk++) {
        // Wait for reader to push tokens_per_chunk pages (row-major data)
        // Reader always reserves/pushes tokens_per_chunk for consistent synchronization
        cb_wait_front(tilize_input_cb_id, tokens_per_chunk);

        // we reserve the entire CB so that we treat it as single buffered
        // this is to allow us to gather into it before mcasting to the MM cores
        cb_reserve_back(tilize_output_cb_id, shared_cb_num_pages);

        fast_tilize_block(tilize_input_cb_id, tiles_per_local_chunk, tilize_output_cb_id);

        cb_push_back(tilize_output_cb_id, shared_cb_num_pages);

        // Pop input from reader (tokens_per_chunk pages)
        cb_pop_front(tilize_input_cb_id, tokens_per_chunk);
    }

    fast_tilize_uninit(tilize_input_cb_id, tilize_output_cb_id);
    cb_pop_front(total_chunks_cb_id, one_page);
}
