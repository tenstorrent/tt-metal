// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/tilize.h"

// Compute kernel for tilizing incoming tokens from the reader.
// Waits for writer to pass total_chunks via CB, then processes each chunk:
// - Wait for reader to push tokens_per_chunk tokens
// - Tilize the block
// - Push tilized output to writer

namespace NAMESPACE {
void MAIN {
    // Compile-time arguments
    constexpr uint32_t tilizer_input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t tilizer_output_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(2);
    constexpr uint32_t tokens_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t total_chunks_cb_id = get_compile_time_arg_val(4);

    // Wait for writer to push total_chunks via CB
    cb_wait_front(total_chunks_cb_id, 1);

    // Read total_chunks from the CB
    uint32_t total_chunks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(total_chunks_cb_id, 0));

    // Process each chunk
    for (uint32_t chunk = 0; chunk < total_chunks; chunk++) {
        // Wait for reader to push tokens_per_chunk pages (row-major data)
        // Reader always reserves/pushes tokens_per_chunk for consistent synchronization
        cb_wait_front(tilizer_input_cb_id, tokens_per_chunk);
        cb_reserve_back(tilizer_output_cb_id, tiles_per_chunk);

        // Pop input from reader (tokens_per_chunk pages)
        cb_pop_front(tilizer_input_cb_id, tokens_per_chunk);

        // Reserve and push dummy output to keep writer in sync
        // TODO: Proper tilization requires CB reconfiguration
        cb_push_back(tilizer_output_cb_id, tiles_per_chunk);
    }

    // Pop the total_chunks CB page (cleanup)
    cb_pop_front(total_chunks_cb_id, 1);
}
}  // namespace NAMESPACE
