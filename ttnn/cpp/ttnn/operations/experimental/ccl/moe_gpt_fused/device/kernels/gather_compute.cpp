// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Gather Compute Kernel for moe_gpt_fused
//
// Tilizes ROW_MAJOR input from c_7 into TILE format in c_16.
// Each tilize core handles tiles_per_local_chunk tiles (30 for 3-core config).

#include <cstdint>
#include "api/compute/cb_api.h"
#include "api/compute/tilize.h"
#include "moe_gpt_fused_ring_common.h"

void kernel_main() {
    constexpr auto tilize_input_cb = tt::CBIndex::c_7;
    constexpr auto tilize_output_cb = tt::CBIndex::c_16;
    constexpr uint32_t tokens_per_chunk = moe_gpt_fused_ring::TOKENS_PER_CHUNK;            // 32
    constexpr uint32_t tiles_per_local_chunk = moe_gpt_fused_ring::TILES_PER_GATHER_CORE;  // 30

    // Output CB is sized for all 90 tiles (drain core gathers from non-drain cores)
    constexpr uint32_t output_cb_num_pages = moe_gpt_fused_ring::K_TILES;  // 90

    compute_kernel_hw_startup(tilize_input_cb, tilize_output_cb);
    fast_tilize_init(tilize_input_cb, tiles_per_local_chunk, tilize_output_cb);

    // Wait for reader to push input data
    cb_wait_front(tilize_input_cb, tokens_per_chunk);

    // Reserve entire output CB (single-buffered for gather)
    cb_reserve_back(tilize_output_cb, output_cb_num_pages);

    // Tilize the block
    fast_tilize_block(tilize_input_cb, tiles_per_local_chunk, tilize_output_cb);

    // Push tilized tiles and pop input
    cb_push_back(tilize_output_cb, output_cb_num_pages);
    cb_pop_front(tilize_input_cb, tokens_per_chunk);

    fast_tilize_uninit(tilize_input_cb, tilize_output_cb);
}
