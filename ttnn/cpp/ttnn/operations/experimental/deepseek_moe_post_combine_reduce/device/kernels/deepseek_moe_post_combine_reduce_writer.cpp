// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

/**
 * Writer kernel for post-combine reduce.
 *
 * Reads TILE_LAYOUT output from compute kernel and writes to DRAM.
 *
 * Output shape: [seq_len, emb_dim] in TILE_LAYOUT
 * - emb_dim_tiles tiles per token (7 tiles for 7168 embedding dim)
 */

constexpr uint32_t cb_output = tt::CBIndex::c_16;

// Compile-time arguments
constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;
constexpr uint32_t num_tokens = get_compile_time_arg_val(1);
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(2);

void kernel_main() {
    // Runtime arguments
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t tokens_per_core = get_arg_val<uint32_t>(1);
    uint32_t token_start_idx = get_arg_val<uint32_t>(2);

    constexpr uint32_t tile_size = 2048;  // bfloat16 tile

    // Output page size = entire row (all emb_dim_tiles)
    constexpr uint32_t output_page_size = emb_dim_tiles * tile_size;

    // Setup address generator for output
    const InterleavedAddrGenFast<output_is_dram> output_addrg = {
        .bank_base_address = output_addr, .page_size = output_page_size, .data_format = DataFormat::Float16_b};

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Process tokens assigned to this core
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for (uint32_t token_idx = 0; token_idx < tokens_per_core; ++token_idx) {
        uint32_t global_token_idx = token_start_idx + token_idx;

        // DPRINT << "WRITER: Writing token " << global_token_idx << ENDL();

        // Wait for output tiles for this token (224 tiles)
        cb_wait_front(cb_output, emb_dim_tiles);

        uint32_t output_read_addr = get_read_ptr(cb_output);

        // ──────────────────────────────────────────────────────────────────────
        // Write all output tiles for this token as one contiguous page
        // (ROW_MAJOR: entire row is one page)
        // ──────────────────────────────────────────────────────────────────────
        uint32_t output_page_idx = global_token_idx;  // One page per token
        noc_async_write_page(output_page_idx, output_addrg, output_read_addr);

        noc_async_write_barrier();

        // DPRINT_DATA1(DPRINT << "4.WRITER[tok=" << global_token_idx << "]: Wrote to page=" << output_page_idx <<
        // ENDL());

        // DPRINT << "  WRITER: Wrote " << emb_dim_tiles << " tiles to DRAM" << ENDL();

        cb_pop_front(cb_output, emb_dim_tiles);
    }
}
