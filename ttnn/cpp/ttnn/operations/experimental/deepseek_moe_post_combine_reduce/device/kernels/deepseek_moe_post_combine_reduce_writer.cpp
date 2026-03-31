// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_output = tt::CBIndex::c_16;

constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;
constexpr uint32_t num_tokens = get_compile_time_arg_val(1);
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(2);

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t tokens_per_core = get_arg_val<uint32_t>(1);
    uint32_t token_start_idx = get_arg_val<uint32_t>(2);

    constexpr uint32_t tile_size = 2048;

    constexpr uint32_t output_page_size = emb_dim_tiles * tile_size;

    const InterleavedAddrGenFast<output_is_dram> output_addrg = {
        .bank_base_address = output_addr, .page_size = output_page_size, .data_format = DataFormat::Float16_b};

    // DPRINT_DATA1(DPRINT << "tokens_per_core == " << tokens_per_core << ENDL());
    // DPRINT_DATA1(DPRINT << "token_start_idx == " << token_start_idx << ENDL());

    for (uint32_t token_idx = 0; token_idx < tokens_per_core; ++token_idx) {
        uint32_t global_token_idx = token_start_idx + token_idx;

        cb_wait_front(cb_output, emb_dim_tiles);

        // SliceRange sr = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};
        // DPRINT_DATA1(DPRINT << "writing data to DRAM -- " << "\n" << ENDL());
        // for (uint32_t j = 0; j < emb_dim_tiles; j++) {
        //     DPRINT_DATA1(DPRINT << "tile " << j << " values = " << TileSlice(cb_output, j, sr, true, false) <<
        //     ENDL());
        // }
        uint32_t output_read_addr = get_read_ptr(cb_output);

        uint32_t output_page_idx = global_token_idx;
        noc_async_write_page(output_page_idx, output_addrg, output_read_addr);

        noc_async_write_barrier();

        cb_pop_front(cb_output, emb_dim_tiles);
    }
}
