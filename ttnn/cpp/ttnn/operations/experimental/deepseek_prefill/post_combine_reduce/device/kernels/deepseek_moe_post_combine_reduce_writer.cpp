// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_output_id = tt::CBIndex::c_16;

constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(0);
constexpr uint32_t emb_dim_out_tiles = get_compile_time_arg_val(1);
constexpr auto output_accessor_args = TensorAccessorArgs<2>();

constexpr uint32_t TOKENS_PER_CHUNK = 32;

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);
    const uint32_t num_chunks = get_arg_val<uint32_t>(2);

    Noc noc;
    CircularBuffer cb_output(cb_output_id);
    const uint32_t output_tile_size = cb_output.get_tile_size();
    const auto output_addrg = TensorAccessor(output_accessor_args, output_addr);

    constexpr uint32_t cb_output_tiles = emb_dim_cb_tiles * TOKENS_PER_CHUNK;

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // The output CB holds TOKENS_PER_CHUNK * emb_dim_cb_tiles tile-sized pages (padded when emb_dim is not
        // 1024-aligned); only the first emb_dim_out_tiles hold real data for this 32-token block.
        cb_output.wait_front(cb_output_tiles);

        const uint32_t start_tile_idx = (token_start_idx / TOKENS_PER_CHUNK) * emb_dim_out_tiles;
        for (uint32_t tile_idx = 0; tile_idx < emb_dim_out_tiles; ++tile_idx) {
            noc.async_write(
                cb_output,
                output_addrg,
                output_tile_size,
                {.offset_bytes = tile_idx * output_tile_size},
                {.page_id = start_tile_idx + tile_idx});
        }
        noc.async_write_barrier();

        cb_output.pop_front(cb_output_tiles);
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
