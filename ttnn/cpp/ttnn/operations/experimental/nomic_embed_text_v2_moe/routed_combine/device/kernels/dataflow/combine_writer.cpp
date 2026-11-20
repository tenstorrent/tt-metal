// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_1 of every core: each unit's output tiles, a chunk of one 32-token block's tile row.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t chunk_tiles = get_named_compile_time_arg_val("chunk_tiles");
    constexpr uint32_t output_cb_id = get_named_compile_time_arg_val("output_cb");
    constexpr auto output_args = TensorAccessorArgs<0>();

    const uint32_t output_address = get_arg_val<uint32_t>(0);
    const uint32_t first_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t chunks = hidden_tiles / chunk_tiles;
    const Noc noc;
    DataflowBuffer output_cb(output_cb_id);
    const auto output_accessor = TensorAccessor(output_args, output_address);
    constexpr uint32_t tile_bytes = get_tile_size(output_cb_id);

    for (uint32_t unit = first_unit; unit < first_unit + num_units; ++unit) {
        uint32_t page = (unit / chunks) * hidden_tiles + (unit % chunks) * chunk_tiles;
        output_cb.wait_front(chunk_tiles);
        for (uint32_t tile = 0; tile < chunk_tiles; ++tile, ++page) {
            noc.async_write(
                output_cb, output_accessor, tile_bytes, {.offset_bytes = tile * tile_bytes}, {.page_id = page});
        }
        noc.async_write_barrier();
        output_cb.pop_front(chunk_tiles);
    }
}
