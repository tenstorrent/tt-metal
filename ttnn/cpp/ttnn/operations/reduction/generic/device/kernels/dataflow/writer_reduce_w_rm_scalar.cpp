// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include <tt-metalium/constants.hpp>

FORCE_INLINE uint32_t get_tilized_idx(uint32_t h_idx, uint32_t w_idx) {
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    constexpr uint32_t half_tile_height = tile_height / 2;
    constexpr uint32_t half_tile_width = tile_width / 2;

    if (h_idx < half_tile_height && w_idx < half_tile_width) {
        return h_idx * half_tile_width + w_idx;
    }
    if (h_idx < half_tile_height && w_idx >= half_tile_width) {
        return h_idx * half_tile_width + (w_idx % half_tile_width) + half_tile_height * half_tile_width;
    }
    if (h_idx >= half_tile_height && w_idx < half_tile_width) {
        return (h_idx % half_tile_height) * half_tile_width + w_idx + half_tile_height * tile_width;
    }
    return (h_idx % half_tile_height) * half_tile_width + (w_idx % half_tile_width) +
           half_tile_height * (tile_width + half_tile_width);
}

// Writes one scalar (datum_bytes) per reduced logical row. Each output tile now contains up to TILE_HEIGHT row
// reductions, so writer emits multiple destination rows per popped tile.
void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_tile = tt::CBIndex::c_3;
    constexpr uint32_t datum_bytes = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    constexpr uint32_t onetile = 1;

    const auto dst_accessor = TensorAccessor(dst_args, dst_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_tile(cb_id_tile);

    uint32_t rows_written = 0;
    while (rows_written < num_rows) {
        cb_tile.wait_front(onetile);
        const uint32_t rows_this_tile = ((num_rows - rows_written) < tt::constants::TILE_HEIGHT)
                                            ? (num_rows - rows_written)
                                            : tt::constants::TILE_HEIGHT;
        for (uint32_t r = 0; r < rows_this_tile; ++r) {
            const uint32_t tile_scalar_idx = get_tilized_idx(r, 0);
            noc.async_write(
                cb_tile,
                dst_accessor,
                datum_bytes,
                {.offset_bytes = tile_scalar_idx * datum_bytes},
                {.page_id = start_page + rows_written + r, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        cb_tile.pop_front(onetile);
        rows_written += rows_this_tile;
    }
}
