// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize 2D-work-split reader (NCRISC / NoC0) — interleaved width-split path.
//
// The height-only interleaved reader (tilize_reader.cpp) gives each core a
// contiguous tile-ROW range and reads the FULL width of those rows. For a wide,
// short tensor (few tile-rows, many tile-columns) that collapses to a handful of
// cores. This reader instead distributes flat WORK UNITS across the grid: a unit
// is one (tile-row, column-chunk) pair — 32 row-major sticks restricted to a
// `Wt_chunk`-wide column slice — the smallest independently-tilizable block.
//
// Column-chunks per tile-row: C = Wt / Wt_chunk. Global unit index u decodes to
//   row   = u / C          (which tile-row's 32 sticks)
//   chunk = u % C          (which Wt_chunk-wide column slice of that row)
// Each core owns a contiguous [u_start, u_start+u_count) unit range (row-major
// over units). The output tiles for unit u land at pages
//   [row*Wt + chunk*Wt_chunk, row*Wt + chunk*Wt_chunk + Wt_chunk)
// (see tilize_writer_2d.cpp) — the standard tile-row-major page order.
//
// Per unit we reuse dataflow_kernel_lib::read_sticks_for_tilize with
// total_num_rows = 32 (one tile-row = one block): it reads the 32 sticks, each
// contributing `chunk_width_bytes` starting at byte `chunk*chunk_width_bytes` of
// its source page, and pushes Wt_chunk tile-pages. The CB stays 2*Wt_chunk*tile
// (constant in W) — the memory-budget bound is unchanged from the height path.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t chunk_width_bytes = get_compile_time_arg_val(0);  // Wt_chunk * 32 * elem_size
    constexpr uint32_t chunks_per_row = get_compile_time_arg_val(1);     // C = Wt / Wt_chunk
    constexpr auto src_args = TensorAccessorArgs<2>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t u_start = get_arg_val<uint32_t>(1);  // per-core first unit
    const uint32_t u_count = get_arg_val<uint32_t>(2);  // per-core unit count

    const auto accessor = TensorAccessor(src_args, src_addr);

    constexpr uint32_t TILE_H = 32;

    for (uint32_t u = u_start; u < u_start + u_count; ++u) {
        const uint32_t row = u / chunks_per_row;
        const uint32_t chunk = u - row * chunks_per_row;
        dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in, dataflow_kernel_lib::TilizeGranularity::TILE>(
            accessor, TILE_H, chunk_width_bytes, row * TILE_H, chunk * chunk_width_bytes);
    }
}
