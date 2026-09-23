// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of writer_unary_interleaved_start_id_wh.cpp, which lives beside it.
// Ops ported to Metal 2.0 bind the fork; the original serves the consumers still on the legacy API.
// Until the last of them migrates and the original is retired, changes here likely belong there too.
//
// The binding names below (dfb::out, tensor::dst) and the named argument set are this fork's interface:
// every later consumer inherits them, so they are taken from the kernel's own vocabulary rather than any
// one op's locals, and are not renamed once a consumer exists.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t start_id = get_arg(args::start_id);
    uint32_t single_block_size_row_arg = get_arg(args::single_block_size_row_arg);
    uint32_t single_block_size_col_arg = get_arg(args::single_block_size_col_arg);

    constexpr auto num_tiles_per_2d = get_arg(args::num_tiles_per_2d);
    constexpr auto third_dim = get_arg(args::third_dim);
    constexpr auto total_tiles_per_row = get_arg(args::total_tiles_per_row);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb(dfb::out);

    // Tile size comes off the DFB object (its entry size), not a free function keyed by buffer id.
    const uint32_t tile_bytes = dfb.get_tile_size();

#ifdef BACKWARDS
    for (uint32_t dim = 0; dim > -third_dim; dim--) {
        for (uint32_t c = 0; c > -single_block_size_col_arg; c--) {
            for (uint32_t r = 0; r > -single_block_size_row_arg; r--) {
                uint32_t tile = -start_id + dim * num_tiles_per_2d + c * total_tiles_per_row + r;
#else
    for (uint32_t dim = 0; dim < third_dim; dim++) {
        for (uint32_t c = 0; c < single_block_size_col_arg; c++) {
            for (uint32_t r = 0; r < single_block_size_row_arg; r++) {
                uint32_t tile = start_id + dim * num_tiles_per_2d + c * total_tiles_per_row + r;
#endif
                dfb.wait_front(onetile);
                noc.async_write(dfb, s, tile_bytes, {}, {.page_id = tile});
                noc.async_writes_flushed();
                dfb.pop_front(onetile);
            }
        }
    }
    noc.async_write_barrier();
}
