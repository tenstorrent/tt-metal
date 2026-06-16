// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp,
// for the multi-core block tilize writer. The legacy source is shared with the sibling
// tilize_with_val_padding block factory, so it is forked rather than edited in place.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t start_id = get_arg(args::start_id);
    uint32_t single_block_size_row_arg = get_arg(args::single_block_size_row_arg);
    uint32_t single_block_size_col_arg = get_arg(args::single_block_size_col_arg);

    constexpr uint32_t cb_id_out = dfb::output;
    constexpr uint32_t num_tiles_per_2d = get_arg(args::num_tiles_per_2d);
    constexpr uint32_t third_dim = get_arg(args::third_dim);
    constexpr uint32_t total_tiles_per_row = get_arg(args::total_tiles_per_row);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    const auto s = TensorAccessor(ta::output);

    Noc noc;
    DataflowBuffer cb(cb_id_out);

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
                cb.wait_front(onetile);
                noc.async_write(cb, s, tile_bytes, {}, {.page_id = tile});
                noc.async_writes_flushed();
                cb.pop_front(onetile);
            }
        }
    }
    noc.async_write_barrier();
}
