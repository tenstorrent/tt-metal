// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of writer_unary_interleaved_start_id_wh.cpp.
//
// Bindings:
//   dfb::out                       — DFB endpoint (CONSUMER)
//   ta::out                        — TensorAccessor (output, interleaved)
//   args::num_tiles_per_2d         — CTA
//   args::third_dim                — CTA
//   args::total_tiles_per_row      — CTA
//   args::start_id                 — RTA
//   args::single_block_size_row    — RTA
//   args::single_block_size_col    — RTA

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto start_id = get_arg(args::start_id);
    auto single_block_size_row_arg = get_arg(args::single_block_size_row);
    auto single_block_size_col_arg = get_arg(args::single_block_size_col);

    constexpr auto num_tiles_per_2d = get_arg(args::num_tiles_per_2d);
    constexpr auto third_dim = get_arg(args::third_dim);
    constexpr auto total_tiles_per_row = get_arg(args::total_tiles_per_row);

    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(dfb::out);

    const auto s = TensorAccessor(ta::out);

    Noc noc;
    DataflowBuffer cb(dfb::out);

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
