
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp.
// The legacy source lives outside this op's directory (eltwise/unary) and is shared by untilize, so
// it is forked here (not edited in place) and ported to Metal 2.0 named bindings for
// untilize_with_unpadding's multi-core block-interleaved factory.
// Logic, loop bounds and #ifdefs are UNCHANGED; only the access mechanism moves to named bindings:
//   src address                -> ta::src (TensorAccessor)
//   CB id 0                     -> dfb::in
//   num_tiles_per_2d / third_dim / total_tiles_per_row CTAs -> named CTAs (get_arg(args::...))
//   start_id / single_block_size_row_arg / single_block_size_col_arg RTAs -> named RTAs
//   the src_addr RTA (slot 0) read disappears (folded into ta::src)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t single_block_size_row_arg = get_arg(args::single_block_size_row_arg);
    const uint32_t single_block_size_col_arg = get_arg(args::single_block_size_col_arg);

    constexpr uint32_t num_tiles_per_2d = get_arg(args::num_tiles_per_2d);
    constexpr uint32_t third_dim = get_arg(args::third_dim);
    constexpr uint32_t total_tiles_per_row = get_arg(args::total_tiles_per_row);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_local_cb_interface(dfb::in).fifo_page_size;

    const auto s = TensorAccessor(ta::src);

    Noc noc;
    DataflowBuffer cb(dfb::in);

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
                cb.reserve_back(onetile);
                noc.async_read(s, cb, tile_bytes, {.page_id = tile}, {.offset_bytes = 0});

                noc.async_read_barrier();
                cb.push_back(onetile);
            }
        }
    }
}
