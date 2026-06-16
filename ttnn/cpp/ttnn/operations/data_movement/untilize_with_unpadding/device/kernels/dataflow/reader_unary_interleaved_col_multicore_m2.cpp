// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_col_multicore.cpp.
// The legacy source lives outside the op directory and is shared, so it is forked here (not edited in
// place) and ported to Metal 2.0 named bindings for untilize_with_unpadding's multi-core COL
// interleaved factory. Only the access mechanism changed:
//   - the source CB id (legacy c_0) comes from the DFB producer token (dfb::in)
//   - the source address comes from the TensorAccessor binding (ta::src)
//   - num_tiles_per_2d / third_dim / number_blocks_per_core become named compile-time args (args::)
//   - core_number / tiles_per_row / num_blocks become named runtime args (args::)
// The read loops / #ifdefs / numeric paths are preserved verbatim.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t core_number = get_arg(args::core_number);
    const uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr uint32_t num_tiles_per_2d = get_arg(args::num_tiles_per_2d);
    constexpr uint32_t third_dim = get_arg(args::third_dim);
    constexpr uint32_t number_blocks_per_core = get_arg(args::number_blocks_per_core);

    const auto s = TensorAccessor(ta::src);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_in(dfb::in);

#ifdef OUT_SHARDED
    dfb_in.wait_front(onetile);
#else

    // single-tile ublocks
    const uint32_t tile_bytes = get_local_cb_interface(dfb::in).fifo_page_size;

#ifdef BACKWARDS
    uint32_t end_id = -num_tiles_per_2d;
    for (uint32_t dim = 0; dim > -third_dim; dim--) {
        for (uint32_t k = 0; k > -num_blocks; k--) {
            for (uint32_t i = num_tiles_per_2d * dim - number_blocks_per_core * core_number;
                 i > end_id + num_tiles_per_2d * dim;
                 i = i - tiles_per_row) {
#else
    uint32_t end_id = num_tiles_per_2d;
    for (uint32_t dim = 0; dim < third_dim; dim++) {
        for (uint32_t k = 0; k < num_blocks; k++) {
            for (uint32_t i = num_tiles_per_2d * dim + number_blocks_per_core * core_number;
                 i < end_id + num_tiles_per_2d * dim;
                 i = i + tiles_per_row) {
#endif
                dfb_in.reserve_back(onetile);
                noc.async_read(s, dfb_in, tile_bytes, {.page_id = static_cast<uint32_t>(i + k)}, {.offset_bytes = 0});

                noc.async_read_barrier();
                dfb_in.push_back(onetile);
            }
        }
    }
#endif
}
