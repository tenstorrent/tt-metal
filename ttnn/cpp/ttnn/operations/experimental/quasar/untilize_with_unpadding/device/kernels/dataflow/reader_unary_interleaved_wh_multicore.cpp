
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // The input base address is carried by the TensorAccessor binding; the legacy src_addr runtime
    // arg is gone.
    uint32_t start_id = get_arg(args::start_id);
    uint32_t single_block_size_row_arg = get_arg(args::single_block_size_row_arg);
    uint32_t single_block_size_col_arg = get_arg(args::single_block_size_col_arg);

    constexpr uint32_t num_tiles_per_2d = get_arg(args::num_tiles_per_2d);
    constexpr uint32_t third_dim = get_arg(args::third_dim);
    constexpr uint32_t total_tiles_per_row = get_arg(args::total_tiles_per_row);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer cb(dfb::in);
    const uint32_t tile_bytes = cb.get_entry_size();

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
