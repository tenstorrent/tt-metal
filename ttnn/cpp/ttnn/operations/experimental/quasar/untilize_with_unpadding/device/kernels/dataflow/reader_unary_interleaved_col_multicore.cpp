// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port. Identical read pattern to the legacy col-multicore reader; the CB,
// tensor accessor, and arguments are sourced from Metal 2.0 named bindings (dfb::in /
// tensor::input) and named args (args::*) instead of positional CTAs/RTAs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t core_number = get_arg(args::core_number);
    uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr uint32_t num_tiles_per_2d = get_arg(args::num_tiles_per_2d);
    constexpr uint32_t third_dim = get_arg(args::third_dim);
    constexpr uint32_t number_blocks_per_core = get_arg(args::number_blocks_per_core);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer cb(dfb::in);

#ifdef OUT_SHARDED
    cb.wait_front(onetile);
#else

    // single-tile ublocks
    const uint32_t tile_bytes = get_local_cb_interface(dfb::in).fifo_page_size;

    const auto s = TensorAccessor(tensor::input);

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
                cb.reserve_back(onetile);
                noc.async_read(s, cb, tile_bytes, {.page_id = static_cast<uint32_t>(i + k)}, {.offset_bytes = 0});

                noc.async_read_barrier();
                cb.push_back(onetile);
            }
        }
    }
#endif
}
