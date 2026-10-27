// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_dims = get_arg(args::num_dims);

    // Two num_dims-long common vararg blocks, in the order the host appends them: the per-dimension
    // unpadded and padded tile counts.
    constexpr uint32_t num_unpadded_tiles_base = 0;
    constexpr uint32_t num_padded_tiles_base = num_dims;

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    // In and out are assumed to be same dataformat
    const auto s0 = TensorAccessor(tensor::src);

    // Create objects for Device 2.0 API
    DataflowBuffer dfb_in0(dfb::in);
    Noc noc;

    // Get tile size from the DFB entry size
    const uint32_t tile_size = dfb_in0.get_entry_size();

    // The per-dimension index odometer this kernel advances as it walks the source. It is seeded from
    // the host's starting position for this core (the per-core vararg block) and then mutated in
    // place, so it needs writable storage of its own; vararg values are read-only.
    Scratchpad<uint32_t> id_per_dim(scratch::id_per_dim);
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    uint32_t src_tile_id = start_id;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        // Copy Input
        dfb_in0.reserve_back(1);
        noc.async_read(s0, dfb_in0, tile_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(1);
        src_tile_id++;
        for (uint32_t j = 0; j < num_dims; ++j) {
            id_per_dim[j]++;
            if (id_per_dim[j] == get_common_vararg(num_unpadded_tiles_base + j)) {
                id_per_dim[j] = 0;
                src_tile_id += get_common_vararg(num_padded_tiles_base + j);
            } else {
                break;
            }
        }
    }
}
