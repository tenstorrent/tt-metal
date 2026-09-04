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

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    // Per-dimension tile counts of the slice (unpadded) and of the gap to skip at the end of each
    // dimension (padded), one entry per dimension. Both ride the common vararg block, unpadded
    // first: get_common_vararg(j) is dimension j's unpadded count, get_common_vararg(num_dims + j)
    // its padded count.

    // This node's per-dimension index vector, copied out of the runtime vararg block because the
    // walk below advances it like an odometer and varargs are read-only values. The host supplies a
    // fresh block on every dispatch, so nothing is carried across enqueues by keeping it local.
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    // In and out are assumed to be same dataformat
    const auto s0 = TensorAccessor(tensor::src);

    // Create objects for Device 2.0 API
    // dfb_in0 stages tiles for the writer to drain; the host binds this kernel as its producer.
    DataflowBuffer dfb_in0(dfb::in0);
    Noc noc;

    // Get tile size from the DFB entry size
    const uint32_t tile_size = dfb_in0.get_entry_size();

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
            if (id_per_dim[j] == get_common_vararg(j)) {
                id_per_dim[j] = 0;
                src_tile_id += get_common_vararg(num_dims + j);
            } else {
                break;
            }
        }
    }
}
