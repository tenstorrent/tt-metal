// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_dims = get_arg(args::num_dims);

    // num_unpadded_tiles / num_padded_tiles are per-dim arrays read in the inner loop by a
    // runtime-varying index, so they arrive as common runtime varargs: [0, num_dims) is
    // num_unpadded_tiles and [num_dims, 2*num_dims) is num_padded_tiles.
    uint32_t num_unpadded_tiles[num_dims];
    uint32_t num_padded_tiles[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        num_unpadded_tiles[j] = get_common_vararg(j);
        num_padded_tiles[j] = get_common_vararg(num_dims + j);
    }

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    // id_per_dim is a per-core array advanced in the inner loop by a runtime-varying index → runtime varargs.
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    // In and out are assumed to be same dataformat
    const auto s0 = TensorAccessor(tensor::in);

    // Create objects for Device 2.0 API
    DataflowBuffer cb_in0(dfb::cb_in);
    Noc noc;

    // Get tile size from DFB interface
    const uint32_t tile_size = cb_in0.get_entry_size();

    uint32_t src_tile_id = start_id;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        // Copy Input
        cb_in0.reserve_back(1);
        noc.async_read(s0, cb_in0, tile_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(1);
        src_tile_id++;
        for (uint32_t j = 0; j < num_dims; ++j) {
            id_per_dim[j]++;
            if (id_per_dim[j] == num_unpadded_tiles[j]) {
                id_per_dim[j] = 0;
                src_tile_id += num_padded_tiles[j];
            } else {
                break;
            }
        }
    }
}
