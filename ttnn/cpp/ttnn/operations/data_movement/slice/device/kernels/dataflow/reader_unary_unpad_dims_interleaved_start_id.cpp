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
    constexpr auto num_dims = get_arg(args::num_dims);

    const auto start_id = get_arg(args::start_id);
    const auto num_tiles = get_arg(args::num_tiles);

    // Two num_dims-long common vararg blocks, in host push order:
    //   [0, num_dims)          num_unpadded_tiles per dim
    //   [num_dims, 2*num_dims) num_padded_tiles per dim
    constexpr uint32_t num_unpadded_tiles_base = 0;
    constexpr uint32_t num_padded_tiles_base = num_dims;

    // The per-dim walk counters are seeded by the host and advanced as this kernel walks the input,
    // so they are copied into a local: get_vararg() reads a vararg but cannot write one back.
    uint32_t id_per_dim[num_dims];
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    // In and out are assumed to be same dataformat
    const auto s0 = TensorAccessor(tensor::input);

    // Create objects for Device 2.0 API
    DataflowBuffer dfb_in0(dfb::in0);
    Noc noc;

    // Get tile size from the DFB
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
            if (id_per_dim[j] == get_common_vararg(num_unpadded_tiles_base + j)) {
                id_per_dim[j] = 0;
                src_tile_id += get_common_vararg(num_padded_tiles_base + j);
            } else {
                break;
            }
        }
    }
}
