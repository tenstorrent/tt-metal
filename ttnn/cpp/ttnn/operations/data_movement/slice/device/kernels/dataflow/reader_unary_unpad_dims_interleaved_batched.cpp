// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tile reader with one read barrier per batch of tiles. The source tile ids still step through the
// padded dims one tile at a time (the odometer), so a batch may span a padded-dim boundary.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_dims = get_arg(args::num_dims);
    constexpr uint32_t tiles_per_batch = get_arg(args::tiles_per_batch);

    constexpr uint32_t num_unpadded_tiles_base = 0;
    constexpr uint32_t num_padded_tiles_base = num_dims;

    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    const auto s0 = TensorAccessor(tensor::src);

    DataflowBuffer dfb_in0(dfb::in);
    Noc noc;

    const uint32_t tile_size = dfb_in0.get_entry_size();

    Scratchpad<uint32_t> id_per_dim(scratch::id_per_dim);
    const auto id_per_dim_lock = id_per_dim.scoped_lock(0, num_dims);
    for (uint32_t j = 0; j < num_dims; ++j) {
        id_per_dim[j] = get_vararg(j);
    }

    uint32_t src_tile_id = start_id;

    for (uint32_t done = 0; done < num_tiles;) {
        const uint32_t remaining = num_tiles - done;
        const uint32_t n = remaining < tiles_per_batch ? remaining : tiles_per_batch;
        dfb_in0.reserve_back(n);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        for (uint32_t t = 0; t < n; ++t) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(s0, dst, tile_size, {.page_id = src_tile_id}, {.offset_bytes = 0});
            l1_write_addr += tile_size;
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
        noc.async_read_barrier();
        dfb_in0.push_back(n);
        done += n;
    }
}
