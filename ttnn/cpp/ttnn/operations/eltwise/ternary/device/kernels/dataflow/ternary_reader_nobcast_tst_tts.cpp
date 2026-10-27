// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb0(dfb::in_a);
    DataflowBuffer dfb1(dfb::in_b);

#if SRC_SHARDED_A
    dfb0.reserve_back(num_tiles);
    dfb0.push_back(num_tiles);
#endif
#if SRC_SHARDED_B
    dfb1.reserve_back(num_tiles);
    dfb1.push_back(num_tiles);
#endif
#if !SRC_SHARDED_A || !SRC_SHARDED_B
#if !SRC_SHARDED_A
    const uint32_t tile_bytes_0 = dfb0.get_entry_size();
    const auto s0 = TensorAccessor(tensor::in_a);
#endif
#if !SRC_SHARDED_B
    const uint32_t tile_bytes_1 = dfb1.get_entry_size();
    const auto s1 = TensorAccessor(tensor::in_b);
#endif

    constexpr uint32_t onetile = 1;

    for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
#if !SRC_SHARDED_A
        dfb0.reserve_back(onetile);
        noc.async_read(s0, dfb0, tile_bytes_0, {.page_id = tile_id}, {.offset_bytes = 0});
#endif
#if !SRC_SHARDED_B
        dfb1.reserve_back(onetile);
        noc.async_read(s1, dfb1, tile_bytes_1, {.page_id = tile_id}, {.offset_bytes = 0});
#endif

        noc.async_read_barrier();

#if !SRC_SHARDED_A
        dfb0.push_back(onetile);
#endif
#if !SRC_SHARDED_B
        dfb1.push_back(onetile);
#endif
    }
#endif
}
