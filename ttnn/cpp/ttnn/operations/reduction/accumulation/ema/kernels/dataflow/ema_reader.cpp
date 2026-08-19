// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "../../../device/kernels/accumulation_common.hpp"

void kernel_main() {
    // Compile time args
    // -----------------
    constexpr uint32_t total_tiles_per_core = get_arg(args::total_tiles_per_core);

    // Runtime args
    // ------------
    const uint32_t src_start_tile = get_arg(args::src_start_tile);

    // Tensor accessor
    // ---------------
    const auto src_accessor = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_src(dfb::src);

    // Tile sizes
    // ----------
    const uint32_t src_tile_size = dfb_src.get_tile_size();

    //-------------------------------------------------------------------------
    // Main loop - pull pages from src and push to the src dataflow buffer
    for (uint32_t tile_id = src_start_tile; tile_id < (src_start_tile + total_tiles_per_core); ++tile_id) {
        dfb_src.reserve_back(ONE_TILE);
        noc.async_read(src_accessor, dfb_src, src_tile_size, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_src.push_back(ONE_TILE);
    }
}
