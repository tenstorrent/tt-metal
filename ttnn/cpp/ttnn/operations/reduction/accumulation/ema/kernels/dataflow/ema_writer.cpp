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
    const uint32_t dst_start_tile = get_arg(args::dst_start_tile);

    // Tensor accessor
    // ---------------
    const auto dst_accessor = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_dst(dfb::dst);

    // Tile sizes
    // ----------
    const uint32_t dst_tile_size = dfb_dst.get_tile_size();

    //-------------------------------------------------------------------------
    // Main loop - pull pages from the dst dataflow buffer and push to dst
    for (uint32_t tile_id = dst_start_tile; tile_id < (dst_start_tile + total_tiles_per_core); ++tile_id) {
        dfb_dst.wait_front(ONE_TILE);
        noc.async_write(dfb_dst, dst_accessor, dst_tile_size, {.offset_bytes = 0}, {.page_id = tile_id});
        noc.async_write_barrier();
        dfb_dst.pop_front(ONE_TILE);
    }
}
