// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    // Compile time args
    // -----------------
    constexpr uint32_t total_tiles_per_core = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    // Runtime args
    // ------------
    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_start_tile = get_arg_val<uint32_t>(1);

    // CB indices
    // ----------
    constexpr auto src_cb = tt::CBIndex::c_0;

    // Tile sizes
    // ----------
    constexpr uint32_t src_tile_size = get_tile_size(src_cb);

    // Tensor accessor
    // ---------------
    const auto src_accessor = TensorAccessor(src_args, src_base_addr, src_tile_size);

    experimental::Noc noc;
    experimental::CircularBuffer cb_src(src_cb);

    //-------------------------------------------------------------------------
    // Main loop - pull pages from src and push to src_cb
    for (uint32_t tile_id = src_start_tile; tile_id < (src_start_tile + total_tiles_per_core); ++tile_id) {
        cb_src.reserve_back(1);
        noc.async_read(src_accessor, cb_src, src_tile_size, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_src.push_back(1);
    }
}
