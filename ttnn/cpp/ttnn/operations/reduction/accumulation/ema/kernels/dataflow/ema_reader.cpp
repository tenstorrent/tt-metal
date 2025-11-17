// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

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

    //-------------------------------------------------------------------------
    // Main loop - pull pages from src and push to src_cb
    for (uint32_t tile_id = src_start_tile; tile_id < (src_start_tile + total_tiles_per_core); ++tile_id) {
        cb_reserve_back(src_cb, 1);
        const uint32_t l1_write_addr = get_write_ptr(src_cb);
        const uint64_t src_noc_addr = src_accessor.get_noc_addr(tile_id);
        noc_async_read(src_noc_addr, l1_write_addr, src_tile_size);
        noc_async_read_barrier();
        cb_push_back(src_cb, 1);
    }
}
