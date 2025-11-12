// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    // Compile time args
    // -----------------
    constexpr uint32_t total_tiles_per_core = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    // Runtime args
    // ------------
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_start_tile = get_arg_val<uint32_t>(1);

    // CB indices
    // ----------
    constexpr auto dst_cb = tt::CBIndex::c_1;

    // Tile sizes
    // ----------
    constexpr uint32_t dst_tile_size = get_tile_size(dst_cb);

    // Tensor accessor
    // ---------------
    const auto dst_accessor = TensorAccessor(dst_args, dst_base_addr, dst_tile_size);

    //-------------------------------------------------------------------------
    // Main loop - pull pages from dst_cb and push to dst
    for (uint32_t tile_id = dst_start_tile; tile_id < (dst_start_tile + total_tiles_per_core); ++tile_id) {
        cb_wait_front(dst_cb, 1);
        const uint32_t l1_read_addr = get_read_ptr(dst_cb);
        const uint64_t dst_noc_addr = dst_accessor.get_noc_addr(tile_id);
        noc_async_write(l1_read_addr, dst_noc_addr, dst_tile_size);
        noc_async_write_barrier();
        cb_pop_front(dst_cb, 1);
    }
}
