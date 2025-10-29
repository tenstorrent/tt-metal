// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    // Compile time args
    // -----------------
    constexpr auto total_channels_per_core = get_compile_time_arg_val(0);
    constexpr auto tiles_per_channel = get_compile_time_arg_val(1);
    constexpr uint32_t tile_stride = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    // Runtime args
    // ------------
    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_tile_offset = get_arg_val<uint32_t>(1);

    // CB indices
    // ----------
    constexpr auto src_cb = tt::CBIndex::c_0;

    // Tile sizes
    constexpr uint32_t src_tile_size = get_tile_size(src_cb);

    // Tensor accessor
    // ---------------
    const auto src_accessor = TensorAccessor(src_args, src_base_addr, src_tile_size);

    //-------------------------------------------------------------------------
    // Main loop - pull pages from src and push to src_cb
    uint32_t src_start_tile = src_tile_offset;
    for (uint32_t channel_id = 0; channel_id < total_channels_per_core; ++channel_id) {
        uint32_t tile_offset = src_start_tile;
        for (uint32_t tile_id = 0; tile_id < tiles_per_channel; ++tile_id) {
            cb_reserve_back(src_cb, 1);
            const uint32_t l1_write_addr = get_write_ptr(src_cb);
            const uint64_t src_noc_addr = src_accessor.get_noc_addr(tile_offset);
            noc_async_read(src_noc_addr, l1_write_addr, src_tile_size);
            noc_async_read_barrier();
            cb_push_back(src_cb, 1);
            tile_offset += tile_stride;
        }
        ++src_start_tile;
    }
}
