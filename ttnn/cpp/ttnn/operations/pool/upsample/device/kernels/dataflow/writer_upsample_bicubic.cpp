// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_pixels = get_arg_val<uint32_t>(1);
    const uint32_t start_pixel_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t aligned_output_stick_nbytes = get_compile_time_arg_val(1);
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(2);
    constexpr uint32_t blocks = get_compile_time_arg_val(3);

    constexpr auto dst_args = TensorAccessorArgs<4>();
    const auto output_tensor_accessor = TensorAccessor(dst_args, output_buffer_addr, aligned_output_stick_nbytes);

    constexpr uint32_t tile_width_bytes = 32 * 2;  // 32 BF16 = 64 bytes

    // Match compute kernel's blocking: tiles_per_block = min(in_ntiles_c, MAX_TILES_PER_REDUCTION)
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    constexpr uint32_t max_tiles_per_block =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t last_block_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_block : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    uint32_t pixel_id = start_pixel_id;
    for (uint32_t p = 0; p < num_output_pixels; p++) {
        uint64_t dst_noc_addr = output_tensor_accessor.get_noc_addr(pixel_id);

        // Consume output per-block, matching compute's tile push count
        uint32_t dram_offset = 0;
        for (uint32_t blk = 0; blk < blocks; blk++) {
            uint32_t tiles_this_block = (blk == blocks - 1) ? last_block_tiles : max_tiles_per_block;

            cb_wait_front(output_cb, tiles_this_block);
            uint32_t l1_read_addr = get_read_ptr(output_cb);

            for (uint32_t t = 0; t < tiles_this_block; t++) {
                noc_async_write(l1_read_addr, dst_noc_addr + dram_offset, tile_width_bytes);
                l1_read_addr += tile_width_bytes;
                dram_offset += tile_width_bytes;
            }
            noc_async_write_barrier();

            cb_pop_front(output_cb, tiles_this_block);
        }

        pixel_id++;
    }
}
