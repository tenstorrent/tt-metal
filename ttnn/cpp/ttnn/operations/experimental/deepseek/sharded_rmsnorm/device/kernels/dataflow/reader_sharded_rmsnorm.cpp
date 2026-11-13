// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "sharded_rmsnorm_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_raw_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_tile_out_id = get_compile_time_arg_val(5);
    constexpr uint32_t w_tiles = get_compile_time_arg_val(6);
    constexpr bool input_is_rm = get_compile_time_arg_val(7) != 0;
    constexpr bool scatter_out = get_compile_time_arg_val(8) != 0;
    constexpr auto weight_args = TensorAccessorArgs<9>();

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_idx = get_arg_val<uint32_t>(1);

    constexpr uint32_t kFaceRowBytes = 16 * sizeof(uint16_t);
    constexpr uint32_t kFaceBytes = 16 * kFaceRowBytes;
    constexpr uint32_t kRowTileBytes = 2 * kFaceRowBytes;
    constexpr uint32_t kTile32x32Bytes = 32 * 32 * sizeof(uint16_t);

    // 1. Expose input row to compute immediately so sum(x^2) overlaps gamma DRAM reads.
    CircularBuffer cb_x(cb_x_id);
    cb_x.reserve_back(w_tiles);
    if constexpr (!input_is_rm) {
        invalidate_l1_cache();
        const uint32_t in_addr = get_write_ptr(cb_in_id);
        const uint32_t x_addr = cb_x.get_write_ptr();
        for (uint32_t t = 0; t < w_tiles; ++t) {
            const uint32_t tile_base = in_addr + t * kTile32x32Bytes;
            sharded_rmsnorm_detail::copy_l1_words(x_addr + t * kRowTileBytes, tile_base, kFaceRowBytes);
            sharded_rmsnorm_detail::copy_l1_words(
                x_addr + t * kRowTileBytes + kFaceRowBytes, tile_base + kFaceBytes, kFaceRowBytes);
        }
    }
    cb_x.push_back(w_tiles);

    // 2. Fetch weight (gamma) row 0 from DRAM in 64-byte aligned chunks, then pack into 1x32 tiles.
    CircularBuffer cb_gamma_raw(cb_gamma_raw_id);
    CircularBuffer cb_gamma(cb_gamma_id);
    const uint32_t num_chunks = 2 * w_tiles;
    cb_gamma_raw.reserve_back(num_chunks);
    const uint32_t raw_addr = cb_gamma_raw.get_write_ptr();
    const auto weight_acc = TensorAccessor(weight_args, weight_addr);
    const uint32_t first_tile = core_idx * w_tiles;
    uint32_t chunk = 0;
    for (uint32_t t = 0; t < w_tiles; ++t) {
        for (uint32_t face = 0; face < 2; ++face) {
            noc_async_read(
                weight_acc.get_noc_addr(first_tile + t, face * kFaceBytes),
                raw_addr + chunk * kRowTileBytes,
                kRowTileBytes);
            ++chunk;
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();

    cb_gamma.reserve_back(w_tiles);
    const uint32_t gamma_addr = cb_gamma.get_write_ptr();
    for (uint32_t c = 0; c < num_chunks; ++c) {
        sharded_rmsnorm_detail::copy_l1_words(
            gamma_addr + c * kFaceRowBytes, raw_addr + c * kRowTileBytes, kFaceRowBytes);
    }
    cb_gamma.push_back(w_tiles);
    cb_gamma_raw.push_back(num_chunks);
    cb_gamma_raw.wait_front(num_chunks);
    cb_gamma_raw.pop_front(num_chunks);

    // 3. When outputting TILE WIDTH_SHARDED, scatter the second half of this core's tiles in parallel with writer.
    if constexpr (scatter_out) {
        constexpr uint32_t first_tile_out = w_tiles / 2;
        constexpr uint32_t num_tiles_out = w_tiles - first_tile_out;
        sharded_rmsnorm_detail::scatter_row_to_tiles(
            cb_out_id, w_tiles, cb_tile_out_id, get_write_ptr(cb_tile_out_id), first_tile_out, num_tiles_out);
    }
}
