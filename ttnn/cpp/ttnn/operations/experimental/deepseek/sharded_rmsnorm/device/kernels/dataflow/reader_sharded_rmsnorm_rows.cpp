// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Several-row block (h x 32 tiles) reader: exposes the resident input shard, builds the hub's ones tile and
// fetches gamma as (h x 32) tiles with gamma in every row.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "sharded_rmsnorm_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_raw_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ones_id = get_compile_time_arg_val(3);
    constexpr uint32_t w_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t h = get_compile_time_arg_val(5);
    constexpr bool gamma_replicated = get_compile_time_arg_val(6) != 0;
    constexpr auto weight_args = TensorAccessorArgs<7>();

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_idx = get_arg_val<uint32_t>(1);

    constexpr uint32_t kFaceRowBytes = 16 * sizeof(uint16_t);
    constexpr uint32_t kBlockBytes = 2 * h * kFaceRowBytes;

    // 1. The input CB aliases this core's shard: publish it.
    CircularBuffer cb_x(cb_x_id);
    cb_x.reserve_back(w_tiles);
    cb_x.push_back(w_tiles);

    const auto weight_acc = TensorAccessor(weight_args, weight_addr);
    const uint32_t first_tile = core_idx * w_tiles;
    CircularBuffer cb_gamma(cb_gamma_id);
    cb_gamma.reserve_back(w_tiles);
    const uint32_t gamma_addr = cb_gamma.get_write_ptr();
    if constexpr (gamma_replicated) {
        // 2a. gamma is already [h, D] in (h x 32) tiles: one page per tile.
        for (uint32_t t = 0; t < w_tiles; ++t) {
            noc_async_read(weight_acc.get_noc_addr(first_tile + t), gamma_addr + t * kBlockBytes, kBlockBytes);
        }
    }

    // 3. The hub's ones[32, 32] (bf16), the row-sum matmul's B.
    if (core_idx == 0) {
        CircularBuffer cb_ones(cb_ones_id);
        cb_ones.reserve_back(1);
        auto* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_ones.get_write_ptr());
        for (uint32_t i = 0; i < 32 * 32 / 2; ++i) {
            ones[i] = 0x3F803F80u;
        }
        cb_ones.push_back(1);
    }

    if constexpr (gamma_replicated) {
        noc_async_read_barrier();
    } else {
        // 2b. gamma [1, D] in 32x32 tiles: fetch row 0 of each face (64-byte aligned reads), then copy each
        // 16-column half into all h rows of its face.
        CircularBuffer cb_gamma_raw(cb_gamma_raw_id);
        constexpr uint32_t kFaceBytes = 16 * kFaceRowBytes;
        constexpr uint32_t kRawChunkBytes = 2 * kFaceRowBytes;
        cb_gamma_raw.reserve_back(2 * w_tiles);
        const uint32_t raw_addr = cb_gamma_raw.get_write_ptr();
        for (uint32_t t = 0; t < w_tiles; ++t) {
            for (uint32_t face = 0; face < 2; ++face) {
                noc_async_read(
                    weight_acc.get_noc_addr(first_tile + t, face * kFaceBytes),
                    raw_addr + (2 * t + face) * kRawChunkBytes,
                    kRawChunkBytes);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t t = 0; t < w_tiles; ++t) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t src = raw_addr + (2 * t + face) * kRawChunkBytes;
                const uint32_t dst = gamma_addr + t * kBlockBytes + face * h * kFaceRowBytes;
                for (uint32_t r = 0; r < h; ++r) {
                    sharded_rmsnorm_detail::copy_l1_words(dst + r * kFaceRowBytes, src, kFaceRowBytes);
                }
            }
        }
        cb_gamma_raw.push_back(2 * w_tiles);
        cb_gamma_raw.wait_front(2 * w_tiles);
        cb_gamma_raw.pop_front(2 * w_tiles);
    }
    cb_gamma.push_back(w_tiles);
}
