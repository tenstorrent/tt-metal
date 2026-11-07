// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

namespace ttnn::experimental::prim {

// Picks default block and subblock sizes (in tiles) when no config is given.
// Starts from 8x8x8 blocks and shrinks them only when the dataflow buffers of that
// size would not fit in L1 for the given tile sizes (bytes). K_block is reduced first,
// then the larger of the M and N blocks, never below the subblock size.
inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> determine_default_block_sizes(
    uint32_t M,
    uint32_t K,
    uint32_t N,
    bool fp32_dest_acc_en,
    uint32_t in0_tile_size,
    uint32_t in1_tile_size,
    uint32_t out_tile_size,
    uint32_t in2_tile_size,
    uint32_t intermediate_tile_size) {
    (void)K;
    uint32_t M_block_tiles = 8;
    uint32_t K_block_tiles = 8;
    uint32_t N_block_tiles = 8;

    uint32_t subblock_h = 2;
    uint32_t subblock_w = 2;
    if (!fp32_dest_acc_en) {
        if (N >= M) {
            subblock_h = 2;
            subblock_w = 4;
        } else {
            subblock_h = 4;
            subblock_w = 2;
        }
    }

    // Conservative budget for the matmul dataflow buffers; leaves room for the other
    // statically allocated L1 on the core.
    constexpr uint64_t l1_budget_bytes = 1380000;
    // in0, in1 and out are double buffered; intermediate and bias are not.
    auto footprint = [&](uint32_t m, uint32_t k, uint32_t n) -> uint64_t {
        return 2ULL * m * k * in0_tile_size + 2ULL * k * n * in1_tile_size + 2ULL * m * n * out_tile_size +
               1ULL * m * n * intermediate_tile_size + 1ULL * n * in2_tile_size;
    };

    while (footprint(M_block_tiles, K_block_tiles, N_block_tiles) > l1_budget_bytes && K_block_tiles > 1) {
        K_block_tiles /= 2;
    }
    while (footprint(M_block_tiles, K_block_tiles, N_block_tiles) > l1_budget_bytes) {
        const bool can_reduce_m = M_block_tiles > subblock_h;
        const bool can_reduce_n = N_block_tiles > subblock_w;
        if (!can_reduce_m && !can_reduce_n) {
            break;
        }
        if (can_reduce_m && (M_block_tiles >= N_block_tiles || !can_reduce_n)) {
            M_block_tiles /= 2;
        } else {
            N_block_tiles /= 2;
        }
    }

    return {M_block_tiles, K_block_tiles, N_block_tiles, subblock_h, subblock_w};
}

}  // namespace ttnn::experimental::prim
