// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 weight-server page runs: the DRAM reads of one slot range of a weight round, shared by the server reader
// (every round) and a dual-source partner's writer (its part of round 0, conv2d_3x3_weight_server_writer.cpp).
//
// The K-block's tiles are consecutive pages (bn is the whole C_out on this path); kw-fold's round of column kw reads
// the prepared tile ((kh * 3 + kw) C_in + c, n) for slot (kh, c, n). kw-fold (conv2d_3x3_kw_fold.hpp): round kb is
// n-group kb / 3 of rw output column tiles, weight column kw = (1, 0, 2)[kb % 3], [3 C_in, rw] in (kh, c, n) order
// (block_k_tiles = C_in, block_n_tiles = 3 rw: the same round size). kwf_arg: 0 off kw-fold, else 1 + the fused side
// rounds' n-group width (0: not fused; kw_fold.hpp): rounds 1 and 2 of an n-group set then hold the kw = 0 | 2 pair
// interleaved per (kh, c) row of 2 rw tiles, each gw-wide n-group's kw = 0 columns, then its kw = 2 columns.
// ND-sharded weights (conv2d_3x3.weight_memory_config: shards of shard_rows tile-rows x shard_cols columns,
// round-robin over the DRAM banks; shard_rows 0 = interleaved, one read per tile): every run of slot tiles that is
// contiguous both in the slot and in one shard is one read.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace conv2d_3x3 {

template <
    uint32_t block_k_tiles,
    uint32_t block_n_tiles,
    uint32_t num_k_blocks,
    uint32_t kwf_arg,
    uint32_t tensor_n_tiles,
    uint32_t shard_rows,
    uint32_t shard_cols>
struct ServerPages {
    static constexpr bool kw_fold = kwf_arg != 0;
    static constexpr uint32_t kwf_fused_gw = kw_fold ? kwf_arg - 1 : 0;
    static constexpr uint32_t round_tiles = block_k_tiles * block_n_tiles;
    // Guarded: this branch is compiled on every shape (block_n_tiles may be < 3 off kw-fold).
    static constexpr uint32_t rw = block_n_tiles >= 3 ? block_n_tiles / 3 : 1;
    // A slot holds rows of row_tiles tiles: (kh, c) rows of one n-group on kw-fold, K-block rows of all C_out
    // otherwise.
    static constexpr uint32_t row_tiles = kw_fold ? rw : block_n_tiles;

    // Prepared page of slot tile t of round r.
    static FORCE_INLINE uint32_t page_of(uint32_t r, uint32_t t) {
        const uint32_t kb = r % num_k_blocks;
        if constexpr (kw_fold) {
            if constexpr (kwf_fused_gw != 0) {
                const uint32_t col = kb % 3;
                if (col != 0) {
                    constexpr uint32_t frow = 2 * rw;
                    const uint32_t f = (col - 1) * round_tiles + t;
                    const uint32_t kc = f / frow;  // kh * C_in + c
                    const uint32_t j = f - kc * frow;
                    const uint32_t grp = j / (2 * kwf_fused_gw);
                    const uint32_t jj = j - grp * 2 * kwf_fused_gw;
                    const uint32_t kh = kc / block_k_tiles;
                    const uint32_t kwf = jj < kwf_fused_gw ? 0 : 2;
                    const uint32_t n = grp * kwf_fused_gw + (jj < kwf_fused_gw ? jj : jj - kwf_fused_gw);
                    return ((kh * 3 + kwf) * block_k_tiles + (kc - kh * block_k_tiles)) * tensor_n_tiles +
                           (kb / 3) * rw + n;
                }
            }
            const uint32_t n = t % rw;
            const uint32_t kc = t / rw;
            const uint32_t kh = kc / block_k_tiles;
            const uint32_t col = kb % 3;
            const uint32_t kw = col == 0 ? 1 : (col == 1 ? 0 : 2);  // kw_fold.hpp KW_ROUND_ORDER
            return ((kh * 3 + kw) * block_k_tiles + (kc - kh * block_k_tiles)) * tensor_n_tiles + (kb / 3) * rw + n;
        } else {
            return kb * round_tiles + t;
        }
    }

    // Slot tiles [t, end) of round r into the slot at l1, as runs. From slot tile t the run goes to the edge of its
    // page's shard row (within the slot row); while a slot row is exactly one shard row, on through the shard's next
    // rows, which the next slot rows hold (kw-fold: up to the end of the kh block; K-block: of the round).
    template <typename Reads, typename Accessor>
    static FORCE_INLINE void read(
        Reads& reads, const Accessor& weights, uint32_t tile_bytes, uint32_t r, uint32_t l1, uint32_t t, uint32_t end) {
        const uint32_t kb = r % num_k_blocks;
        while (t < end) {
            const uint32_t page = page_of(r, t);
            uint32_t len = 1;
            if constexpr (shard_rows > 0 && kwf_fused_gw != 0) {
                if (kb % 3 != 0) {
                    // Fused side round: a run is one n-group's kw = 0 (or kw = 2) columns of one (kh, c) row.
                    constexpr uint32_t gw = kwf_fused_gw != 0 ? kwf_fused_gw : 1;
                    static_assert(shard_cols % gw == 0);
                    len = gw - ((kb % 3 - 1) * round_tiles + t) % gw;
                    if (len > end - t) {
                        len = end - t;
                    }
                    reads.read(weights.get_noc_addr(page), l1 + t * tile_bytes, len * tile_bytes);
                    t += len;
                    continue;
                }
            }
            if constexpr (shard_rows > 0) {
                const uint32_t i = t / row_tiles;
                const uint32_t n = t - i * row_tiles;
                const uint32_t row = page / tensor_n_tiles;
                const uint32_t c = page - row * tensor_n_tiles;
                len = shard_cols - c % shard_cols;
                if (len > row_tiles - n) {
                    len = row_tiles - n;
                }
                if (row_tiles == shard_cols && (c - n) % shard_cols == 0 && n + len == row_tiles) {
                    const uint32_t in_shard = shard_rows - 1 - row % shard_rows;
                    const uint32_t follow = block_k_tiles - 1 - (kw_fold ? i % block_k_tiles : i);
                    len += (in_shard < follow ? in_shard : follow) * row_tiles;
                }
                if (len > end - t) {
                    len = end - t;
                }
            }
            reads.read(weights.get_noc_addr(page), l1 + t * tile_bytes, len * tile_bytes);
            t += len;
        }
    }
};

}  // namespace conv2d_3x3
