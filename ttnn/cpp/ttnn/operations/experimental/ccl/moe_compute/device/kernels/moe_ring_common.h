// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "../hostdevcommon/config.hpp"

namespace moe_ring {

namespace detail {
constexpr uint32_t div_up(const uint32_t a, const uint32_t b) { return (a + b - 1) / b; }

template <uint32_t a, uint32_t b>
constexpr uint32_t div_up() {
    return (a + b - 1) / b;
}

}  // namespace detail

constexpr uint32_t W0_W1_TXNS_PER_BLOCK = 2;
// The DRAM transaction size in tiles is a per-shape compile-time parameter (tiles_per_txn_for_shape below; the host
// passes it to the kernels as the "tiles_per_txn" named compile arg and they build MoeRingConfig with it). Every block
// height derives from it (MoeRingConfig::block_tiles_h / half_block_tiles_h): there are deliberately no fixed
// 14-tile *_TILES_H / *_TILES_PER_TXN constants, so code written for the fixed geometry fails to compile rather than
// read 28-tile blocks from a 20-tile stream.
constexpr uint32_t DEFAULT_TILES_PER_TXN = 14;  // 8,064 B BF4; 28-tile blocks, 4 wide x 7 K rows
constexpr uint32_t ALT_TILES_PER_TXN = 20;      // 11,520 B BF4 (one NoC packet); 40-tile blocks, 4 wide x 10 K rows

// probably don't need this for W0_W1 and W2 because it's the same
constexpr uint32_t W0_W1_BLOCK_TILES_W = 4;

// Half block-column: the odd gate/up column of a ring core that owns an odd column count is stored as
// (W0 c, W1 c) only. A block is still 2 transactions but 2 tiles wide and twice as many K rows high; each
// stored 4-tile row holds two consecutive K rows (W0 k, W1 k, W0 k+1, W1 k+1).
constexpr uint32_t W0_W1_HALF_BLOCK_TILES_W = W0_W1_BLOCK_TILES_W / 2;

// Let's call this a constant
constexpr uint32_t W2_TILES_PER_A2A_ITER_W = 4;
// Half-width last a2a iteration (ALT_TILES_PER_TXN only, when every core has 1 or 2 output tiles left for it):
// 2 output tiles wide, twice the K rows per block, two consecutive K rows per stored 4-tile row.
constexpr uint32_t W2_HALF_A2A_ITER_TILES_W = W2_TILES_PER_A2A_ITER_W / 2;

//-----------------------------------------------------------------------------
// Shard distribution functions (hardware-agnostic).
// Identical to the Python equivalents in ttnn/ttnn/_experimental/moe_compute_utils.py.
//-----------------------------------------------------------------------------

constexpr bool is_big_w0w1(uint32_t core_id, uint32_t n_big, uint32_t n_cores) {
    return n_big > 0 && (core_id * n_big) % n_cores < n_big;
}

constexpr uint32_t shard_tiles(uint32_t n_tiles, uint32_t core_id, uint32_t n_cores) {
    const uint32_t n_big = n_tiles % n_cores;
    const uint32_t small = n_tiles / n_cores;
    return small + (is_big_w0w1(core_id, n_big, n_cores) ? 1u : 0u);
}

constexpr uint32_t w2_shard_tiles(uint32_t Ht, uint32_t core_id, uint32_t Nt, uint32_t n_cores) {
    const uint32_t n_big_nt = Nt % n_cores;
    const uint32_t n_big_ht = Ht % n_cores;
    const uint32_t small_ht = Ht / n_cores;
    if (n_big_nt + n_big_ht == n_cores) {
        return is_big_w0w1(core_id, n_big_nt, n_cores) ? small_ht : small_ht + 1u;
    }
    return shard_tiles(Ht, core_id, n_cores);
}

template <uint32_t N>
struct ShardLUT {
    uint32_t data[N];
    constexpr uint32_t operator[](uint32_t i) const { return data[i]; }
};

template <uint32_t n_tiles, uint32_t n_cores>
constexpr ShardLUT<n_cores> make_shard_lut() {
    ShardLUT<n_cores> lut{};
    for (uint32_t c = 0; c < n_cores; ++c) {
        lut.data[c] = shard_tiles(n_tiles, c, n_cores);
    }
    return lut;
}

constexpr uint32_t even_stride_at_least_a2a_width(uint32_t tiles) {
    const uint32_t even_tiles = (tiles + 1) & ~1u;
    return even_tiles < W2_TILES_PER_A2A_ITER_W ? W2_TILES_PER_A2A_ITER_W : even_tiles;
}

// Per-shape choice of the W0/W1 column layout. Compact (each ring core stores only its own shard_tiles(Nt, c)
// columns) when that shortens the critical path: the busiest core owns fewer columns than the uniform even stride
// (1-3 columns, or an odd count). Otherwise (e.g. 6/5 or 8/7 columns) the busiest core does the same work either way
// and every core keeps the uniform stride, byte-identical to the per-core stride layout: compacting such shapes only
// added cross-bank reads and half-width passes (GLM-4.5-Air 4096/1408, Nemotron-3-Nano 2688/1856 measured ~1 % slower).
constexpr bool w0_w1_compact_for_shape(uint32_t Nt, uint32_t n_cores) {
    const uint32_t max_cols = (Nt + n_cores - 1) / n_cores;
    return max_cols < even_stride_at_least_a2a_width(max_cols);
}

// Gate/up columns ring core c stores (and produces for a routed expert): its own columns when compact, else the
// uniform stride (its own columns followed by zero columns).
constexpr uint32_t w0_w1_stored_cols(uint32_t Nt, uint32_t core_id, uint32_t n_cores) {
    return w0_w1_compact_for_shape(Nt, n_cores) ? shard_tiles(Nt, core_id, n_cores)
                                                : even_stride_at_least_a2a_width((Nt + n_cores - 1) / n_cores);
}

// Width in tiles of the in2 slice every ring core hands to the a2a ring: with the compact layout the largest per-core
// column count (at least 2), else the uniform stride.
constexpr uint32_t a2a_exchange_tiles(uint32_t Nt, uint32_t n_cores) {
    const uint32_t max_cols = (Nt + n_cores - 1) / n_cores;
    if (!w0_w1_compact_for_shape(Nt, n_cores)) {
        return even_stride_at_least_a2a_width(max_cols);
    }
    return max_cols < 2 ? 2 : max_cols;
}

// Entry c = block offset of ring core c's compact W0/W1 slice in one (layer, expert) stream; entry n_cores = the
// stream's total blocks. Cfg is a MoeRingConfig.
template <typename Cfg, uint32_t n_cores>
constexpr ShardLUT<n_cores + 1> make_w0_w1_block_offset_lut() {
    ShardLUT<n_cores + 1> lut{};
    for (uint32_t c = 0; c <= n_cores; ++c) {
        lut.data[c] = Cfg::w0_w1_core_block_offset(c);
    }
    return lut;
}

template <uint32_t Ht, uint32_t Nt, uint32_t n_cores>
constexpr ShardLUT<n_cores> make_w2_shard_lut() {
    ShardLUT<n_cores> lut{};
    for (uint32_t c = 0; c < n_cores; ++c) {
        lut.data[c] = w2_shard_tiles(Ht, c, Nt, n_cores);
    }
    return lut;
}

template <uint32_t Ht, uint32_t Nt, uint32_t n_cores>
constexpr ShardLUT<n_cores> make_w2_offset_lut() {
    ShardLUT<n_cores> lut{};
    uint32_t offset = 0;
    for (uint32_t c = 0; c < n_cores; ++c) {
        lut.data[c] = offset;
        offset += w2_shard_tiles(Ht, c, Nt, n_cores);
    }
    return lut;
}

// Compact W0/W1 layout helpers (see MoeRingConfig): blocks for `cols` gate/up columns, and the block offset of
// ring core `core_id`'s slice when all cores' slices of one (layer, expert) are laid back to back.
constexpr uint32_t w0_w1_blocks_for_cols(uint32_t cols, uint32_t blocks_per_col, uint32_t blocks_per_half_col) {
    return (cols / 2) * blocks_per_col + (cols % 2) * blocks_per_half_col;
}

constexpr uint32_t w0_w1_core_block_offset(
    uint32_t Nt, uint32_t core_id, uint32_t n_cores, uint32_t blocks_per_col, uint32_t blocks_per_half_col) {
    uint32_t offset = 0;
    for (uint32_t c = 0; c < core_id; ++c) {
        offset += w0_w1_blocks_for_cols(w0_w1_stored_cols(Nt, c, n_cores), blocks_per_col, blocks_per_half_col);
    }
    return offset;
}

// Block geometry for a transaction size of tiles_per_txn tiles: a block is W0_W1_TXNS_PER_BLOCK transactions,
// 4 tiles wide (W0/W1 block-column, W2 a2a iteration) or 2 tiles wide (half block-column, half a2a iteration).
constexpr uint32_t block_tiles_h(uint32_t tiles_per_txn) {
    return W0_W1_TXNS_PER_BLOCK * tiles_per_txn / W0_W1_BLOCK_TILES_W;
}
constexpr uint32_t half_block_tiles_h(uint32_t tiles_per_txn) {
    return W0_W1_TXNS_PER_BLOCK * tiles_per_txn / W0_W1_HALF_BLOCK_TILES_W;
}

// The per-shape DRAM transaction size (tiles) of both weight streams. 20-tile transactions store the
// Qwen3.8-Flash-Next expert (hidden 2560 = 80 tiles, intermediate 640 = 20 tiles, no bias) on the 8-bank ring, where
// the layout has no padding at all (gate/up K 80 = 8 x 10 and 4 x 20, W2 K 20 = 2 x 10 and 1 x 20; 80 blocks =
// 10 per bank; W2 a2a iterations 4 + 4 + a half). Other rings and every other shape keep the 14-tile layout: 14
// tiles is one 8 KB Wormhole NoC packet, and the 20-tile blocks with padded bank pieces and no half-width W2
// iteration (12 cores: 7 blocks per bank + 4 pad, W2 4 + 3) produced wrong outputs on a Wormhole chip. Mirrored by
// moe_compute_utils.py::_tiles_per_txn.
constexpr uint32_t tiles_per_txn_for_shape(uint32_t Ht, uint32_t Nt, bool has_bias, uint32_t num_cores) {
    return (Ht == 80 && Nt == 20 && !has_bias && num_cores == 8) ? ALT_TILES_PER_TXN : DEFAULT_TILES_PER_TXN;
}

// Blocks the weight CB (c_3) holds: as many as fit in the 3-block budget of 14-tile transactions (84 tiles), and at
// least 3 (3 for 14-tile and for 20-tile transactions, 4 for 10). dm0 keeps all but one of them in flight as DRAM
// reads, so smaller transactions keep about the same bytes in flight.
constexpr uint32_t weight_cb_slots(uint32_t tiles_per_txn) {
    const uint32_t fit = (3 * W0_W1_TXNS_PER_BLOCK * DEFAULT_TILES_PER_TXN) / (W0_W1_TXNS_PER_BLOCK * tiles_per_txn);
    return fit < 3 ? 3 : fit;
}

// Blocks one DRAM bank holds per (layer, expert) in the compact W0/W1 layout, for a stored K height of
// k_dram_tiles (hidden tiles, plus one with bias): the cores' slices back to back, cut into num_banks equal
// pieces of whole blocks. Runtime form of MoeRingConfig::w0_w1_bank_blocks_per_expert (host side).
constexpr uint32_t w0_w1_bank_blocks_per_expert(
    uint32_t k_dram_tiles, uint32_t Nt, uint32_t n_cores, uint32_t num_banks, uint32_t tiles_per_txn) {
    const uint32_t blocks_per_col = detail::div_up(k_dram_tiles, block_tiles_h(tiles_per_txn));
    const uint32_t blocks_per_half_col = detail::div_up(k_dram_tiles, half_block_tiles_h(tiles_per_txn));
    const uint32_t expert_blocks = w0_w1_core_block_offset(Nt, n_cores, n_cores, blocks_per_col, blocks_per_half_col);
    return detail::div_up(expert_blocks, num_banks);
}

// W2 a2a iterations: ceil(max W2 output tiles per core / 4). With ALT_TILES_PER_TXN the last one is half width
// when it has 1 or 2 output tiles; 14-tile layouts keep 4-wide iterations only (unchanged).
constexpr uint32_t w2_num_a2a_iters(uint32_t Ht, uint32_t n_cores) {
    return detail::div_up(detail::div_up(Ht, n_cores), W2_TILES_PER_A2A_ITER_W);
}
constexpr bool w2_last_a2a_iter_half(uint32_t Ht, uint32_t n_cores, uint32_t tiles_per_txn) {
    const uint32_t last = detail::div_up(Ht, n_cores) % W2_TILES_PER_A2A_ITER_W;
    return tiles_per_txn != DEFAULT_TILES_PER_TXN && last != 0 && last <= W2_HALF_A2A_ITER_TILES_W;
}

// W2 blocks one ring core reads per (layer, expert): each full a2a iteration is ceil(K / block_tiles_h) blocks of
// 4 x block_tiles_h, a half last iteration ceil(K / half_block_tiles_h) blocks of 2 x half_block_tiles_h, with
// K = w2_dram_tiles_h (intermediate tiles, plus one with bias).
constexpr uint32_t w2_core_blocks_per_expert(
    uint32_t Ht, uint32_t w2_dram_tiles_h, uint32_t n_cores, uint32_t tiles_per_txn) {
    const uint32_t iters = w2_num_a2a_iters(Ht, n_cores);
    const uint32_t half = w2_last_a2a_iter_half(Ht, n_cores, tiles_per_txn) ? 1u : 0u;
    return (iters - half) * detail::div_up(w2_dram_tiles_h, block_tiles_h(tiles_per_txn)) +
           half * detail::div_up(w2_dram_tiles_h, half_block_tiles_h(tiles_per_txn));
}

//-----------------------------------------------------------------------------
// Derived ring constants — single source of truth for compute, dm0, dm1.
//-----------------------------------------------------------------------------
// TilesPerTxn: the kernels pass their "tiles_per_txn" named compile arg (tiles_per_txn_for_shape). The block
// geometry, the W0/W1 block counts and the W2 block counts are only valid for a Cfg built with it; in2_tiles_per_step,
// num_a2a_iters and w2_tiles_per_expert_w do not depend on it.
template <
    uint32_t Ht,
    uint32_t Nt,
    uint32_t num_cores,
    bool has_bias,
    uint32_t SharedExpertTp = 1,
    uint32_t TilesPerTxn = DEFAULT_TILES_PER_TXN>
struct MoeRingConfig {
    // DRAM transaction geometry (both weight streams): a block is 2 transactions of tiles_per_txn tiles,
    // consumed 4 wide x block_tiles_h high (or 2 wide x half_block_tiles_h high for the half variants).
    static constexpr uint32_t tiles_per_txn = TilesPerTxn;
    static constexpr uint32_t txns_per_block = W0_W1_TXNS_PER_BLOCK;
    static constexpr uint32_t tiles_per_block = txns_per_block * tiles_per_txn;        // 28 (14-tile txn) or 40 (20)
    static constexpr uint32_t block_tiles_h = moe_ring::block_tiles_h(tiles_per_txn);  // 7 or 10
    static constexpr uint32_t half_block_tiles_h = moe_ring::half_block_tiles_h(tiles_per_txn);  // 14 or 20
    static_assert(tiles_per_block % W0_W1_BLOCK_TILES_W == 0, "a block must be whole 4-tile rows");
    static constexpr uint32_t weight_cb_slots = moe_ring::weight_cb_slots(tiles_per_txn);  // 3 (14- or 20-tile)

    // W0/W1
    static constexpr uint32_t w0_w1_dram_tiles_h = has_bias ? Ht + 1 : Ht;
    static constexpr uint32_t w0_w1_blocks_per_col = (w0_w1_dram_tiles_h + block_tiles_h - 1) / block_tiles_h;
    static constexpr uint32_t in2_tiles_per_step = a2a_exchange_tiles(Nt, num_cores);

    // W0/W1 layout: ring core c stores w0_w1_stored_cols(Nt, c) columns (its own shard_tiles(Nt, c) columns for a
    // compact shape, else the uniform stride) -- floor(cols / 2) block-columns of w0_w1_blocks_per_col blocks
    // (4 wide x block_tiles_h high) and, for an odd count, one half block-column of w0_w1_blocks_per_half_col blocks
    // (2 wide x half_block_tiles_h high), in that order. The in2 slice it hands to the a2a ring is in2_tiles_per_step
    // (a2a_exchange_tiles) wide; tiles past its own columns are never read (the W2 walk takes shard_tiles(src) tiles
    // from each source's slice).
    static constexpr uint32_t w0_w1_blocks_per_half_col =
        (w0_w1_dram_tiles_h + half_block_tiles_h - 1) / half_block_tiles_h;
    // Block offset of core c's slice inside one (layer, expert) stream (the cores' slices back to back).
    static constexpr uint32_t w0_w1_core_block_offset(uint32_t core_id) {
        return moe_ring::w0_w1_core_block_offset(
            Nt, core_id, num_cores, w0_w1_blocks_per_col, w0_w1_blocks_per_half_col);
    }
    // The (layer, expert) stream is cut into num_banks equal pieces of whole blocks (zero-padded at the end);
    // piece b is stored in bank b, so every bank holds the same bytes per expert and core c reads its slice
    // from its own bank plus, where its slice crosses a piece boundary, the next one.
    static constexpr uint32_t w0_w1_bank_blocks_per_expert(uint32_t num_banks) {
        return moe_ring::w0_w1_bank_blocks_per_expert(w0_w1_dram_tiles_h, Nt, num_cores, num_banks, tiles_per_txn);
    }

    // Shared-expert (TpNt) variants: the intermediate dim is TP-split to TpNt = ceil(Nt/tp).
    // After add_shared_expert_weights front-packs each core's real TpNt slice to the front of its
    // full-Nt shard, the kernel reads/produces only the real prefix (in2_tiles_per_step_shared per
    // core) and zero-fills the rest of the core's columns; the full W2 walk then contracts
    // real×real in the prefix and zero×zero past it.
    static constexpr uint32_t TpNt = detail::div_up<Nt, SharedExpertTp>();
    static constexpr uint32_t in2_tiles_per_step_shared =
        even_stride_at_least_a2a_width((TpNt + num_cores - 1) / num_cores);
    // Columns core c produces for an expert: all of its columns for a routed expert; for a shared expert the
    // front-packed prefix, capped at the columns it stores (an even prefix shorter than cols reads pairs only).
    static constexpr uint32_t w0_w1_prod_cols(uint32_t core_id, bool is_shared_expert) {
        const uint32_t cols = w0_w1_stored_cols(Nt, core_id, num_cores);
        return (is_shared_expert && in2_tiles_per_step_shared < cols) ? in2_tiles_per_step_shared : cols;
    }

    // W2
    static constexpr uint32_t max_w2_tiles_per_core = (Ht + num_cores - 1) / num_cores;
    static constexpr uint32_t num_a2a_iters =
        (max_w2_tiles_per_core + W2_TILES_PER_A2A_ITER_W - 1) / W2_TILES_PER_A2A_ITER_W;
    static constexpr uint32_t w2_tiles_per_expert_w = num_a2a_iters * W2_TILES_PER_A2A_ITER_W;
    // With ALT_TILES_PER_TXN the last a2a iteration is half width (2 output tiles, w2_blocks_per_half_a2a_iter
    // blocks of 2 wide x half_block_tiles_h) when every core's last iteration has at most 2 output tiles. The
    // output rows keep the w2_tiles_per_expert_w pitch: the half iteration still packs 4 DEST tiles, the two it does
    // not compute are zero (the packer clears the DEST half at every tile_regs_release) and dm1 copies only
    // w2_shard_tiles(c) tiles of each row.
    static constexpr uint32_t w2_dram_tiles_h = has_bias ? Nt + 1 : Nt;
    static constexpr bool w2_last_iter_half = w2_last_a2a_iter_half(Ht, num_cores, tiles_per_txn);
    static constexpr uint32_t w2_blocks_per_a2a_iter = (w2_dram_tiles_h + block_tiles_h - 1) / block_tiles_h;
    static constexpr uint32_t w2_blocks_per_half_a2a_iter =
        (w2_dram_tiles_h + half_block_tiles_h - 1) / half_block_tiles_h;
    // Blocks per expert that dm0 reads and compute consumes: NOT w2_blocks_per_a2a_iter * num_a2a_iters when the
    // last iteration is half width -- consume per iteration.
    static constexpr uint32_t w2_blocks_per_expert =
        w2_core_blocks_per_expert(Ht, w2_dram_tiles_h, num_cores, tiles_per_txn);
    static_assert(
        w2_blocks_per_expert == (num_a2a_iters - (w2_last_iter_half ? 1u : 0u)) * w2_blocks_per_a2a_iter +
                                    (w2_last_iter_half ? w2_blocks_per_half_a2a_iter : 0u),
        "W2 blocks per expert must be the full iterations' blocks plus the half iteration's");
};

}  // namespace moe_ring
