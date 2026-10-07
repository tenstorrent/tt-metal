// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::experimental::matmul_decode {

// Where one weight lives inside a larger fused weight tensor.
//
// The fused tensor is a single HEIGHT_SHARDED L1 tensor that packs many weights: every core
// carries one equal-sized, one-tile-wide shard, and a weight's per-core slab occupies the tile
// range [tile_offset, tile_offset + slab_tiles) of the shard on each of `cores`. The tiles of a
// region are the slab's 32x32 tiles in row-major order -- exactly the order a width-sharded
// weight shard stores them -- so the compute kernels consume a region the same way they consume
// a dedicated weight tensor; only the base address differs.
//
// The fused tensor's own shape carries no information about the weight, so the logical geometry
// travels here instead:
//   * full width-sharded (k_blocks == 1, batch == 1): slab [K, N / n] on n = cores.num_cores()
//     cores, N split across the cores in row-major shard order;
//   * partial width-sharded (k_blocks > 1): slab [K / k_blocks, N / n] on k_blocks * n cores,
//     core i holding K-block i / n and N-block i % n; the K-partials reduce onto the first n
//     cores;
//   * batched (batch > 1): slab [(batch / b_blocks) * K, N / n] on b_blocks * n cores, core i
//     holding batch-block i / n and N-block i % n.
struct PackedWeightSpec {
    // First tile of this weight's region within each core's shard of the fused tensor.
    uint32_t tile_offset = 0;
    // The weight's logical [K, N] (must be tile-aligned; K must match input A's inner dim).
    uint32_t K = 0;
    uint32_t N = 0;
    // The cores holding the weight's slabs, in row-major shard order.
    CoreRangeSet cores;
    // Partial width-sharded mode when > 1: the weight is cut into k_blocks x n_blocks blocks.
    uint32_t k_blocks = 1;
    // Batched (BatchedLinearDecode) mode when batch > 1: b_blocks x n_blocks grid of
    // [(batch / b_blocks) * K, N / n_blocks] slabs.
    uint32_t batch = 1;
    uint32_t b_blocks = 1;

    uint32_t num_cores() const { return cores.num_cores(); }

    // n_blocks is implied: the cores not consumed by the K- or batch-cut split N.
    uint32_t n_blocks() const {
        const uint32_t denom = batch > 1 ? b_blocks : k_blocks;
        return denom > 0 ? num_cores() / denom : 0;
    }
};

}  // namespace ttnn::operations::experimental::matmul_decode
