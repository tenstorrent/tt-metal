// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "neighborhood_kernel_args.hpp"

// How a query CHUNK maps onto bricks, shared by the reader and the writer so they cannot
// disagree about which tile row is which query.
//
// A chunk is the unit of work: its bricks share one gather, one mask and one flash pass. That
// sharing is the whole point -- keys gathered per query is what governs cost, and a chunk of
// one brick re-gathers nearly the same keys for every tile row.
//
// Chunks tile the volume time-major, and bricks tile a chunk the same way, so both index
// calculations below are the ordinary row-major one applied twice.

namespace ttnn::transformer::neighborhood::chunk_layout {

struct BrickCoordinate {
    uint32_t time;
    uint32_t height;
    uint32_t width;
};

// Row-major decode shared by both levels: chunks within the volume, bricks within a chunk.
FORCE_INLINE BrickCoordinate decode_row_major(uint32_t index, const kernel_args::AxisExtents& extent) {
    const uint32_t per_time_slice = extent.height * extent.width;
    const uint32_t time = index / per_time_slice;
    const uint32_t remainder = index % per_time_slice;
    return BrickCoordinate{time, remainder / extent.width, remainder % extent.width};
}

// The brick coordinate of a chunk's first brick.
FORCE_INLINE BrickCoordinate chunk_origin_brick(
    uint32_t chunk_index, const kernel_args::AxisExtents& volume_chunks, const kernel_args::AxisExtents& chunk_bricks) {
    const BrickCoordinate chunk = decode_row_major(chunk_index, volume_chunks);
    return BrickCoordinate{
        chunk.time * chunk_bricks.time, chunk.height * chunk_bricks.height, chunk.width * chunk_bricks.width};
}

// The brick coordinate of the `index`-th brick inside a chunk.
FORCE_INLINE BrickCoordinate brick_within_chunk(
    uint32_t index_in_chunk, const BrickCoordinate& origin, const kernel_args::AxisExtents& chunk_bricks) {
    const BrickCoordinate offset = decode_row_major(index_in_chunk, chunk_bricks);
    return BrickCoordinate{origin.time + offset.time, origin.height + offset.height, origin.width + offset.width};
}

// Linear brick index into the bricked tensor.
FORCE_INLINE uint32_t brick_index(const BrickCoordinate& brick, const kernel_args::AxisExtents& volume_bricks) {
    return (brick.time * volume_bricks.height + brick.height) * volume_bricks.width + brick.width;
}

// A brick that runs past the volume holds only ghost sites; its queries are discarded.
FORCE_INLINE bool brick_is_inside(const BrickCoordinate& brick, const kernel_args::AxisExtents& volume_bricks) {
    return brick.time < volume_bricks.time && brick.height < volume_bricks.height && brick.width < volume_bricks.width;
}

// Where one (brick, head) pair's tiles begin.
//
// Q, K, V and the output are all stored SITE-MAJOR: [batch, brick_count * 32 sites, head_count,
// head_dim]. Head-major would put a head's sites contiguously, but nothing here reads a head
// contiguously -- every read is one brick of one head -- and it would force the caller to
// transpose heads against sites on the way in and back again on the way out. That transpose
// measured 24.6 ms per block at stage-5 size, a third of all layout cost, for no arithmetic.
FORCE_INLINE uint32_t tile_offset(
    uint32_t batch_index,
    uint32_t brick,
    uint32_t head_index,
    uint32_t brick_count,
    uint32_t head_count,
    uint32_t head_dim_tiles) {
    return ((batch_index * brick_count + brick) * head_count + head_index) * head_dim_tiles;
}

}  // namespace ttnn::transformer::neighborhood::chunk_layout
