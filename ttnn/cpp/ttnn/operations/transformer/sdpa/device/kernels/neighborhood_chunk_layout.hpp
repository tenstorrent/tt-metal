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

// A linear index into a (time, height, width) grid -> the point it names. The exact inverse of
// point3_to_linear below, and both are ROW-MAJOR over (time, height, width) -- the order tokens
// arrive in, the order bricks tile a chunk, and the order site_to_bricked_index uses on the host.
// All four have to agree: a work item is decoded to a point here and re-linearised there to
// address the tensor, so a disagreement reads the wrong tile row for every brick and still
// returns plausible video.
//
// UNIT must be given explicitly, because it appears only in the return type and so cannot be
// deduced -- and it genuinely differs between the two levels this serves. Decoding a chunk index
// yields a ChunkPoint, which is a whole chunk shape away from the BrickPoint the caller wants;
// decoding a brick index within a chunk yields a BrickPoint directly. Naming the unit at the call
// site is what stops those two being confused.
//
// The grid must be measured in the SAME unit as the point being decoded, which the signature now
// enforces. Its PER is free: decoding chunks over the volume uses a plain ShapeInChunks, decoding
// bricks within a chunk uses the ChunkShapeInBricks, and both are legitimate brick/chunk grids.
template <Unit UNIT, Unit PER>
FORCE_INLINE Point3<uint32_t, UNIT> linear_to_point3(uint32_t index, Shape<UNIT, PER> grid) {
    const uint32_t width = grid[Axis::Width];
    const uint32_t per_time_slice = grid[Axis::Height] * width;
    const uint32_t time = index / per_time_slice;
    const uint32_t remainder = index % per_time_slice;
    return Point3<uint32_t, UNIT>::at(time, remainder / width, remainder % width);
}

// The brick coordinate of a chunk's first brick.
FORCE_INLINE BrickPoint
chunk_origin_brick(uint32_t chunk_index, ShapeInChunks volume_chunks, ChunkShapeInBricks chunk_shape) {
    return first_brick_of(linear_to_point3<Unit::Chunks>(chunk_index, volume_chunks), chunk_shape);
}

// The brick coordinate of the `index`-th brick inside a chunk.
FORCE_INLINE BrickPoint
brick_within_chunk(uint32_t index_in_chunk, const BrickPoint& origin, ChunkShapeInBricks chunk_shape) {
    return origin + linear_to_point3<Unit::Bricks>(index_in_chunk, chunk_shape);
}

// A point -> the linear index naming it. The exact inverse of linear_to_point3 above, and the
// call that turns a brick coordinate into a tile row of the bricked tensor.
//
// UNIT is DEDUCED here rather than named, which is the one asymmetry with the decode: it appears
// in the argument, not only in the return type. Nothing is lost by that -- handing this a
// ChunkPoint where a BrickPoint belongs is already a type error, and `grid` says which space the
// index lands in.
template <Unit UNIT, Unit PER>
FORCE_INLINE uint32_t point3_to_linear(Point3<uint32_t, UNIT> point, Shape<UNIT, PER> grid) {
    return (point.time() * grid[Axis::Height] + point.height()) * grid[Axis::Width] + point.width();
}

// Is this point inside the grid? A brick that runs past the volume holds only ghost sites, and
// its queries are discarded. Generic for the same reason point3_to_linear is: the test is the
// same on a chunk grid as on a brick grid, and the unit tag keeps the two apart.
template <Unit UNIT, Unit PER>
FORCE_INLINE bool point3_is_inside(Point3<uint32_t, UNIT> point, Shape<UNIT, PER> grid) {
    return point.time() < grid[Axis::Time] && point.height() < grid[Axis::Height] && point.width() < grid[Axis::Width];
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
