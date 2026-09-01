// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "neighborhood_point3.hpp"

// The argument layout shared by the program factory and the three kernels: circular buffer
// ids, the compile-time argument slots, and the geometry the reader needs to address bricks.
//
// Everything here is named on BOTH sides. The factory writes an argument by name into a
// vector sized ...::COUNT; each kernel reads it back by the same name. Neither side counts
// argument positions, so inserting one in the wrong place fails to compile rather than
// silently shifting every later argument -- which is the failure mode
// sliding_window_geometry.hpp warns about in its own header, where the reader and the compute
// kernel each recompute the same bounds and must agree.

namespace ttnn::transformer::neighborhood::kernel_args {

// Fixed ids rather than compile-time arguments: one definition, no ordering to keep in sync.
enum CircularBufferId : uint32_t {
    cb_query = 0,        // [head_dim_tiles] -- one brick of queries
    cb_key,              // [tiles_per_kv_chunk * head_dim_tiles], double buffered
    cb_value,            // [tiles_per_kv_chunk * head_dim_tiles], double buffered
    cb_mask,             // [tiles_per_kv_chunk] additive {0, -inf}, double buffered
    cb_reduce_scalar,    // 1 tile: the reduce identity
    cb_zero,             // 1 tile of zeros: matmul_blocks adds (zero + mask) into the score dst
    cb_column_identity,  // 1 tile, ones in column 0: finalises the deferred row-sum
    cb_scores,           // [tiles_per_kv_chunk] -- Q . K^T for one chunk
    cb_row_max_current,  // running max, ping-pong
    cb_row_max_previous,
    cb_row_sum_current,  // running sum, ping-pong
    cb_row_sum_previous,
    cb_exp_max_difference,          // exp(previous_max - current_max), the rescale factor
    cb_output_accumulator_current,  // running output, ping-pong
    cb_output_accumulator_previous,
    cb_output,         // [head_dim_tiles] -- normalized, handed to the writer
    cb_gather_origin,  // reader-internal scratch: one row of the host-built origin table
    cb_resident_mask,  // reader-internal scratch: one regime's whole uploaded mask set, kept
                       // across work items so a shared pattern is fetched once, not per chunk
    CB_COUNT
};

// Columns in the gather origin table. The row is padded out to 64 bytes so each chunk's entry
// is one DRAM-aligned page on both Wormhole (32B) and Blackhole (64B), which leaves room to
// carry the shard origin alongside the gather origin at no cost.
namespace gather_origin_column {
enum : uint32_t {
    gather_time = 0,  // where this chunk's gather starts, in LOCAL sites
    gather_height,
    gather_width,

    // Where this device's tensor starts in the GLOBAL volume, as a SIGNED site offset -- a halo
    // at the low edge of the volume puts a device at a negative origin. It rides the table because it is
    // the one geometric value that DIFFERS PER DEVICE, and a mesh runs a single program: as a
    // compile-time argument it would be uniform across the mesh, so every shard would believe
    // it sat at the origin and would clamp its windows at its own seam instead of at the true
    // volume boundary. Repeated in every row -- three words against needing a second tensor.
    shard_origin_time,
    shard_origin_height,
    shard_origin_width,
};
}
constexpr uint32_t GATHER_ORIGIN_COLUMNS = 16;
constexpr uint32_t GATHER_ORIGIN_ROW_BYTES = GATHER_ORIGIN_COLUMNS * sizeof(uint32_t);

namespace reader_arg {
enum : uint32_t {
    head_count = 0,
    brick_count,
    head_dim_tiles,

    // A query CHUNK is the unit of work: its bricks share one gather, one mask and one flash
    // pass. Chunk shape in bricks, plus the volume measured in chunks.
    query_chunk_bricks_time,
    query_chunk_bricks_height,
    query_chunk_bricks_width,
    bricks_per_query_chunk,
    volume_chunks_time,
    volume_chunks_height,
    volume_chunks_width,
    tiles_per_kv_chunk,
    kv_chunk_count,
    gather_brick_count,

    // 1 = emit one mask tile per (query brick, gather slot) instead of one per slot shared by the
    // whole chunk. Required once the chunk is wider than the stride, because then the bricks do
    // not share a window and the broadcast mask silently attends to the wrong one.
    per_brick_mask,

    // 1 = fill every mask tile with a constant (DIFFVAE_NA_MASK_MEMSET_ONLY). Diagnostic only:
    // wrong output, but it isolates tile WRITE cost from tile CONTENT cost.
    mask_memset_only,

    // 1 = the uploaded mask is indexed by the RELATIVE brick offset (key_brick - query_brick)
    // rather than by (regime, gather slot). That is what a stride-1 mask needs: every query
    // centres its own window, so the pattern depends only on the relative offset -- and unlike a
    // regime set it carries no dependence on the gather origin's brick phase or on the shard
    // origin, so ONE table serves every chunk and every shard.
    relative_mask,

    // DIFFVAE_NA_TABLE_ALWAYS: skip the per-brick clamping gate. Diagnostic -- edge bricks get
    // the interior pattern, so the frame is wrong there, but it shows what the gate is costing.
    table_always,

    // DIFFVAE_NA_SKIP_KV: issue no K/V reads at all. Diagnostic with WRONG output; it separates
    // the gather's cost from the compute kernel's.
    skip_kv,

    // The volume and the gather span, measured in bricks -- one brick is one tile row, so
    // these are also tile counts.
    volume_bricks_time,
    volume_bricks_height,
    volume_bricks_width,
    gather_bricks_time,
    gather_bricks_height,
    gather_bricks_width,

    // The QUERY region, in bricks, and where it starts inside the resident brick grid. K, V and
    // the gather address the resident grid (volume_bricks / brick_count); Q addresses this one.
    // They coincide unless the host asked for a query sub-region, which is how a W-shard says
    // "my queries are the columns I own, my keys are those plus the halo".
    query_bricks_time,
    query_bricks_height,
    query_bricks_width,
    query_brick_count,
    query_origin_bricks_time,
    query_origin_bricks_height,
    query_origin_bricks_width,

    // Sites per brick, per axis. Their product is SITES_PER_BRICK.
    brick_sites_time,
    brick_sites_height,
    brick_sites_width,

    // What the mask generator needs to place each query's window.
    context_window_time,
    context_window_height,
    context_window_width,
    stride_time,
    stride_height,
    stride_width,
    volume_time,
    volume_height,
    volume_width,

    // Sharding. `volume_*` above is the GLOBAL grid and drives window placement; `resident_*`
    // says how much of it this device holds. Addressing is local, window placement is global --
    // conflating them clamps windows at shard seams.
    //
    // The shard ORIGIN is deliberately absent: it differs per device, and a mesh runs one
    // program. It arrives per chunk in the gather origin table instead -- see
    // gather_origin_column above.
    resident_time,
    resident_height,
    resident_width,

    // 1 when an interior mask tensor is supplied. Away from the volume boundary every query
    // brick shares one mask pattern, so it is built once on the HOST and read like K or V
    // instead of being re-evaluated per (query brick, key brick) pair -- the same collapse the
    // reference gets by broadcasting one mask over a whole group of query tiles.
    has_interior_mask,

    COUNT
};
}  // namespace reader_arg

namespace compute_arg {
enum : uint32_t {
    head_dim_tiles = 0,
    query_tile_rows,  // = bricks_per_query_chunk; one brick is one tile row of queries
    tiles_per_kv_chunk,
    kv_chunk_count,
    work_item_count,      // (batch * head_count * brick_count) assigned to this core
    scale_as_float_bits,  // std::bit_cast of the float scale

    // Matmul subblock shape. A subblock's tiles live in DST, which holds 8 (4 with fp32
    // accumulate) -- so the output width CANNOT simply be the full chunk. Overrunning DST
    // does not fault, it silently returns wrong numbers for every site.
    scores_subblock_width,
    scores_subblock_count,
    output_subblock_width,
    output_subblock_count,

    // Tiles to advance the mask CB per in0 subblock (= per query tile row = per brick). 0 keeps
    // the broadcast: every brick re-reads the same mask, which is only right when the chunk's
    // bricks share one context window. tiles_per_kv_chunk gives each brick its OWN mask, which
    // is what a chunk wider than the stride needs.
    mask_subblock_stride,
    COUNT
};
}  // namespace compute_arg

namespace writer_arg {
enum : uint32_t {
    head_count = 0,
    brick_count,
    head_dim_tiles,
    // A query CHUNK is the unit of work: its bricks share one gather, one mask and one flash
    // pass. Chunk shape in bricks, plus the volume measured in chunks.
    query_chunk_bricks_time,
    query_chunk_bricks_height,
    query_chunk_bricks_width,
    bricks_per_query_chunk,
    volume_chunks_time,
    volume_chunks_height,
    volume_chunks_width,

    // The writer drains the QUERY region: the output is query-sized, so these are the query
    // brick grid, not the resident one. `brick_count` above is likewise the query brick count.
    volume_bricks_time,
    volume_bricks_height,
    volume_bricks_width,
    COUNT
};
}  // namespace writer_arg

// The per-axis shapes the mask generator needs. Passed as one struct rather than nine loose
// arguments so a caller cannot transpose height and width without noticing. Every member is a
// Shape from neighborhood_point3.hpp, so the unit each is measured in is part of its type -- a
// brick grid cannot be handed where a site region belongs, and the two unit shapes cannot be
// swapped for each other.
struct NeighborhoodExtents {
    BrickShapeInSites brick_sites;  // sites per brick -- a conversion factor, not a region
    ShapeInSites context_window;    // the unclamped window from the config
    ShapeInSites stride;            // query group extent
    ShapeInSites volume;            // the true GLOBAL volume, NOT the brick-padded one
    ShapeInSites query_chunk;       // one query chunk, in sites -- the unit that shares a window
    SiteOffset shard_origin;        // where this device's tensor starts in the global volume; signed
    ShapeInSites resident;          // how much of it this device holds (owned + halo)
};

}  // namespace ttnn::transformer::neighborhood::kernel_args
