// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <vector>

// Site, SiteOffset, BrickPoint, Axis and the unit conversions between them. Shared with the
// kernels so that host and device cannot disagree about what a position means.
#include "kernels/neighborhood_point3.hpp"

// 3D neighborhood attention geometry.
//
// Every neighborhood concept in the design lives in this file and nowhere else. It has no
// ttnn, kernel, or device dependencies on purpose: the geometry is where the bugs are, and
// this way it is testable on the host against a brute-force oracle with no hardware.
//
// The rule that makes neighborhood attention tractable: the context window keeps its SIZE
// constant and slides inward at a volume boundary, rather than truncating. A query at site 0
// attends to [0, K), not to a half-empty [0, K/2]. So every context window is fully in
// bounds, every query group gathers the same number of sites, and there is nothing
// out-of-range to mask. A truncating window looks plausible and is wrong near every edge.
//
// Naming: every count carries its unit -- _sites, _bricks, _tiles, _index, _count. You
// cannot add a brick count to a site count when the names do not match. POSITIONS enforce the
// same rule through the type system instead: see Unit in kernels/neighborhood_point3.hpp.

namespace ttnn::transformer::neighborhood {

// One brick is one hardware tile row: 32 sites, in some 3D arrangement.
constexpr uint32_t SITES_PER_BRICK = 32;

// Natural: row-major over (time, height, width), the order tokens arrive in.
// Bricked: 32 consecutive tokens form one compact 3D box -- see site_to_bricked_index.
enum class Order : uint8_t { Natural, Bricked };

// Where a context window sits against a volume boundary on one axis. Three states per axis
// means at most 27 distinct window geometries in a volume of any size. These change the
// window's ORIGIN, never its validity -- see the clamping rule above.
enum class Regime : uint8_t { Low, Interior, High };

struct NeighborhoodConfig {
    ShapeInSites volume;          // the GLOBAL token grid, in sites
    ShapeInSites context_window;  // what one query group attends to, in sites
    ShapeInSites stride;          // query group extent, in sites
    BrickShapeInSites brick;      // layout unit: sites per brick; brick.count() == SITES_PER_BRICK

    // How many bricks one query CHUNK spans, per axis. A chunk is the set of queries that share
    // a single gather, so this is the knob that decides how far the gather amortises.
    //
    // Keys-gathered-per-query is what governs cost, and it is `box / chunk_queries`. With a
    // one-brick chunk an 11^3 window costs 1728/32 = 54 keys per query; spread over 4x2x2
    // bricks it is 5376/512 = 10.5. The reference implementation runs blocks of 480 queries for
    // exactly this reason -- a small chunk re-gathers almost the same keys for every tile row.
    //
    // (1,1,1) is one brick per chunk, the original behaviour.
    ChunkShapeInBricks query_chunk_bricks{{1, 1, 1}};

    // Sharding. `shard_extent` is what this device actually holds -- its owned region PLUS the
    // halo its queries reach into -- and `shard_origin` is where that sits in the global volume.
    // Unsharded is shard_extent == volume with a zero origin, which is the default.
    //
    // The split matters because window placement must stay GLOBAL: a window is clamped at the
    // true volume boundary, never at a shard's internal edge. Clamping locally would silently
    // truncate every query within half a window of a shard seam -- correct-looking output with a
    // wrong receptive field along every internal boundary.
    //
    // `shard_origin` must be brick-aligned, so the local tensor bricks the same way the global
    // one would. It is SIGNED -- see SiteOffset.
    ShapeInSites shard_extent{{0, 0, 0}};  // zero means "same as volume"
    SiteOffset shard_origin{{0, 0, 0}};

    // The sub-box of the resident region this device actually produces OUTPUT for, and where it
    // starts inside that region. Zero extent means "the whole resident region", which is what an
    // unsharded run gets and what every caller predating this got.
    //
    // A query needs a widened KEY region -- its window reaches past the shard seam -- but never a
    // widened QUERY region: the halo's own queries belong to the neighbour, which computes them
    // itself. Without this split the halo's queries are computed and thrown away, which at the
    // stage-5 W-shard is 16 of every 76 resident columns, and Q has to be widened by a halo
    // exchange purely to satisfy a shape that is then discarded.
    //
    // Both must be brick-aligned: the query region is addressed in whole bricks, and a sub-box
    // starting or ending mid-brick would put owned and neighbour sites in one tile row.
    ShapeInSites query_extent{{0, 0, 0}};
    Site query_origin{{0, 0, 0}};  // in RESIDENT-local sites, so always >= 0

    // The query chunk's extent in SITES.
    ShapeInSites query_chunk_sites() const {
        return ShapeInSites::of(
            query_chunk_bricks.time() * brick.time(),
            query_chunk_bricks.height() * brick.height(),
            query_chunk_bricks.width() * brick.width());
    }
    uint32_t bricks_per_query_chunk() const { return query_chunk_bricks.count(); }

    ShapeInSites resident_extent() const { return shard_extent.count() == 0 ? volume : shard_extent; }
    bool is_sharded() const { return shard_extent.count() != 0 && !(shard_extent == volume); }

    // The region queries are drawn from. Defaults to the whole resident region, so a config that
    // never sets query_extent behaves exactly as it did before the split existed.
    ShapeInSites query_region() const { return query_extent.count() == 0 ? resident_extent() : query_extent; }
    bool has_query_subregion() const {
        return query_extent.count() != 0 && !(query_extent == resident_extent() && query_origin == Site{{0, 0, 0}});
    }
};

// A placed context window. `extent` equals config.context_window except on an axis shorter
// than the window, where it is the whole axis.
struct ContextWindow {
    Site origin;
    ShapeInSites extent;
};

// Built once per (volume, context_window, stride, brick) and cached: it uploads index
// tables, so rebuilding it per block would dominate.
struct NeighborhoodPlan {
    NeighborhoodConfig config;

    ShapeInBricks volume_bricks;  // the RESIDENT region measured in bricks (rounded up)
    uint32_t brick_count = 0;

    // The QUERY region measured in bricks, and where it starts inside the resident brick grid.
    // K, V and the gather address the resident grid above; Q and the output address this one.
    // They coincide unless the config asked for a query sub-region.
    ShapeInBricks query_bricks;
    uint32_t query_brick_count = 0;
    BrickPoint query_origin_bricks;  // a position, not a size: config.query_origin in bricks

    // The QUERY region measured in query chunks. One chunk is one unit of work: its bricks share
    // a gather, a mask and a flash pass. Chunks are counted over the query region, not the
    // resident one -- computing a chunk for a brick whose output is discarded is the waste this
    // whole split exists to remove.
    ShapeInChunks volume_chunks;
    uint32_t chunk_count = 0;

    // What ONE brick must gather: the union of the context windows of every query group it
    // contains. At stride == brick that is exactly one context window. At stride 1 the 32
    // queries have 32 distinct windows and the union is wider on every axis.
    ShapeInSites gather_extent;
    uint32_t gather_sites = 0;
    uint32_t gather_tiles = 0;  // site-exact: ceil(gather_sites / SITES_PER_BRICK)

    // The same region rounded out to whole bricks, which is what a tile-granular read can
    // actually fetch: one brick is one tile row, so the key axis is measured in bricks.
    // Larger than gather_tiles -- 175 against 74 for an 11^3 window at stride 1 -- because a
    // window whose half-extent is not a multiple of the brick straddles bricks on every axis.
    // That gap is what stride buys back: at stride == brick the two converge.
    ShapeInBricks gather_bricks;
    uint32_t gather_brick_count = 0;

    // Indexed by CHUNK index; each is rounded DOWN to a brick boundary so a whole-brick read
    // starting here covers the region. Constant size, because the clamping rule keeps every
    // gather the same extent -- which is also why core load balancing is trivial.
    std::vector<Site> gather_origin_by_chunk;
};

// Pick the brick shape minimising the gathered union for a given context window. A function
// rather than a constant: a window flat in time wants a brick flat in time. `11x11x11` and
// `2x4x4` belong in a model config, never here and never in a kernel.
BrickShapeInSites choose_brick(ShapeInSites context_window);

// Which boundary regime one site sits in on one axis.
Regime regime_for_axis(uint32_t site_index, uint32_t volume_extent_sites, uint32_t context_window_extent_sites);

// The context window for the query group starting at `query_group_origin`. Always fully in
// bounds; always contains its own query group.
ContextWindow context_window_for(Site query_group_origin, const NeighborhoodConfig& config);

// Natural <-> Bricked. Round-trips for every site in the volume.
uint32_t site_to_bricked_index(Site site, const NeighborhoodConfig& config);
Site bricked_index_to_site(uint32_t bricked_index, const NeighborhoodConfig& config);

// The brick a site belongs to, and the first site of a brick.
uint32_t site_to_brick_index(Site site, const NeighborhoodConfig& config);
Site brick_index_to_origin(uint32_t brick_index, const NeighborhoodConfig& config);

// Throws std::invalid_argument on a config that cannot be built.
void validate_config(const NeighborhoodConfig& config);

NeighborhoodPlan build_plan(const NeighborhoodConfig& config);

}  // namespace ttnn::transformer::neighborhood
