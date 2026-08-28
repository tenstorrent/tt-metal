// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "neighborhood_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

// The window rule itself lives here so the device mask generator uses the SAME definition
// rather than a transcription of it.
#include "kernels/neighborhood_window_rule.hpp"

namespace ttnn::transformer::neighborhood {

namespace {

uint32_t ceil_div(uint32_t numerator, uint32_t denominator) { return (numerator + denominator - 1) / denominator; }

// Bricking is over the RESIDENT extent (this device's shard plus halo), not the global volume:
// brick indices address the local tensor. Window placement stays global -- see the config.
ShapeInBricks volume_in_bricks(const NeighborhoodConfig& config) {
    const ShapeInSites resident = config.resident_extent();
    ShapeInBricks result;
    for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
        result.by_axis[axis_index] = ceil_div(resident.by_axis[axis_index], config.brick.by_axis[axis_index]);
    }
    return result;
}

// The context window never exceeds the axis it sits on: an axis shorter than the window is
// attended to in full.
uint32_t window_extent_on_axis(uint32_t context_window_extent_sites, uint32_t volume_extent_sites) {
    return std::min(context_window_extent_sites, volume_extent_sites);
}

// The union of the context windows of every query group inside one CHUNK, on one axis.
//
// A chunk holding one group (stride == the chunk's extent) gathers exactly one window -- the
// maskless regime, and the cheapest keys-per-query there is. Every extra group widens the union
// by one stride, so the gather grows far more slowly than the query count: that is why a chunk
// spanning many bricks amortises so much better than one brick.
uint32_t gather_extent_on_axis(
    uint32_t chunk_extent_sites,
    uint32_t stride_extent_sites,
    uint32_t window_extent_sites,
    uint32_t volume_extent_sites) {
    const uint32_t query_groups_in_chunk = ceil_div(chunk_extent_sites, stride_extent_sites);
    const uint32_t union_extent_sites = window_extent_sites + (query_groups_in_chunk - 1) * stride_extent_sites;
    return std::min(union_extent_sites, volume_extent_sites);
}

// Where one chunk starts, in sites, local to this device's tensor.
// The query region in bricks, and where it starts inside the resident brick grid. Both are exact
// divisions: validate_config requires the query origin and extent to be brick-aligned.
ShapeInBricks query_in_bricks(const NeighborhoodConfig& config) {
    const ShapeInSites query = config.query_region();
    ShapeInBricks result;
    for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
        result.by_axis[axis_index] = ceil_div(query.by_axis[axis_index], config.brick.by_axis[axis_index]);
    }
    return result;
}

BrickPoint query_origin_in_bricks(const NeighborhoodConfig& config) {
    return containing_brick(config.query_origin, config.brick);
}

// A chunk's origin, in QUERY-region-local sites. Add config.query_origin for resident-local.
Site chunk_index_to_origin(uint32_t chunk_index, const NeighborhoodPlan& plan) {
    const ShapeInSites chunk = plan.config.query_chunk_sites();
    const uint32_t chunks_per_time_slice = plan.volume_chunks.height() * plan.volume_chunks.width();
    const uint32_t time_index = chunk_index / chunks_per_time_slice;
    const uint32_t remainder = chunk_index % chunks_per_time_slice;
    return Site::at(
        time_index * chunk.time(),
        (remainder / plan.volume_chunks.width()) * chunk.height(),
        (remainder % plan.volume_chunks.width()) * chunk.width());
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument("neighborhood_plan: " + message);
    }
}

}  // namespace

BrickShapeInSites choose_brick(ShapeInSites context_window) {
    BrickShapeInSites best_brick = BrickShapeInSites::of(1, 1, SITES_PER_BRICK);
    uint64_t best_union_sites = UINT64_MAX;
    uint32_t best_largest_extent = UINT32_MAX;

    for (uint32_t time_extent = 1; time_extent <= SITES_PER_BRICK; time_extent *= 2) {
        for (uint32_t height_extent = 1; time_extent * height_extent <= SITES_PER_BRICK; height_extent *= 2) {
            const uint32_t planar_sites = time_extent * height_extent;
            if (SITES_PER_BRICK % planar_sites != 0) {
                continue;
            }
            const uint32_t width_extent = SITES_PER_BRICK / planar_sites;
            const BrickShapeInSites brick_candidate = BrickShapeInSites::of(time_extent, height_extent, width_extent);

            // The union one brick must gather at stride 1 -- the shape-sensitive cost.
            uint64_t union_sites = 1;
            for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
                union_sites *= context_window.by_axis[axis_index] + brick_candidate.by_axis[axis_index] - 1;
            }
            const uint32_t largest_extent =
                std::max({brick_candidate.time(), brick_candidate.height(), brick_candidate.width()});

            // Tie-break toward the most cubic brick, then lexicographically, so the result is
            // deterministic rather than dependent on enumeration order.
            const bool better = union_sites < best_union_sites ||
                                (union_sites == best_union_sites && largest_extent < best_largest_extent);
            if (better) {
                best_union_sites = union_sites;
                best_largest_extent = largest_extent;
                best_brick = brick_candidate;
            }
        }
    }
    return best_brick;
}

Regime regime_for_axis(uint32_t site_index, uint32_t volume_extent_sites, uint32_t context_window_extent_sites) {
    const uint32_t window_extent_sites = window_extent_on_axis(context_window_extent_sites, volume_extent_sites);
    if (window_extent_sites >= volume_extent_sites) {
        // The window covers the whole axis; there is only one placement.
        return Regime::Interior;
    }
    const uint32_t half_window_sites = window_extent_sites / 2;
    const uint32_t highest_origin = volume_extent_sites - window_extent_sites;

    if (site_index < half_window_sites) {
        return Regime::Low;
    }
    if (site_index - half_window_sites > highest_origin) {
        return Regime::High;
    }
    return Regime::Interior;
}

ContextWindow context_window_for(Site query_group_origin, const NeighborhoodConfig& config) {
    ContextWindow window;
    for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
        const uint32_t volume_extent_sites = config.volume.by_axis[axis_index];
        const uint32_t stride_extent_sites = config.stride.by_axis[axis_index];
        const uint32_t window_extent_sites =
            window_extent_on_axis(config.context_window.by_axis[axis_index], volume_extent_sites);
        const uint32_t query_group_index = query_group_origin.by_axis[axis_index] / stride_extent_sites;

        // Same snapping rule the gather uses, or the two disagree about where the window is.
        const uint32_t snap_brick = snap_extent_on_axis(stride_extent_sites, config.brick.by_axis[axis_index]);
        window.origin.by_axis[axis_index] = window_origin_on_axis(
            query_group_index, stride_extent_sites, window_extent_sites, volume_extent_sites, snap_brick);
        window.extent.by_axis[axis_index] = window_extent_sites;
    }
    return window;
}

void validate_config(const NeighborhoodConfig& config) {
    require(config.brick.count() == SITES_PER_BRICK, "brick must hold exactly 32 sites");
    for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
        require(config.query_chunk_bricks.by_axis[axis_index] > 0, "query_chunk_bricks must be non-zero on every axis");
    }

    // A multi-brick chunk must be exactly one query group, so that every one of its tile rows
    // shares a single context window. That is what lets the kernel store one mask tile per
    // gather slot and broadcast it down the rows -- and it is the whole reason a bigger chunk
    // is cheap rather than merely bigger: the gathered box stays the context window however
    // many bricks the chunk holds.
    //
    // Violating it is silently wrong, not loud: the kernel would apply the first row's mask to
    // every row, so queries would attend to a window that is not theirs and still return
    // plausible video. Hence a hard check rather than a comment.
    if (config.bricks_per_query_chunk() > 1) {
        // DIFFVAE_NA_UNSAFE_CHUNK=1 turns this into a PERF PROBE and nothing else. The numbers it
        // produces are WRONG in exactly the way described above -- every brick in the chunk gets
        // the first brick's mask, so queries attend to windows that are not theirs -- but the
        // TIMING is real, and it measures what amortising the gather across a multi-brick chunk
        // would be worth at stride 1 (175 keys/query today). That is the case for building the
        // per-brick mask that would make it correct. Never ship a frame rendered with this set.
        const char* probe = std::getenv("DIFFVAE_NA_UNSAFE_CHUNK");
        if (probe == nullptr || probe[0] != '1') {
            require(
                config.query_chunk_sites() == config.stride,
                "a multi-brick query chunk must equal the stride exactly, so its bricks form one "
                "query group sharing one context window");
        }
    }

    const ShapeInSites resident = config.resident_extent();
    for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
        const int32_t shard_start = config.shard_origin.by_axis[axis_index];
        const int32_t brick_extent = static_cast<int32_t>(config.brick.by_axis[axis_index]);
        require(
            shard_start % brick_extent == 0,
            "shard_origin must be brick-aligned, or the local tensor bricks differently from the global one");
        // A halo may hang off EITHER end of the volume -- those columns are storage the device
        // holds but the volume does not contain, and nothing ever reads them. What must hold is
        // that the shard overlaps the volume at all.
        require(
            shard_start < static_cast<int32_t>(config.volume.by_axis[axis_index]) &&
                shard_start + static_cast<int32_t>(resident.by_axis[axis_index]) > 0,
            "the shard must overlap the global volume");
        require(resident.by_axis[axis_index] > 0, "resident extent must be non-zero on every axis");
    }

    // The query sub-region. Brick-aligned on both ends, and inside the resident region: it is
    // addressed in whole bricks, so a sub-box starting or ending mid-brick would put owned and
    // neighbour sites in the same tile row and there would be no way to write only the owned half.
    if (config.query_extent.count() != 0) {
        const ShapeInSites query = config.query_region();
        for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
            const uint32_t brick_extent = config.brick.by_axis[axis_index];
            require(query.by_axis[axis_index] > 0, "query extent must be non-zero on every axis");
            // The ORIGIN must be brick-aligned: the query grid is a whole-brick sub-grid of the
            // resident one, and an origin mid-brick would put owned and neighbour sites in one
            // tile row with no way to address either half.
            require(
                config.query_origin.by_axis[axis_index] % brick_extent == 0,
                "query_origin must be brick-aligned, or a tile row would straddle the query region's edge");
            // The EXTENT need not be: an axis whose resident extent is not a whole number of
            // bricks rounds up into ghost sites exactly as the resident grid already does (stage
            // 5 at 145 frames is 77 or 78 deep against a 2-deep brick). What must hold is that
            // the rounded-up query bricks still fit inside the rounded-up resident bricks.
            const uint32_t query_bricks = ceil_div(query.by_axis[axis_index], brick_extent);
            const uint32_t resident_bricks = ceil_div(resident.by_axis[axis_index], brick_extent);
            require(
                config.query_origin.by_axis[axis_index] / brick_extent + query_bricks <= resident_bricks,
                "the query region must lie inside the resident region");
        }
    }
    for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
        require(config.volume.by_axis[axis_index] > 0, "volume extent must be non-zero on every axis");
        require(config.context_window.by_axis[axis_index] > 0, "context window must be non-zero on every axis");
        require(config.stride.by_axis[axis_index] > 0, "stride must be non-zero on every axis");
        require(config.brick.by_axis[axis_index] > 0, "brick extent must be non-zero on every axis");
        require(
            config.stride.by_axis[axis_index] <= config.context_window.by_axis[axis_index],
            "stride must not exceed the context window, or a query would fall outside its own window");
    }
}

NeighborhoodPlan build_plan(const NeighborhoodConfig& config) {
    validate_config(config);

    NeighborhoodPlan plan;
    plan.config = config;
    plan.volume_bricks = volume_in_bricks(config);
    plan.brick_count = plan.volume_bricks.count();

    plan.query_bricks = query_in_bricks(config);
    plan.query_brick_count = plan.query_bricks.count();
    plan.query_origin_bricks = query_origin_in_bricks(config);

    // Over the QUERY bricks, not the resident ones: a chunk whose output is discarded is work
    // that should never be scheduled.
    for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
        plan.volume_chunks.by_axis[axis_index] =
            ceil_div(plan.query_bricks.by_axis[axis_index], config.query_chunk_bricks.by_axis[axis_index]);
    }
    plan.chunk_count = plan.volume_chunks.count();

    for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
        const uint32_t volume_extent_sites = config.volume.by_axis[axis_index];
        const uint32_t window_extent_sites =
            window_extent_on_axis(config.context_window.by_axis[axis_index], volume_extent_sites);
        plan.gather_extent.by_axis[axis_index] = gather_extent_on_axis(
            config.query_chunk_sites().by_axis[axis_index],
            config.stride.by_axis[axis_index],
            window_extent_sites,
            volume_extent_sites);
    }
    plan.gather_sites = plan.gather_extent.count();
    plan.gather_tiles = ceil_div(plan.gather_sites, SITES_PER_BRICK);

    // Where each brick's gather starts, before rounding to a brick boundary. Window origins
    // are non-decreasing in the query group index, so the union of a brick's windows starts
    // at the window of the first query group inside it.
    // `brick_origin` is QUERY-region-local; window placement needs the GLOBAL position, which is
    // two hops away: + query_origin puts it in resident-local sites, + shard_origin in global ones.
    const auto union_origin_for = [&](const Site& brick_origin, uint32_t axis_index) {
        const uint32_t volume_extent_sites = config.volume.by_axis[axis_index];
        const uint32_t stride_extent_sites = config.stride.by_axis[axis_index];
        const uint32_t window_extent_sites =
            window_extent_on_axis(config.context_window.by_axis[axis_index], volume_extent_sites);
        // Bricks below the volume belong to a halo that hangs off the low edge; clamp so the
        // group index stays sane. Their windows are never used -- see SiteOffset.
        const int32_t signed_global = static_cast<int32_t>(brick_origin.by_axis[axis_index]) +
                                      static_cast<int32_t>(config.query_origin.by_axis[axis_index]) +
                                      config.shard_origin.by_axis[axis_index];
        const uint32_t global_site = signed_global > 0 ? static_cast<uint32_t>(signed_global) : 0u;
        const uint32_t first_query_group_index = global_site / stride_extent_sites;
        // Snapping is only legal when the whole brick is one query group; otherwise the queries
        // in it have distinct windows and moving the origin would drop keys out of them.
        const uint32_t snap_brick = snap_extent_on_axis(stride_extent_sites, config.brick.by_axis[axis_index]);
        const uint32_t union_origin = window_origin_on_axis(
            first_query_group_index, stride_extent_sites, window_extent_sites, volume_extent_sites, snap_brick);
        // Every gather is the same extent, so pulling an edge brick's origin down keeps it in
        // bounds without changing the tile count. It never under-gathers: moving the origin
        // down only widens the covered range.
        return std::min(union_origin, volume_extent_sites - plan.gather_extent.by_axis[axis_index]);
    };

    // How many whole bricks the widest gather spans. Measured rather than bounded: a
    // conservative "+1 per axis for misalignment" would cost 112 tiles where 54 suffice at
    // stride == brick, hiding most of what stride buys. Uniform tile counts matter more than
    // a tile saved on one edge brick, so every brick reads this many.
    for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
        const uint32_t brick_extent_sites = config.brick.by_axis[axis_index];
        const uint32_t gather_extent_sites = plan.gather_extent.by_axis[axis_index];

        uint32_t widest_bricks = 1;
        for (uint32_t chunk_index = 0; chunk_index < plan.chunk_count; ++chunk_index) {
            const Site brick_origin = chunk_index_to_origin(chunk_index, plan);
            const uint32_t misalignment_sites = union_origin_for(brick_origin, axis_index) % brick_extent_sites;
            widest_bricks =
                std::max(widest_bricks, ceil_div(misalignment_sites + gather_extent_sites, brick_extent_sites));
        }
        plan.gather_bricks.by_axis[axis_index] = std::min(widest_bricks, plan.volume_bricks.by_axis[axis_index]);
    }
    plan.gather_brick_count = plan.gather_bricks.count();

    plan.gather_origin_by_chunk.reserve(plan.chunk_count);
    for (uint32_t chunk_index = 0; chunk_index < plan.chunk_count; ++chunk_index) {
        const Site brick_origin = chunk_index_to_origin(chunk_index, plan);

        Site gather_origin;
        for (uint32_t axis_index = 0; axis_index < AXIS_COUNT; ++axis_index) {
            const uint32_t brick_extent_sites = config.brick.by_axis[axis_index];

            // A tile-granular read cannot start mid-brick; rounding down only widens the
            // covered range downward, so coverage holds.
            uint32_t origin = (union_origin_for(brick_origin, axis_index) / brick_extent_sites) * brick_extent_sites;

            // Keep the whole-brick span inside the padded volume. When this binds, the span
            // ends exactly at the padded volume, so it still covers every window above it.
            const uint32_t brick_span_sites = plan.gather_bricks.by_axis[axis_index] * brick_extent_sites;
            const uint32_t padded_resident_sites = plan.volume_bricks.by_axis[axis_index] * brick_extent_sites;

            // `origin` is global; the reader addresses the local tensor. A negative shard start
            // shifts the local origin UP, which is exactly how a low-edge halo is addressed.
            const int32_t shard_start = config.shard_origin.by_axis[axis_index];
            const int32_t signed_local = static_cast<int32_t>(origin) - shard_start;
            const uint32_t local_origin = signed_local > 0 ? static_cast<uint32_t>(signed_local) : 0u;
            gather_origin.by_axis[axis_index] = std::min(local_origin, padded_resident_sites - brick_span_sites);
        }
        plan.gather_origin_by_chunk.push_back(gather_origin);
    }

    return plan;
}

}  // namespace ttnn::transformer::neighborhood
