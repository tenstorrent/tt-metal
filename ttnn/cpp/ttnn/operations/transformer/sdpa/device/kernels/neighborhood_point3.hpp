// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

// A point in (time, height, width), the unit shapes that rescale one, and the conversions
// between the units it can be measured in.
//
// Included by BOTH the host planner (neighborhood_plan.hpp) and the device kernels
// (neighborhood_kernel_args.hpp, neighborhood_chunk_layout.hpp, dataflow/neighborhood_mask_gen.hpp),
// for the same reason neighborhood_window_rule.hpp is: a position was previously spelled five
// different ways -- Site, Offset3, SignedAxisOffsets, SiteInBrick, BrickCoordinate -- which were
// distinct types by NAME only. Nothing stopped a brick coordinate being written into a site, and
// that mistake is silent: the mask attends to the wrong keys and still returns plausible video.
//
// Hence no includes beyond <array> and <cstdint>, no dependency on ttnn or on the kernel API, and
// `inline constexpr` rather than FORCE_INLINE, which does not exist host-side.

namespace ttnn::transformer::neighborhood {

constexpr uint32_t AXIS_COUNT = 3;

enum class Axis : uint32_t { Time = 0, Height = 1, Width = 2 };

// What a coordinate COUNTS. Sites, bricks and chunks are all (T,H,W) triples and none of them is
// interchangeable with another: one brick is 32 sites in some 3D arrangement, and one chunk is a
// box of bricks. Assigning across units silently scales a position by the brick or the chunk
// shape -- which is why this is a type and not a comment.
//
// The tag is PHANTOM: it appears in the type and nowhere in the object. Every alias below is the
// same three-word aggregate the hand-written structs were, and folds at compile time the same way.
enum class Unit : uint8_t { Sites, Bricks, Chunks };

template <typename Scalar, Unit UNIT>
struct Point3 {
    std::array<Scalar, AXIS_COUNT> by_axis{0, 0, 0};

    static constexpr Point3 at(Scalar time, Scalar height, Scalar width) { return Point3{{time, height, width}}; }
    constexpr Scalar time() const { return by_axis[0]; }
    constexpr Scalar height() const { return by_axis[1]; }
    constexpr Scalar width() const { return by_axis[2]; }
    constexpr Scalar operator[](Axis axis) const { return by_axis[static_cast<uint32_t>(axis)]; }
    constexpr Scalar& operator[](Axis axis) { return by_axis[static_cast<uint32_t>(axis)]; }

    friend constexpr bool operator==(const Point3& left, const Point3& right) { return left.by_axis == right.by_axis; }

    // Same unit only, which is the point: brick + brick is a brick, and brick + site does not
    // compile at all.
    friend constexpr Point3 operator+(const Point3& left, const Point3& right) {
        return Point3::at(
            left.by_axis[0] + right.by_axis[0], left.by_axis[1] + right.by_axis[1], left.by_axis[2] + right.by_axis[2]);
    }
    friend constexpr Point3 operator-(const Point3& left, const Point3& right) {
        return Point3::at(
            left.by_axis[0] - right.by_axis[0], left.by_axis[1] - right.by_axis[1], left.by_axis[2] - right.by_axis[2]);
    }
};

// A position, in sites.
using Site = Point3<uint32_t, Unit::Sites>;

// A SIGNED position, in sites. Distinct from Site because a shard's origin can be NEGATIVE: a
// symmetric halo puts the device at the low edge of the volume at -halo, and those columns are
// real storage that simply lies outside the volume. Its queries never use them and its windows
// never reach them, but the local -> global conversion still has to be able to say where it is.
using SiteOffset = Point3<int32_t, Unit::Sites>;

// A position, in bricks. One brick is 32 sites, so this is a position in the BRICKED tensor --
// equivalently a tile row -- and it is a factor of the brick shape away from a Site.
using BrickPoint = Point3<uint32_t, Unit::Bricks>;

// A position, in query chunks. One chunk is a box of bricks that shares a gather, a mask and a
// flash pass.
using ChunkPoint = Point3<uint32_t, Unit::Chunks>;

// ---- unit ratios ----
//
// How many FINE units make up one COARSE unit. The aliases below name both halves --
// BrickShapeInSites is a brick measured in sites, ChunkShapeInBricks a chunk measured in bricks --
// because at a call site the fine unit is what the conversion actually multiplies or divides by.
// These are the SECOND argument to every conversion, and they are deliberately NOT Extent3.
//
// An Extent3 is a SIZE -- "how big is this region". A ratio is a SCALE -- "what do I multiply by
// to change units". They are the same three numbers, which is exactly the problem: while the
// conversions took any extent-shaped type, `first_site_of(brick_point, config.volume)` and
// `first_brick_of(chunk_point, config.brick)` both compiled, and both are wrong by a factor of
// the whole volume or of the brick shape. A tagged POINT with an untyped extent checks only half
// of each call, and the half it leaves open is the one that silently rescales a position.
template <Unit COARSE, Unit FINE>
struct UnitRatio {
    std::array<uint32_t, AXIS_COUNT> by_axis{1, 1, 1};

    static constexpr UnitRatio of(uint32_t time, uint32_t height, uint32_t width) {
        return UnitRatio{{time, height, width}};
    }
    constexpr uint32_t time() const { return by_axis[0]; }
    constexpr uint32_t height() const { return by_axis[1]; }
    constexpr uint32_t width() const { return by_axis[2]; }
    constexpr uint32_t operator[](Axis axis) const { return by_axis[static_cast<uint32_t>(axis)]; }

    // FINE units per one COARSE unit -- 32 sites for a BrickShapeInSites, the bricks in one
    // chunk for a ChunkShapeInBricks. Not spelled sites(), which would be a false claim on the
    // second; the alias name already says which unit the answer is in.
    constexpr uint32_t product() const { return by_axis[0] * by_axis[1] * by_axis[2]; }

    friend constexpr bool operator==(const UnitRatio& left, const UnitRatio& right) {
        return left.by_axis == right.by_axis;
    }
};

// The layout unit: one brick is this many sites, so product() == SITES_PER_BRICK.
using BrickShapeInSites = UnitRatio<Unit::Bricks, Unit::Sites>;

// One query chunk is this many bricks -- the knob deciding how far one gather amortises.
using ChunkShapeInBricks = UnitRatio<Unit::Chunks, Unit::Bricks>;

// ---- unit conversions ----
//
// The ONLY route between units, and the only place the multiply or divide by a unit shape
// appears. Both arguments are typed, so neither the position nor the scale can be the wrong
// thing: these were six hand-written triples spread over the reader and the planner, each one an
// opportunity to transpose height and width or to scale by the wrong extent entirely.

// Where a brick BEGINS, in sites.
inline constexpr Site first_site_of(BrickPoint brick, BrickShapeInSites brick_shape) {
    return Site::at(
        brick.time() * brick_shape.time(), brick.height() * brick_shape.height(), brick.width() * brick_shape.width());
}

// Where a chunk BEGINS, in bricks.
inline constexpr BrickPoint first_brick_of(ChunkPoint chunk, ChunkShapeInBricks chunk_shape) {
    return BrickPoint::at(
        chunk.time() * chunk_shape.time(), chunk.height() * chunk_shape.height(), chunk.width() * chunk_shape.width());
}

// The brick holding a site. Rounds DOWN, which is what a tile-granular read needs: one brick is
// one tile row, and a read cannot start mid-row.
inline constexpr BrickPoint containing_brick(Site site, BrickShapeInSites brick_shape) {
    return BrickPoint::at(
        site.time() / brick_shape.time(), site.height() / brick_shape.height(), site.width() / brick_shape.width());
}

}  // namespace ttnn::transformer::neighborhood
