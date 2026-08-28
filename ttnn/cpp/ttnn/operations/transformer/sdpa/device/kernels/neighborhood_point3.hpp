// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

// The (time, height, width) triples the neighborhood geometry is built from: POSITIONS, SHAPES,
// and the conversions between the units either can be measured in.
//
// Included by BOTH the host planner (neighborhood_plan.hpp) and the device kernels
// (neighborhood_kernel_args.hpp, neighborhood_chunk_layout.hpp, dataflow/neighborhood_mask_gen.hpp),
// for the same reason neighborhood_window_rule.hpp is: a triple used to be spelled seven different
// ways -- Site, Offset3, SignedAxisOffsets, SiteInBrick, BrickCoordinate, Extent3, AxisExtents --
// which were distinct types by NAME only. Nothing stopped a brick coordinate being written into a
// site, or a volume being used where a brick shape belonged, and those mistakes are silent: the
// mask attends to the wrong keys and still returns plausible video.
//
// Hence no includes beyond <array> and <cstdint>, no dependency on ttnn or on the kernel API, and
// `inline constexpr` rather than FORCE_INLINE, which does not exist host-side.
//
// A POSITION and a SHAPE are separate types, and a shape carries no origin. That is not an
// omission. In this design the gathered region's extent is the SAME for every query chunk -- the
// window slides inward at a boundary rather than truncating -- so the extent is a plan-wide
// constant while only the origin varies per chunk. Keeping them apart is what collapses a
// per-chunk 6-tuple of bounds into one shared shape plus a 3-word table row.
//
// The windowed SDPA path next door cannot do that: its chunks are flat token runs rather than 3D
// boxes, so its box extent varies per chunk (H and W fall back to the whole axis when a chunk
// straddles a frame or a row) and it must carry lo/hi bounds. See NeighborhoodBox in
// windowed_loop_geometry.hpp -- a genuinely different concept despite the similar shape.

namespace ttnn::transformer::neighborhood {

constexpr uint32_t AXIS_COUNT = 3;

enum class Axis : uint32_t { Time = 0, Height = 1, Width = 2 };

// For `for (Axis axis : ALL_AXES)`. Both Point3 and Shape index by Axis straight off a std::array,
// so a per-axis loop reads them directly instead of hoisting into a local uint32_t[3] first.
constexpr std::array<Axis, AXIS_COUNT> ALL_AXES{Axis::Time, Axis::Height, Axis::Width};

// What a triple COUNTS. Sites, bricks and chunks are all (T,H,W) triples and none is
// interchangeable with another: one brick is 32 sites in some 3D arrangement, one chunk is a box of
// bricks. Assigning across units silently scales by the brick or chunk shape -- which is why this
// is a type parameter and not a comment.
//
// `None` is only ever the PER of a plain shape; see Shape.
//
// The tag is PHANTOM: it appears in the type and nowhere in the object, so everything below is the
// same three-word aggregate the hand-written structs were, and folds at compile time the same way.
enum class Unit : uint8_t { Sites, Bricks, Chunks, None };

// ---- positions ----

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
// symmetric halo puts the device at the low edge of the volume at -halo, and those columns are real
// storage that simply lies outside the volume. Its queries never use them and its windows never
// reach them, but the local -> global conversion still has to be able to say where it is.
using SiteOffset = Point3<int32_t, Unit::Sites>;

// A position, in bricks -- equivalently a tile row of the bricked tensor.
using BrickPoint = Point3<uint32_t, Unit::Bricks>;

// A position, in query chunks.
using ChunkPoint = Point3<uint32_t, Unit::Chunks>;

// ---- shapes ----
//
// MEASURED_IN is the unit the three numbers count. PER names the COARSER unit this is the shape of
// exactly one of; `None` means a plain region shape rather than a conversion factor.
//
// That second parameter is what keeps `first_point_of(brick_point, config.volume)` from compiling:
// the volume is a region (PER == None), not the shape of one brick. A tagged POSITION with an
// untyped shape checks only half of each conversion, and the half left open is the one that
// silently rescales a position.
template <Unit MEASURED_IN, Unit PER = Unit::None>
struct Shape {
    std::array<uint32_t, AXIS_COUNT> by_axis{0, 0, 0};

    static constexpr Shape of(uint32_t time, uint32_t height, uint32_t width) { return Shape{{time, height, width}}; }
    constexpr uint32_t time() const { return by_axis[0]; }
    constexpr uint32_t height() const { return by_axis[1]; }
    constexpr uint32_t width() const { return by_axis[2]; }
    constexpr uint32_t operator[](Axis axis) const { return by_axis[static_cast<uint32_t>(axis)]; }
    constexpr uint32_t& operator[](Axis axis) { return by_axis[static_cast<uint32_t>(axis)]; }

    // MEASURED_IN units spanned. True for every instantiation, which the old Extent3::sites() was
    // not: on a brick grid it returned a brick count, on a chunk grid a chunk count.
    constexpr uint32_t count() const { return by_axis[0] * by_axis[1] * by_axis[2]; }

    friend constexpr bool operator==(const Shape& left, const Shape& right) { return left.by_axis == right.by_axis; }
};

// Plain region shapes.
using ShapeInSites = Shape<Unit::Sites>;    // volume, context window, stride, gather extent
using ShapeInBricks = Shape<Unit::Bricks>;  // volume_bricks, query_bricks, gather_bricks
using ShapeInChunks = Shape<Unit::Chunks>;  // the chunk grid

// Unit shapes: conversion factors, not regions. The layout unit holds SITES_PER_BRICK sites; the
// chunk shape is the knob deciding how far one gather amortises.
using BrickShapeInSites = Shape<Unit::Sites, Unit::Bricks>;
using ChunkShapeInBricks = Shape<Unit::Bricks, Unit::Chunks>;

// ---- unit conversions ----
//
// One pair, with BOTH units read off the unit shape, so these serve every level: brick -> site and
// chunk -> brick are the same function. Replaces six hand-written triples spread over the reader
// and the planner, each an opportunity to transpose height and width or to scale by the wrong shape.

// Where one COARSE unit BEGINS, in FINE units.
template <Unit FINE, Unit COARSE>
inline constexpr Point3<uint32_t, FINE> first_point_of(Point3<uint32_t, COARSE> point, Shape<FINE, COARSE> unit_shape) {
    return Point3<uint32_t, FINE>::at(
        point.time() * unit_shape.time(), point.height() * unit_shape.height(), point.width() * unit_shape.width());
}

// Which COARSE unit contains this FINE position. Rounds DOWN, which is what a tile-granular read
// needs: one brick is one tile row, and a read cannot start mid-row.
template <Unit FINE, Unit COARSE>
inline constexpr Point3<uint32_t, COARSE> containing_unit(
    Point3<uint32_t, FINE> point, Shape<FINE, COARSE> unit_shape) {
    return Point3<uint32_t, COARSE>::at(
        point.time() / unit_shape.time(), point.height() / unit_shape.height(), point.width() / unit_shape.width());
}

// Named wrappers. The return unit in the name is worth keeping at a call site.
inline constexpr Site first_site_of(BrickPoint brick, BrickShapeInSites brick_shape) {
    return first_point_of(brick, brick_shape);
}
inline constexpr BrickPoint first_brick_of(ChunkPoint chunk, ChunkShapeInBricks chunk_shape) {
    return first_point_of(chunk, chunk_shape);
}
inline constexpr BrickPoint containing_brick(Site site, BrickShapeInSites brick_shape) {
    return containing_unit(site, brick_shape);
}

}  // namespace ttnn::transformer::neighborhood
