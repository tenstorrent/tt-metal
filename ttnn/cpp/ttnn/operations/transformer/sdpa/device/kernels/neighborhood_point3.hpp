// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

// A point in (time, height, width), and the conversions between the units one can be measured in.
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

// ---- unit conversions ----
//
// Templated on the extent type so one definition serves both the host's Extent3 and the kernel's
// AxisExtents; both provide operator[](Axis). These were six hand-written triples spread over the
// reader and the planner, each one an opportunity to transpose height and width or to forget the
// scale entirely.

// Where a brick BEGINS, in sites.
template <typename ExtentT>
inline constexpr Site first_site_of(BrickPoint brick, const ExtentT& brick_sites) {
    return Site::at(
        brick.time() * brick_sites[Axis::Time],
        brick.height() * brick_sites[Axis::Height],
        brick.width() * brick_sites[Axis::Width]);
}

// Where a chunk BEGINS, in bricks.
template <typename ExtentT>
inline constexpr BrickPoint first_brick_of(ChunkPoint chunk, const ExtentT& chunk_bricks) {
    return BrickPoint::at(
        chunk.time() * chunk_bricks[Axis::Time],
        chunk.height() * chunk_bricks[Axis::Height],
        chunk.width() * chunk_bricks[Axis::Width]);
}

// The brick holding a site. Rounds DOWN, which is what a tile-granular read needs: one brick is
// one tile row, and a read cannot start mid-row.
template <typename ExtentT>
inline constexpr BrickPoint containing_brick(Site site, const ExtentT& brick_sites) {
    return BrickPoint::at(
        site.time() / brick_sites[Axis::Time],
        site.height() / brick_sites[Axis::Height],
        site.width() / brick_sites[Axis::Width]);
}

}  // namespace ttnn::transformer::neighborhood
