// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The neighborhood-attention window rule, on one axis.
//
// Included by BOTH the host planner (neighborhood_plan.cpp) and the device mask generator
// (kernels/dataflow/neighborhood_mask_gen.hpp), so there is exactly one definition of where a
// context window starts. Two transcriptions of this rule that drift apart produce a kernel
// that reads the wrong keys and still returns plausible-looking video -- which is why this
// header has no includes beyond <cstdint> and no dependency on ttnn or on the kernel API.
//
// The rule: the window keeps its SIZE and slides inward at a boundary rather than truncating.
// A query at site 0 attends to [0, K), not to a half-empty [0, K/2). Consequences: every
// query attends to the same number of keys, every query is inside its own window, and there
// is never anything out of range to mask.

namespace ttnn::transformer::neighborhood {

// Whether the window origin may be snapped to a brick boundary on one axis, and to what.
// Returns the brick extent when snapping is legal and 0 when it is not, so the result feeds
// `window_origin_on_axis` directly.
//
// Legal exactly when a whole brick lies inside ONE query group: then all 32 of its sites share
// a window, and moving that window moves it for all of them together. Query groups begin at
// multiples of the stride, so the condition is simply that the stride is a whole number of
// bricks. That covers stride == brick and every multi-brick query chunk; at stride 1 with a
// brick wider than one site it is false, which is what keeps the shipped architecture exact.
//
// Four places used to spell this out separately -- the planner's window, the planner's gather,
// the device mask, and the host regime masks. They must agree to the site, so it lives here.
inline constexpr uint32_t snap_extent_on_axis(uint32_t stride_extent_sites, uint32_t brick_extent_sites) {
    return (brick_extent_sites != 0 && stride_extent_sites % brick_extent_sites == 0) ? brick_extent_sites : 0u;
}

// Where the context window starts for the query group at `query_group_index`.
// All arguments are in sites, on a single axis.
//
// `window_extent_sites` must already be clamped to the axis (min(context_window, volume));
// an axis shorter than the window is attended to in full.
// `brick_extent_sites` enables BRICK SNAPPING and is only meaningful when a whole query group
// shares one window (stride == brick). Pass 0 to disable.
//
// Centring is not the only valid placement: any origin that keeps the window in bounds and
// still contains the whole group is legal, and several of those are usually brick-aligned. An
// unaligned origin makes a window straddle one extra brick on every axis, which is pure waste
// -- a 12-site window from an origin 3 mod 4 spans 4 bricks instead of 3. Snapping picks an
// aligned one when the valid range allows, so the gathered region can equal the window exactly
// and the interior needs no mask at all.
//
// Use `snap_extent_on_axis` to decide the argument rather than testing the stride by hand.
inline constexpr uint32_t window_origin_on_axis(
    uint32_t query_group_index,
    uint32_t stride_extent_sites,
    uint32_t window_extent_sites,
    uint32_t volume_extent_sites,
    uint32_t brick_extent_sites = 0) {
    const uint32_t group_first_site = query_group_index * stride_extent_sites;
    const uint32_t group_last_site = (group_first_site + stride_extent_sites - 1) < (volume_extent_sites - 1)
                                         ? (group_first_site + stride_extent_sites - 1)
                                         : (volume_extent_sites - 1);

    // Centre on the group, so a group wider than one site is not lopsided in its own window.
    const uint32_t group_centre_site = group_first_site + (group_last_site - group_first_site) / 2;
    const uint32_t half_window_sites = window_extent_sites / 2;
    const uint32_t highest_origin = volume_extent_sites - window_extent_sites;

    // Slide inward rather than truncate. Written with unsigned compares rather than a signed
    // clamp because this also compiles for the device kernels.
    uint32_t origin = 0;
    if (group_centre_site >= half_window_sites) {
        const uint32_t centred_origin = group_centre_site - half_window_sites;
        origin = centred_origin > highest_origin ? highest_origin : centred_origin;
    }

    if (brick_extent_sites <= 1) {
        return origin;
    }

    // The window must still contain the whole group, which bounds how far the origin may move.
    const uint32_t lowest_containing =
        (group_last_site + 1 > window_extent_sites) ? (group_last_site + 1 - window_extent_sites) : 0;
    const uint32_t highest_containing = group_first_site < highest_origin ? group_first_site : highest_origin;

    const uint32_t snapped_down = (origin / brick_extent_sites) * brick_extent_sites;
    if (snapped_down >= lowest_containing) {
        return snapped_down;
    }
    const uint32_t snapped_up = snapped_down + brick_extent_sites;
    if (snapped_up <= highest_containing) {
        return snapped_up;
    }
    return origin;  // no aligned placement contains the group; keep the centred one
}

// True when `key_site` is inside the context window of the query at `query_site`, on one axis.
inline constexpr bool key_is_in_window_on_axis(
    uint32_t query_site,
    uint32_t key_site,
    uint32_t stride_extent_sites,
    uint32_t window_extent_sites,
    uint32_t volume_extent_sites,
    uint32_t brick_extent_sites = 0) {
    const uint32_t origin = window_origin_on_axis(
        query_site / stride_extent_sites,
        stride_extent_sites,
        window_extent_sites,
        volume_extent_sites,
        brick_extent_sites);
    return key_site >= origin && key_site < origin + window_extent_sites;
}

}  // namespace ttnn::transformer::neighborhood
