// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/transformer/sdpa/device/kernels/neighborhood_kernel_args.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/neighborhood_window_rule.hpp"

// Additive attention masks for one (query brick, key brick) pair, generated on device.
//
// Why a mask exists at all: at stride 1 the 32 queries in a brick have 32 DIFFERENT context
// windows, so a single tile of scores contains entries that are valid for some rows and not
// for others. At stride == brick all 32 share one window and every generated tile is
// uniformly zero -- the mask becomes dead weight, which is exactly what the stride sweep is
// meant to reclaim.
//
// Generated rather than uploaded: a per-brick mask table at 1080p would be tens of megabytes
// of DRAM traffic per block. The window rule comes from neighborhood_window_rule.hpp, the
// same header the host planner uses, so device and host cannot disagree about where a window
// starts.
//
// NOTE: this lives under dataflow/ deliberately. The compute kernel consumes mask tiles from
// a circular buffer and knows nothing about context windows, strides, bricks or volumes.

namespace ttnn::transformer::neighborhood::mask_gen {

// Local site -> global site. A device holding a halo at the LOW edge of the volume sits at a
// negative origin, so its first columns map below the volume. Those columns are storage the
// volume does not contain: no query owns them and no window reaches them, so clamping keeps the
// unsigned arithmetic below in range without moving any real query or key.
FORCE_INLINE uint32_t to_global_site(uint32_t local_site, int32_t shard_origin) {
    const int32_t global_site = static_cast<int32_t>(local_site) + shard_origin;
    return global_site > 0 ? static_cast<uint32_t>(global_site) : 0u;
}

// bfloat16 bit patterns. The mask is ADDITIVE: 0 leaves a score alone, -inf drives it to zero
// through the softmax.
constexpr uint16_t KEEP_SCORE = 0x0000;
constexpr uint16_t MASK_SCORE = 0xFF80;  // -infinity

// Positions here are `Site`s from neighborhood_point3.hpp: RESIDENT-local sites, in the same
// units the gather origin table carries. Bricks are laid out time-major then height then width --
// the same order as site_to_bricked_index on the host.

// How much of a key brick a query brick can see. When the whole brick is one query group
// (stride == brick) every row of the mask tile is the same, so a brick that lies wholly inside
// or wholly outside the window needs no per-element work at all -- just a constant fill.
//
// This is what makes a brick-aligned window cheap: with window 12 and brick (2,4,4) the window
// is exactly 6x3x3 bricks starting on a brick boundary, so NOTHING straddles and the interior
// mask is entirely memsets. An 11-wide window cuts through bricks on every axis, so its
// boundary bricks still have to be evaluated site by site.
enum class BrickCoverage : uint8_t { AllVisible, NoneVisible, Mixed };

// Widest a single brick axis can be: the brick holds 32 sites in total.
constexpr uint32_t SITES_PER_BRICK_AXIS_MAX = 32;

inline BrickCoverage classify_brick(
    const Site& query_brick_origin, const Site& key_brick_origin, const kernel_args::NeighborhoodExtents& extents) {
    // Copy the shapes this loop reads into locals. NOT a style choice: `extents` arrives by
    // reference here and is ~21 words by value in fill_mask_tile, so it lives in memory and every
    // `extents.stride[axis]` is a fresh load. A 3-word Shape local is small enough to register
    // allocate, which is what the old uint32_t[3] hoists were buying. Removing them cost 9%
    // (645 -> 703 ms at the stage-5 width-sharded band), so they are back -- just spelled as the
    // shape itself rather than as three loose fields.
    const auto brick_sites = extents.brick_sites;
    const auto stride = extents.stride;
    const auto volume = extents.volume;
    const auto context_window = extents.context_window;
    const auto resident = extents.resident;
    const auto shard_origin = extents.shard_origin;

    bool all_visible = true;
    for (Axis axis : ALL_AXES) {
        const uint32_t window_axis = context_window[axis] < volume[axis] ? context_window[axis] : volume[axis];
        const int32_t shard_base = shard_origin[axis];

        // When the brick holds MANY query groups (stride 1), there is no single window -- but
        // there is a union of them, and a key brick outside that union is invisible to every
        // row. At 11^3 stride 1 the gather is 7x5x5 bricks while the union spans only 6x4x4, so
        // ~45% of gathered bricks are uniformly masked and skip per-element work entirely.
        if (stride[axis] != brick_sites[axis]) {
            const uint32_t first_group = to_global_site(query_brick_origin[axis], shard_base) / stride[axis];
            const uint32_t last_group =
                to_global_site(query_brick_origin[axis] + brick_sites[axis] - 1, shard_base) / stride[axis];
            const uint32_t union_low = window_origin_on_axis(first_group, stride[axis], window_axis, volume[axis], 0);
            const uint32_t union_high =
                window_origin_on_axis(last_group, stride[axis], window_axis, volume[axis], 0) + window_axis;
            const uint32_t key_first = to_global_site(key_brick_origin[axis], shard_base);
            const uint32_t key_last = key_first + brick_sites[axis] - 1;
            if (key_last < union_low || key_first >= union_high) {
                return BrickCoverage::NoneVisible;
            }
            all_visible = false;  // inside the union, but rows differ: still needs evaluation
            continue;
        }
        const uint32_t window = context_window[axis] < volume[axis] ? context_window[axis] : volume[axis];
        const int32_t shard_start = shard_origin[axis];
        const uint32_t origin = window_origin_on_axis(
            to_global_site(query_brick_origin[axis], shard_start) / stride[axis],
            stride[axis],
            window,
            volume[axis],
            brick_sites[axis]);
        const uint32_t key_first_global = to_global_site(key_brick_origin[axis], shard_start);
        const uint32_t key_last_global = key_first_global + brick_sites[axis] - 1;

        if (key_last_global < origin || key_first_global >= origin + window) {
            return BrickCoverage::NoneVisible;  // disjoint on this axis, so disjoint entirely
        }
        // Ghost sites past what is resident are never visible, so a brick holding any is not uniform.
        const bool inside_window = key_first_global >= origin && key_last_global < origin + window;
        const bool inside_volume = key_brick_origin[axis] + brick_sites[axis] - 1 < resident[axis];
        all_visible = all_visible && inside_window && inside_volume;
    }
    return all_visible ? BrickCoverage::AllVisible : BrickCoverage::Mixed;
}

// Fill one 32x32 bfloat16 tile: rows are the query brick's sites, columns the key brick's.
//
// A Float16_b tile is four row-major 16x16 faces, so an element at (row, column) is not at
// row * 32 + column. Getting this wrong yields a mask that is subtly transposed within each
// quadrant -- correct along the diagonal, wrong everywhere else.
// Taken BY VALUE and force-inlined so the caller's compile-time extents fold in.
FORCE_INLINE void fill_mask_tile(
    uint32_t write_address, Site query_brick_origin, Site key_brick_origin, kernel_args::NeighborhoodExtents extents) {
    constexpr uint32_t FACE_HEIGHT = 16;
    constexpr uint32_t FACE_WIDTH = 16;
    // A 16-bit half of one face row, as bfloat16 pairs packed into words. Indexed by two MASK
    // bits, low column first, because the low uint16 of a little-endian word is the lower column.
    constexpr uint32_t PAIR[4] = {0x00000000u, 0x0000FF80u, 0xFF800000u, 0xFF80FF80u};

    volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_address);

    // Shapes copied into 3-word locals -- see the note in classify_brick: `extents` is too big to
    // register allocate, so reading through it per axis measured 9% slower.
    //
    // `window` and `snap` are different again: they are COMPUTED per axis (clamp to the volume, and
    // the brick-snapping rule), not a respelling of a member, and the element loops read them once
    // per tile rather than recomputing them 1024 times.
    const auto brick_sites = extents.brick_sites;
    const auto stride = extents.stride;
    const auto volume = extents.volume;
    const auto context_window = extents.context_window;
    const auto resident = extents.resident;
    const auto shard_origin = extents.shard_origin;
    const uint32_t window[3] = {
        context_window[Axis::Time] < volume[Axis::Time] ? context_window[Axis::Time] : volume[Axis::Time],
        context_window[Axis::Height] < volume[Axis::Height] ? context_window[Axis::Height] : volume[Axis::Height],
        context_window[Axis::Width] < volume[Axis::Width] ? context_window[Axis::Width] : volume[Axis::Width]};
    const uint32_t snap[3] = {
        snap_extent_on_axis(stride[Axis::Time], brick_sites[Axis::Time]),
        snap_extent_on_axis(stride[Axis::Height], brick_sites[Axis::Height]),
        snap_extent_on_axis(stride[Axis::Width], brick_sites[Axis::Width])};

    // The element loops below walk the tile with CONSTANT axis subscripts, so these three are for
    // brevity, not to dodge an indexed access.
    const uint32_t brick_time = brick_sites[Axis::Time];
    const uint32_t brick_height = brick_sites[Axis::Height];
    const uint32_t brick_width = brick_sites[Axis::Width];

    // ---- per AXIS, not per element ----
    //
    // Visibility is the AND of three independent range tests, and a brick offset takes only
    // brick_sites[axis] values per axis -- 2, 8 and 2 for the shipped brick. So the whole 32x32 tile is
    // decided by brick_time + brick_height + brick_width window resolutions and the same number squared per
    // axis of range tests -- 12 and 72 there -- against 96 resolutions and 6144 tests for the same
    // answer element by element.
    //
    // What the elementwise version actually cost, though, was not the arithmetic but the 1024
    // volatile 16-bit stores. This assembles each row as a 32-bit visibility bitmap and writes 512
    // packed words, the same traffic as a memset. Generating a tile now costs about what copying
    // one does, which is why bypassing the boundary gate entirely (DIFFVAE_NA_TABLE_ALWAYS, all
    // bricks served from the table) measures 0.2 s SLOWER rather than faster.
    uint32_t accept[3][SITES_PER_BRICK_AXIS_MAX];  // accept[static_cast<uint32_t>(axis)][query offset] = bitmask over
                                                   // key offsets
    uint32_t key_present[3] = {0, 0, 0};           // key offsets this device actually holds
    uint32_t query_ghost[3] = {0, 0, 0};           // query offsets it does not
    for (Axis axis : ALL_AXES) {
        const uint32_t extent = brick_sites[axis];
        uint32_t key_global[SITES_PER_BRICK_AXIS_MAX];
        for (uint32_t offset = 0; offset < extent; ++offset) {
            const uint32_t local = key_brick_origin[axis] + offset;
            key_global[offset] = to_global_site(local, shard_origin[axis]);
            if (local < resident[axis]) {
                key_present[static_cast<uint32_t>(axis)] |= 1u << offset;
            }
        }
        for (uint32_t offset = 0; offset < extent; ++offset) {
            const uint32_t local = query_brick_origin[axis] + offset;
            if (local >= resident[axis]) {
                query_ghost[static_cast<uint32_t>(axis)] |= 1u << offset;
            }
            const uint32_t group = to_global_site(local, shard_origin[axis]) / stride[axis];
            const uint32_t origin = window_origin_on_axis(
                group,
                stride[axis],
                window[static_cast<uint32_t>(axis)],
                volume[axis],
                snap[static_cast<uint32_t>(axis)]);
            const uint32_t high = origin + window[static_cast<uint32_t>(axis)];
            uint32_t visible = 0;
            for (uint32_t key_offset = 0; key_offset < extent; ++key_offset) {
                if (key_global[key_offset] >= origin && key_global[key_offset] < high) {
                    visible |= 1u << key_offset;
                }
            }
            accept[static_cast<uint32_t>(axis)][offset] = visible & key_present[static_cast<uint32_t>(axis)];
        }
    }

    // Sites run time-major inside a brick, so a key offset triple lands at bit
    // kt * (Bh * Bw) + kh * Bw + kw. Ghost columns are masked whatever the row says.
    uint32_t all_present = 0;
    for (uint32_t kt = 0; kt < brick_time; ++kt) {
        if ((key_present[0] & (1u << kt)) == 0) {
            continue;
        }
        for (uint32_t kh = 0; kh < brick_height; ++kh) {
            if ((key_present[1] & (1u << kh)) == 0) {
                continue;
            }
            // `1u << 32` is undefined, and a (1,1,32) brick reaches it.
            const uint32_t width_bits = brick_width >= 32 ? 0xFFFFFFFFu : ((1u << brick_width) - 1u);
            all_present |= (key_present[2] & width_bits) << (kt * brick_height * brick_width + kh * brick_width);
        }
    }

    for (uint32_t query_time = 0; query_time < brick_time; ++query_time) {
        for (uint32_t query_height = 0; query_height < brick_height; ++query_height) {
            // The (height, width) slice repeats for every accepted time offset, so build it once.
            uint32_t slice[SITES_PER_BRICK_AXIS_MAX];
            for (uint32_t query_width = 0; query_width < brick_width; ++query_width) {
                uint32_t bits = 0;
                for (uint32_t kh = 0; kh < brick_height; ++kh) {
                    if (((accept[1][query_height] >> kh) & 1u) != 0) {
                        bits |= accept[2][query_width] << (kh * brick_width);
                    }
                }
                slice[query_width] = bits;
            }

            for (uint32_t query_width = 0; query_width < brick_width; ++query_width) {
                const uint32_t row =
                    query_time * (brick_height * brick_width) + query_height * brick_width + query_width;

                // A ghost query's output is discarded, but its row is still softmaxed and an all
                // -inf row yields NaN, which propagates through the rescale into real
                // accumulators. Leave ghost rows OPEN rather than fully masked.
                uint32_t visible;
                if (((query_ghost[0] >> query_time) | (query_ghost[1] >> query_height) |
                     (query_ghost[2] >> query_width)) &
                    1u) {
                    visible = all_present;
                } else {
                    visible = 0;
                    for (uint32_t kt = 0; kt < brick_time; ++kt) {
                        if (((accept[0][query_time] >> kt) & 1u) != 0) {
                            visible |= slice[query_width] << (kt * brick_height * brick_width);
                        }
                    }
                }
                const uint32_t masked = ~visible;

                // Float16_b tiles are four row-major 16x16 faces, so one tile row is two runs of
                // 16 -- columns 0-15 in one face, 16-31 in the next. Getting this wrong yields a
                // mask that is subtly transposed within each quadrant: right on the diagonal,
                // wrong everywhere else.
                const uint32_t face_row_base = (row >= FACE_HEIGHT) ? 2u : 0u;
                const uint32_t row_in_face = row % FACE_HEIGHT;
                const uint32_t words_per_face = (FACE_HEIGHT * FACE_WIDTH) / 2;
                const uint32_t words_per_row = FACE_WIDTH / 2;
                uint32_t out = face_row_base * words_per_face + row_in_face * words_per_row;
                for (uint32_t word = 0; word < words_per_row; ++word) {
                    tile[out + word] = PAIR[(masked >> (2u * word)) & 3u];
                }
                out = (face_row_base + 1u) * words_per_face + row_in_face * words_per_row;
                for (uint32_t word = 0; word < words_per_row; ++word) {
                    tile[out + word] = PAIR[(masked >> (FACE_WIDTH + 2u * word)) & 3u];
                }
            }
        }
    }
}

}  // namespace ttnn::transformer::neighborhood::mask_gen
