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
    const uint32_t brick[3] = {extents.brick_sites.time(), extents.brick_sites.height(), extents.brick_sites.width()};
    const uint32_t stride[3] = {extents.stride.time, extents.stride.height, extents.stride.width};
    const uint32_t volume[3] = {extents.volume.time, extents.volume.height, extents.volume.width};
    const uint32_t window_config[3] = {
        extents.context_window.time, extents.context_window.height, extents.context_window.width};
    const uint32_t query_origin[3] = {
        query_brick_origin.time(), query_brick_origin.height(), query_brick_origin.width()};
    const uint32_t key_origin[3] = {key_brick_origin.time(), key_brick_origin.height(), key_brick_origin.width()};
    const int32_t shard_origin[3] = {
        extents.shard_origin.time(), extents.shard_origin.height(), extents.shard_origin.width()};
    const uint32_t resident[3] = {extents.resident.time, extents.resident.height, extents.resident.width};

    bool all_visible = true;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const uint32_t window_axis = window_config[axis] < volume[axis] ? window_config[axis] : volume[axis];
        const int32_t shard_base = shard_origin[axis];

        // When the brick holds MANY query groups (stride 1), there is no single window -- but
        // there is a union of them, and a key brick outside that union is invisible to every
        // row. At 11^3 stride 1 the gather is 7x5x5 bricks while the union spans only 6x4x4, so
        // ~45% of gathered bricks are uniformly masked and skip per-element work entirely.
        if (stride[axis] != brick[axis]) {
            const uint32_t first_group = to_global_site(query_origin[axis], shard_base) / stride[axis];
            const uint32_t last_group = to_global_site(query_origin[axis] + brick[axis] - 1, shard_base) / stride[axis];
            const uint32_t union_low = window_origin_on_axis(first_group, stride[axis], window_axis, volume[axis], 0);
            const uint32_t union_high =
                window_origin_on_axis(last_group, stride[axis], window_axis, volume[axis], 0) + window_axis;
            const uint32_t key_first = to_global_site(key_origin[axis], shard_base);
            const uint32_t key_last = key_first + brick[axis] - 1;
            if (key_last < union_low || key_first >= union_high) {
                return BrickCoverage::NoneVisible;
            }
            all_visible = false;  // inside the union, but rows differ: still needs evaluation
            continue;
        }
        const uint32_t window = window_config[axis] < volume[axis] ? window_config[axis] : volume[axis];
        const int32_t shard_start = shard_origin[axis];
        const uint32_t origin = window_origin_on_axis(
            to_global_site(query_origin[axis], shard_start) / stride[axis],
            stride[axis],
            window,
            volume[axis],
            brick[axis]);
        const uint32_t key_first_global = to_global_site(key_origin[axis], shard_start);
        const uint32_t key_last_global = key_first_global + brick[axis] - 1;

        if (key_last_global < origin || key_first_global >= origin + window) {
            return BrickCoverage::NoneVisible;  // disjoint on this axis, so disjoint entirely
        }
        // Ghost sites past what is resident are never visible, so a brick holding any is not uniform.
        const bool inside_window = key_first_global >= origin && key_last_global < origin + window;
        const bool inside_volume = key_origin[axis] + brick[axis] - 1 < resident[axis];
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

    const uint32_t brick[3] = {extents.brick_sites.time(), extents.brick_sites.height(), extents.brick_sites.width()};
    const uint32_t stride[3] = {extents.stride.time, extents.stride.height, extents.stride.width};
    const uint32_t volume[3] = {extents.volume.time, extents.volume.height, extents.volume.width};
    const int32_t shard[3] = {extents.shard_origin.time(), extents.shard_origin.height(), extents.shard_origin.width()};
    const uint32_t resident[3] = {extents.resident.time, extents.resident.height, extents.resident.width};
    const uint32_t window[3] = {
        extents.context_window.time < volume[0] ? extents.context_window.time : volume[0],
        extents.context_window.height < volume[1] ? extents.context_window.height : volume[1],
        extents.context_window.width < volume[2] ? extents.context_window.width : volume[2]};
    const uint32_t query_base[3] = {query_brick_origin.time(), query_brick_origin.height(), query_brick_origin.width()};
    const uint32_t key_base[3] = {key_brick_origin.time(), key_brick_origin.height(), key_brick_origin.width()};
    const uint32_t snap[3] = {
        snap_extent_on_axis(stride[0], brick[0]),
        snap_extent_on_axis(stride[1], brick[1]),
        snap_extent_on_axis(stride[2], brick[2])};

    // ---- per AXIS, not per element ----
    //
    // Visibility is the AND of three independent range tests, and a brick offset takes only
    // brick[axis] values per axis -- 2, 8 and 2 for the shipped brick. So the whole 32x32 tile is
    // decided by brick[0] + brick[1] + brick[2] window resolutions and the same number squared per
    // axis of range tests -- 12 and 72 there -- against 96 resolutions and 6144 tests for the same
    // answer element by element.
    //
    // What the elementwise version actually cost, though, was not the arithmetic but the 1024
    // volatile 16-bit stores. This assembles each row as a 32-bit visibility bitmap and writes 512
    // packed words, the same traffic as a memset. Generating a tile now costs about what copying
    // one does, which is why bypassing the boundary gate entirely (DIFFVAE_NA_TABLE_ALWAYS, all
    // bricks served from the table) measures 0.2 s SLOWER rather than faster.
    uint32_t accept[3][SITES_PER_BRICK_AXIS_MAX];  // accept[axis][query offset] = bitmask over key offsets
    uint32_t key_present[3] = {0, 0, 0};           // key offsets this device actually holds
    uint32_t query_ghost[3] = {0, 0, 0};           // query offsets it does not
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const uint32_t extent = brick[axis];
        uint32_t key_global[SITES_PER_BRICK_AXIS_MAX];
        for (uint32_t offset = 0; offset < extent; ++offset) {
            const uint32_t local = key_base[axis] + offset;
            key_global[offset] = to_global_site(local, shard[axis]);
            if (local < resident[axis]) {
                key_present[axis] |= 1u << offset;
            }
        }
        for (uint32_t offset = 0; offset < extent; ++offset) {
            const uint32_t local = query_base[axis] + offset;
            if (local >= resident[axis]) {
                query_ghost[axis] |= 1u << offset;
            }
            const uint32_t group = to_global_site(local, shard[axis]) / stride[axis];
            const uint32_t origin = window_origin_on_axis(group, stride[axis], window[axis], volume[axis], snap[axis]);
            const uint32_t high = origin + window[axis];
            uint32_t visible = 0;
            for (uint32_t key_offset = 0; key_offset < extent; ++key_offset) {
                if (key_global[key_offset] >= origin && key_global[key_offset] < high) {
                    visible |= 1u << key_offset;
                }
            }
            accept[axis][offset] = visible & key_present[axis];
        }
    }

    // Sites run time-major inside a brick, so a key offset triple lands at bit
    // kt * (Bh * Bw) + kh * Bw + kw. Ghost columns are masked whatever the row says.
    uint32_t all_present = 0;
    for (uint32_t kt = 0; kt < brick[0]; ++kt) {
        if ((key_present[0] & (1u << kt)) == 0) {
            continue;
        }
        for (uint32_t kh = 0; kh < brick[1]; ++kh) {
            if ((key_present[1] & (1u << kh)) == 0) {
                continue;
            }
            // `1u << 32` is undefined, and a (1,1,32) brick reaches it.
            const uint32_t width_bits = brick[2] >= 32 ? 0xFFFFFFFFu : ((1u << brick[2]) - 1u);
            all_present |= (key_present[2] & width_bits) << (kt * brick[1] * brick[2] + kh * brick[2]);
        }
    }

    for (uint32_t query_time = 0; query_time < brick[0]; ++query_time) {
        for (uint32_t query_height = 0; query_height < brick[1]; ++query_height) {
            // The (height, width) slice repeats for every accepted time offset, so build it once.
            uint32_t slice[SITES_PER_BRICK_AXIS_MAX];
            for (uint32_t query_width = 0; query_width < brick[2]; ++query_width) {
                uint32_t bits = 0;
                for (uint32_t kh = 0; kh < brick[1]; ++kh) {
                    if (((accept[1][query_height] >> kh) & 1u) != 0) {
                        bits |= accept[2][query_width] << (kh * brick[2]);
                    }
                }
                slice[query_width] = bits;
            }

            for (uint32_t query_width = 0; query_width < brick[2]; ++query_width) {
                const uint32_t row = query_time * (brick[1] * brick[2]) + query_height * brick[2] + query_width;

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
                    for (uint32_t kt = 0; kt < brick[0]; ++kt) {
                        if (((accept[0][query_time] >> kt) & 1u) != 0) {
                            visible |= slice[query_width] << (kt * brick[1] * brick[2]);
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
