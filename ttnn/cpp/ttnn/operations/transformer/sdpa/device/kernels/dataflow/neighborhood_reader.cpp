// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "neighborhood_mask_gen.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/neighborhood_chunk_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/neighborhood_kernel_args.hpp"

// Factory define: 1 = interior TU, 2 = edge TU, 0 = unsplit. Must be a preprocessor value so
// classify is absent from the interior ELF (if constexpr on path_mode was DCE'd).
#ifndef NA_PATH_KIND
#define NA_PATH_KIND 0
#endif
// Factory -DNA_PATH_KIND=1/2 is the reliable skip switch: wrapper #defines were not
// enough (the included reader still walked every brick). 2 = skip edges, 3 = skip interiors.
#ifndef NA_SKIP_IF
#if NA_PATH_KIND == 1
#define NA_SKIP_IF 2u
#elif NA_PATH_KIND == 2
#define NA_SKIP_IF 3u
#else
#define NA_SKIP_IF 0u
#endif
#endif

// Feeds one query CHUNK at a time: the Q tiles of every brick in the chunk, then the context
// window's K and V tiles in chunks, with a matching additive mask.
//
// The chunk is the unit that makes this affordable. Its bricks form ONE query group, so they
// share one window: the gather happens once for the whole chunk instead of once per brick, and
// the mask is one tile per gather slot that broadcasts down the rows. That is the difference
// between 54 keys gathered per query and 4.8.
//
// Sites arrive BRICKED, so one brick is exactly one tile row and the key axis is measured in
// bricks -- "tiles_per_kv_chunk" is literally how many key bricks the compute kernel gets per
// flash step. This is the only kernel that knows what a context window is; the compute kernel
// downstream sees tiles and a mask.

namespace kernel_args = ttnn::transformer::neighborhood::kernel_args;
namespace mask_gen = ttnn::transformer::neighborhood::mask_gen;
namespace layout = ttnn::transformer::neighborhood::chunk_layout;

namespace {

// tiles_per_kv_chunk is bounded by DST capacity (8), so the per-chunk scratch is fixed size.
constexpr uint32_t MAX_TILES_PER_KV_CHUNK = 8;

using layout::BrickCoordinate;

// The key brick a gather slot names, and where it starts in sites. W fastest, then H, then T --
// the same raster the planner uses. The hot gather loop must NOT call this per slot: those
// divides are ~64 ns each and dominate the op. Decode the chunk start once, then
// `advance_gather_raster`.
FORCE_INLINE BrickCoordinate gather_slot_brick(
    const BrickCoordinate& gather_origin_brick, uint32_t slot, const kernel_args::AxisExtents& gather_bricks) {
    const uint32_t per_time_slice = gather_bricks.height * gather_bricks.width;
    const uint32_t within = slot % per_time_slice;
    return BrickCoordinate{
        gather_origin_brick.time + slot / per_time_slice,
        gather_origin_brick.height + within / gather_bricks.width,
        gather_origin_brick.width + within % gather_bricks.width};
}

// Next brick in gather raster. Along W the bricked tensor is contiguous, so the brick index is
// just +1; H/T wraps recompute. Used by the hot loop so it never divides a slot index.
FORCE_INLINE void advance_gather_raster(
    BrickCoordinate& key_brick,
    uint32_t& key_brick_index,
    const BrickCoordinate& gather_origin_brick,
    const kernel_args::AxisExtents& gather_bricks,
    const kernel_args::AxisExtents& volume_bricks) {
    key_brick.width += 1;
    if (key_brick.width != gather_origin_brick.width + gather_bricks.width) {
        key_brick_index += 1;
        return;
    }
    key_brick.width = gather_origin_brick.width;
    key_brick.height += 1;
    if (key_brick.height == gather_origin_brick.height + gather_bricks.height) {
        key_brick.height = gather_origin_brick.height;
        key_brick.time += 1;
    }
    key_brick_index = layout::brick_index(key_brick, volume_bricks);
}

FORCE_INLINE mask_gen::SiteInBrick gather_slot_origin(
    const BrickCoordinate& gather_origin_brick,
    uint32_t slot,
    const kernel_args::AxisExtents& gather_bricks,
    const kernel_args::NeighborhoodExtents& extents) {
    const BrickCoordinate brick = gather_slot_brick(gather_origin_brick, slot, gather_bricks);
    return mask_gen::SiteInBrick{
        brick.time * extents.brick_sites.time,
        brick.height * extents.brick_sites.height,
        brick.width * extents.brick_sites.width};
}

// A query brick clamped on no axis and wholly resident. For those the mask pattern is the same
// as every other interior brick's, so the uploaded tensor applies.
// Which uploaded mask set applies to this query brick, or NO_REGIME when none does.
//
// A brick's mask depends on its position only through CLAMPING: every brick whose queries all
// clamp low shares one window origin (0) and one gather origin, so they share a pattern. Same
// for all-clamp-high, and for none-clamped (interior). Three classes per axis, 27 in 3D --
// which is the same collapse the reference gets by grouping query tiles by window geometry.
// A brick straddling a transition has no shared pattern and must be evaluated; there is at
// most one such brick per edge per axis.
constexpr uint32_t NO_REGIME = 0xFFFFFFFFu;

// Inclusive range of (key_brick - query_brick) a window can reach on one axis. MUST match
// relative_mask_span in neighborhood_attention.py, which writes the tiles this indexes.
FORCE_INLINE int32_t relative_span_low(uint32_t window_extent, uint32_t brick_extent) {
    const uint32_t half = window_extent / 2;
    return -static_cast<int32_t>((half + brick_extent - 1) / brick_extent);
}

FORCE_INLINE int32_t relative_span_high(uint32_t window_extent, uint32_t brick_extent) {
    const uint32_t half = window_extent / 2;
    return static_cast<int32_t>((window_extent - 1 - half + brick_extent - 1) / brick_extent);
}

// Tile index into the RELATIVE table for this (query brick, key brick) pair, or NO_REGIME when
// the pair falls outside it. Mirrors the linearisation in _build_relative_masks.
FORCE_INLINE uint32_t relative_table_index(
    const mask_gen::SiteInBrick& query_origin_site,
    const mask_gen::SiteInBrick& key_origin_site,
    const kernel_args::NeighborhoodExtents& extents) {
    const uint32_t brick[3] = {extents.brick_sites.time, extents.brick_sites.height, extents.brick_sites.width};
    const uint32_t window[3] = {
        extents.context_window.time, extents.context_window.height, extents.context_window.width};
    const int32_t relative[3] = {
        static_cast<int32_t>(key_origin_site.time / brick[0]) - static_cast<int32_t>(query_origin_site.time / brick[0]),
        static_cast<int32_t>(key_origin_site.height / brick[1]) -
            static_cast<int32_t>(query_origin_site.height / brick[1]),
        static_cast<int32_t>(key_origin_site.width / brick[2]) -
            static_cast<int32_t>(query_origin_site.width / brick[2])};

    uint32_t index = 0;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const int32_t low = relative_span_low(window[axis], brick[axis]);
        const int32_t high = relative_span_high(window[axis], brick[axis]);
        if (relative[axis] < low || relative[axis] > high) {
            return NO_REGIME;
        }
        index = index * static_cast<uint32_t>(high - low + 1) + static_cast<uint32_t>(relative[axis] - low);
    }
    return index;
}

// Does gather slot `s` name relative offset `s`, for every slot?
//
// It does exactly when the gather origin sits at the LOW end of the relative span on every axis
// and spans the span's extent, because both are linearised time-major over the same extents. Then
// no per-slot arithmetic is needed at all -- table page == gather slot -- and, more to the point,
// the mapping is the same for every such chunk, so the tiles can be written once and reused.
//
// It is NOT automatic for an unclamped brick. `build_plan` clamps the gather origin into the local
// tensor (`max(origin - shard_start, 0)`), and for a brick sitting in this device's HALO the
// window reaches below local 0, so the origin lands short and the whole mapping shifts. Those
// bricks are not owned by this device and their output is sliced off, which is why a wrong mask
// there was harmless -- until reuse let one of them decide the tiles a real brick reads.
FORCE_INLINE bool gather_is_canonical(
    const BrickCoordinate& gather_origin_brick,
    const mask_gen::SiteInBrick& query_origin_site,
    const kernel_args::AxisExtents& gather_bricks,
    const kernel_args::NeighborhoodExtents& extents) {
    const uint32_t brick[3] = {extents.brick_sites.time, extents.brick_sites.height, extents.brick_sites.width};
    const uint32_t window[3] = {
        extents.context_window.time, extents.context_window.height, extents.context_window.width};
    const uint32_t gather[3] = {gather_bricks.time, gather_bricks.height, gather_bricks.width};
    const uint32_t origin[3] = {gather_origin_brick.time, gather_origin_brick.height, gather_origin_brick.width};
    const uint32_t query[3] = {
        query_origin_site.time / brick[0], query_origin_site.height / brick[1], query_origin_site.width / brick[2]};

    for (uint32_t axis = 0; axis < 3; ++axis) {
        const int32_t low = relative_span_low(window[axis], brick[axis]);
        const int32_t high = relative_span_high(window[axis], brick[axis]);
        if (static_cast<int32_t>(origin[axis]) - static_cast<int32_t>(query[axis]) != low) {
            return false;
        }
        if (gather[axis] != static_cast<uint32_t>(high - low + 1)) {
            return false;
        }
    }
    return true;
}

// Is this query brick far enough from every volume edge that none of its 32 queries clamps? Only
// then does the relative table describe it; a clamped brick's window sits at 0 or at the high
// stop and no longer centres on the query, so the kernel still generates those. At most one
// brick per edge per axis.
FORCE_INLINE bool brick_window_is_unclamped(
    const mask_gen::SiteInBrick& query_origin_site, const kernel_args::NeighborhoodExtents& extents) {
    const uint32_t brick[3] = {extents.brick_sites.time, extents.brick_sites.height, extents.brick_sites.width};
    const uint32_t volume[3] = {extents.volume.time, extents.volume.height, extents.volume.width};
    const uint32_t configured[3] = {
        extents.context_window.time, extents.context_window.height, extents.context_window.width};
    const int32_t shard[3] = {extents.shard_origin.time, extents.shard_origin.height, extents.shard_origin.width};
    const uint32_t local[3] = {query_origin_site.time, query_origin_site.height, query_origin_site.width};
    const uint32_t resident[3] = {extents.resident.time, extents.resident.height, extents.resident.width};

    for (uint32_t axis = 0; axis < 3; ++axis) {
        const uint32_t window = configured[axis] < volume[axis] ? configured[axis] : volume[axis];
        if (window >= volume[axis]) {
            return false;
        }
        // A brick hanging off what this shard holds carries ghost rows, which the table's
        // always-visible interior pattern does not describe.
        if (local[axis] + brick[axis] > resident[axis]) {
            return false;
        }
        const int32_t first = static_cast<int32_t>(local[axis]) + shard[axis];
        const int32_t last = first + static_cast<int32_t>(brick[axis]) - 1;
        const int32_t half = static_cast<int32_t>(window / 2);
        if (first < 0) {
            return false;  // a low-edge halo brick: below the volume, no window of its own
        }
        // Ask the window rule itself rather than re-deriving its bounds: the table describes a
        // window that sits exactly half a window below its query, so it applies precisely when
        // the rule returns that for both ends of the brick (it is monotonic between them).
        // Re-deriving this by hand admitted only ~10% of bricks where ~90% qualify.
        const uint32_t origin_first = ttnn::transformer::neighborhood::window_origin_on_axis(
            static_cast<uint32_t>(first), 1u, window, volume[axis], 0u);
        const uint32_t origin_last = ttnn::transformer::neighborhood::window_origin_on_axis(
            static_cast<uint32_t>(last), 1u, window, volume[axis], 0u);
        if (origin_first != static_cast<uint32_t>(first - half) || origin_last != static_cast<uint32_t>(last - half)) {
            return false;
        }
    }
    return true;
}

// Which uploaded mask set applies to this query chunk, or NO_REGIME when none does.
//
// The scan is over the CHUNK, not one brick: the chunk is the unit that shares a window, so it
// is the unit whose clamping behaviour decides the pattern. Scanning a brick would give the
// right answer only when the chunk is one brick.
FORCE_INLINE uint32_t
chunk_regime(const mask_gen::SiteInBrick& chunk_origin_site, const kernel_args::NeighborhoodExtents& extents) {
    const uint32_t group[3] = {extents.query_chunk.time, extents.query_chunk.height, extents.query_chunk.width};
    const uint32_t brick[3] = {extents.brick_sites.time, extents.brick_sites.height, extents.brick_sites.width};
    const uint32_t stride[3] = {extents.stride.time, extents.stride.height, extents.stride.width};
    const uint32_t volume[3] = {extents.volume.time, extents.volume.height, extents.volume.width};
    const int32_t shard[3] = {extents.shard_origin.time, extents.shard_origin.height, extents.shard_origin.width};
    const uint32_t resident[3] = {extents.resident.time, extents.resident.height, extents.resident.width};
    const uint32_t configured[3] = {
        extents.context_window.time, extents.context_window.height, extents.context_window.width};
    const uint32_t local[3] = {chunk_origin_site.time, chunk_origin_site.height, chunk_origin_site.width};

    uint32_t regime = 0;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const uint32_t window = configured[axis] < volume[axis] ? configured[axis] : volume[axis];
        if (window >= volume[axis]) {
            return NO_REGIME;
        }
        // A chunk that overhangs what is resident carries ghosts, which the shared patterns do
        // not describe.
        if (local[axis] + group[axis] > resident[axis]) {
            return NO_REGIME;
        }
        const uint32_t snap = ttnn::transformer::neighborhood::snap_extent_on_axis(stride[axis], brick[axis]);
        const uint32_t highest = volume[axis] - window;

        bool all_low = true;
        bool all_high = true;
        bool all_centred = true;
        for (uint32_t offset = 0; offset < group[axis]; ++offset) {
            // A brick in a low-edge halo sits below the volume; it has no window and is never
            // read, so clamping here just keeps the arithmetic in range.
            const int32_t signed_global = static_cast<int32_t>(local[axis] + offset) + shard[axis];
            const uint32_t global = signed_global > 0 ? static_cast<uint32_t>(signed_global) : 0u;
            const uint32_t origin = ttnn::transformer::neighborhood::window_origin_on_axis(
                global / stride[axis], stride[axis], window, volume[axis], snap);
            all_low = all_low && origin == 0;
            all_high = all_high && origin == highest;
            all_centred = all_centred && origin != 0 && origin != highest;
        }
        const uint32_t axis_class = all_low ? 0u : (all_centred ? 1u : (all_high ? 2u : NO_REGIME));
        if (axis_class == NO_REGIME) {
            return NO_REGIME;
        }
        regime = regime * 3 + axis_class;
    }
    return regime;
}

// Out of kernel_main so the interior gather loop does not share an I-cache working set with
// classify/fill_mask_tile. Compiling both in one function was 64 ns/slot (416 ms) even when
// the host bit admitted 73% interior; compiling the interior path alone was 20 ns/slot (130 ms).
#if NA_PATH_KIND != 1
template <typename KeyReader, typename ValueReader, typename MaskReader>
__attribute__((noinline, noclone)) void gather_edge_flash_chunk(
    Noc& noc,
    const KeyReader& key_reader,
    const ValueReader& value_reader,
    const MaskReader& interior_mask_reader,
    kernel_args::NeighborhoodExtents& extents,
    const BrickCoordinate& gather_origin_brick,
    const layout::BrickCoordinate& chunk_origin,
    const mask_gen::SiteInBrick& chunk_origin_site,
    const kernel_args::AxisExtents& gather_bricks,
    const kernel_args::AxisExtents& volume_bricks,
    const kernel_args::AxisExtents& query_chunk_bricks,
    const kernel_args::AxisExtents& query_origin_bricks,
    CircularBuffer& cb_key,
    CircularBuffer& cb_value,
    CircularBuffer& cb_mask,
    uint32_t kv_chunk_index,
    uint32_t batch_index,
    uint32_t head_index,
    uint32_t brick_count,
    uint32_t head_count,
    uint32_t head_dim_tiles,
    uint32_t tiles_per_kv_chunk,
    uint32_t kv_chunk_count,
    uint32_t gather_brick_count,
    uint32_t bricks_per_query_chunk,
    uint32_t tile_bytes,
    uint32_t skip_kv,
    uint32_t per_brick_mask,
    uint32_t mask_memset_only,
    uint32_t relative_mask,
    uint32_t table_always,
    uint32_t use_uploaded_mask,
    uint32_t regime) {
    const uint32_t key_base_pointer = cb_key.get_write_ptr();
    uint32_t value_write_pointer = cb_value.get_write_ptr();
    uint32_t mask_write_pointer = cb_mask.get_write_ptr();

    mask_gen::BrickCoverage coverage[MAX_TILES_PER_KV_CHUNK];
    mask_gen::SiteInBrick key_origins[MAX_TILES_PER_KV_CHUNK];

    BrickCoordinate key_brick;
    if (tiles_per_kv_chunk == gather_bricks.width && kv_chunk_count == gather_bricks.time * gather_bricks.height) {
        key_brick = BrickCoordinate{
            gather_origin_brick.time + kv_chunk_index / gather_bricks.height,
            gather_origin_brick.height + kv_chunk_index % gather_bricks.height,
            gather_origin_brick.width};
    } else {
        const uint32_t start_slot = kv_chunk_index * tiles_per_kv_chunk;
        key_brick = start_slot >= gather_brick_count
                        ? gather_origin_brick
                        : gather_slot_brick(gather_origin_brick, start_slot, gather_bricks);
    }
    uint32_t key_brick_index = layout::brick_index(key_brick, volume_bricks);

    for (uint32_t slot = 0; slot < tiles_per_kv_chunk; ++slot) {
        const uint32_t gather_slot = kv_chunk_index * tiles_per_kv_chunk + slot;
        const bool slot_is_padding = gather_slot >= gather_brick_count;
        const BrickCoordinate& slot_brick = slot_is_padding ? gather_origin_brick : key_brick;
        const uint32_t slot_brick_index =
            slot_is_padding ? layout::brick_index(gather_origin_brick, volume_bricks) : key_brick_index;

        key_origins[slot] = mask_gen::SiteInBrick{
            slot_brick.time * extents.brick_sites.time,
            slot_brick.height * extents.brick_sites.height,
            slot_brick.width * extents.brick_sites.width};
        coverage[slot] = slot_is_padding ? mask_gen::BrickCoverage::NoneVisible
                                         : mask_gen::classify_brick(chunk_origin_site, key_origins[slot], extents);

        const uint32_t key_first_tile =
            layout::tile_offset(batch_index, slot_brick_index, head_index, brick_count, head_count, head_dim_tiles);
        if (skip_kv == 0) {
            for (uint32_t head_dim_tile = 0; head_dim_tile < head_dim_tiles; ++head_dim_tile) {
                const uint32_t key_write_pointer =
                    key_base_pointer + (head_dim_tile * tiles_per_kv_chunk + slot) * tile_bytes;
                noc.async_read(
                    key_reader,
                    CoreLocalMem<uint32_t>(key_write_pointer),
                    tile_bytes,
                    {.page_id = key_first_tile + head_dim_tile},
                    {});
                noc.async_read(
                    value_reader,
                    CoreLocalMem<uint32_t>(value_write_pointer),
                    tile_bytes,
                    {.page_id = key_first_tile + head_dim_tile},
                    {});
                value_write_pointer += tile_bytes;
            }
        } else {
            value_write_pointer += head_dim_tiles * tile_bytes;
        }
        if (slot + 1 < tiles_per_kv_chunk && !slot_is_padding) {
            advance_gather_raster(key_brick, key_brick_index, gather_origin_brick, gather_bricks, volume_bricks);
        }
    }

    if (per_brick_mask != 0) {
        for (uint32_t brick_in_chunk = 0; brick_in_chunk < bricks_per_query_chunk; ++brick_in_chunk) {
            const layout::BrickCoordinate query_brick =
                layout::brick_within_chunk(brick_in_chunk, chunk_origin, query_chunk_bricks);
            const mask_gen::SiteInBrick query_origin_site{
                (query_brick.time + query_origin_bricks.time) * extents.brick_sites.time,
                (query_brick.height + query_origin_bricks.height) * extents.brick_sites.height,
                (query_brick.width + query_origin_bricks.width) * extents.brick_sites.width};
            const uint32_t brick_base = mask_write_pointer + brick_in_chunk * tiles_per_kv_chunk * tile_bytes;
            const bool brick_takes_table = relative_mask != 0 && use_uploaded_mask != 0 &&
                                           (table_always != 0 || brick_window_is_unclamped(query_origin_site, extents));
            for (uint32_t slot = 0; slot < tiles_per_kv_chunk; ++slot) {
                const uint32_t gather_slot = kv_chunk_index * tiles_per_kv_chunk + slot;
                if (mask_memset_only != 0) {
                    volatile tt_l1_ptr uint32_t* zero_destination =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(brick_base + slot * tile_bytes);
                    for (uint32_t word = 0; word < tile_bytes / sizeof(uint32_t); ++word) {
                        zero_destination[word] = 0x00000000u;
                    }
                    continue;
                }
                const mask_gen::BrickCoverage brick_coverage =
                    gather_slot >= gather_brick_count
                        ? mask_gen::BrickCoverage::NoneVisible
                        : mask_gen::classify_brick(query_origin_site, key_origins[slot], extents);
                if (brick_coverage == mask_gen::BrickCoverage::Mixed) {
                    if (brick_takes_table) {
                        const uint32_t table_index =
                            relative_table_index(query_origin_site, key_origins[slot], extents);
                        if (table_index != NO_REGIME) {
                            noc.async_read(
                                interior_mask_reader,
                                CoreLocalMem<uint32_t>(brick_base + slot * tile_bytes),
                                tile_bytes,
                                {.page_id = table_index},
                                {});
                            continue;
                        }
                    }
                    mask_gen::fill_mask_tile(
                        brick_base + slot * tile_bytes, query_origin_site, key_origins[slot], extents);
                    continue;
                }
                const uint32_t fill = brick_coverage == mask_gen::BrickCoverage::AllVisible ? 0x00000000u : 0xFF80FF80u;
                volatile tt_l1_ptr uint32_t* destination =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(brick_base + slot * tile_bytes);
                for (uint32_t word = 0; word < tile_bytes / sizeof(uint32_t); ++word) {
                    destination[word] = fill;
                }
            }
        }
        noc.async_read_barrier();
        cb_key.push_back(tiles_per_kv_chunk * head_dim_tiles);
        cb_value.push_back(tiles_per_kv_chunk * head_dim_tiles);
        cb_mask.push_back(per_brick_mask != 0 ? bricks_per_query_chunk * tiles_per_kv_chunk : tiles_per_kv_chunk);
        return;
    }

    for (uint32_t slot = 0; slot < tiles_per_kv_chunk; ++slot) {
        if (coverage[slot] == mask_gen::BrickCoverage::Mixed) {
            continue;
        }
        const uint32_t fill = coverage[slot] == mask_gen::BrickCoverage::AllVisible ? 0x00000000u : 0xFF80FF80u;
        volatile tt_l1_ptr uint32_t* destination =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_write_pointer + slot * tile_bytes);
        for (uint32_t word = 0; word < tile_bytes / sizeof(uint32_t); ++word) {
            destination[word] = fill;
        }
    }

    if (use_uploaded_mask != 0 && relative_mask == 0) {
        for (uint32_t slot = 0; slot < tiles_per_kv_chunk; ++slot) {
            if (coverage[slot] != mask_gen::BrickCoverage::Mixed) {
                continue;
            }
            const uint32_t gather_slot = kv_chunk_index * tiles_per_kv_chunk + slot;
            const uint32_t page = regime * gather_brick_count + gather_slot;
            noc.async_read(
                interior_mask_reader,
                CoreLocalMem<uint32_t>(mask_write_pointer + slot * tile_bytes),
                tile_bytes,
                {.page_id = page},
                {});
        }
    } else {
        for (uint32_t slot = 0; slot < tiles_per_kv_chunk; ++slot) {
            if (coverage[slot] != mask_gen::BrickCoverage::Mixed) {
                continue;
            }
            mask_gen::fill_mask_tile(
                mask_write_pointer + slot * tile_bytes, chunk_origin_site, key_origins[slot], extents);
        }
    }

    noc.async_read_barrier();
    cb_key.push_back(tiles_per_kv_chunk * head_dim_tiles);
    cb_value.push_back(tiles_per_kv_chunk * head_dim_tiles);
    cb_mask.push_back(tiles_per_kv_chunk);
}
#endif  // NA_PATH_KIND != 1

}  // namespace

#ifndef NA_HAS_PATH_SKIP
// Unsplit path_mode 0: process every chunk. Split wrappers define NA_SKIP_IF before including us.
__attribute__((noinline, noclone)) bool na_path_skips_chunk(uint32_t) { return false; }
#endif

#ifdef NA_SKIP_IF
template <uint32_t SkipIf>
__attribute__((noinline, noclone)) bool na_skip_kind(uint32_t packed_width) {
    return (2u + (packed_width >> 31)) == SkipIf;
}
#endif

__attribute__((noinline, noclone)) bool na_should_skip(uint32_t packed_width, uint32_t skip_if) {
    return (2u + (packed_width >> 31)) == skip_if;
}

__attribute__((noinline, noclone)) void handshake_skip_work_item(
    CircularBuffer& cb_key,
    CircularBuffer& cb_value,
    CircularBuffer& cb_mask,
    CircularBuffer& cb_gather_origin,
    uint32_t kv_chunk_count,
    uint32_t kv_pages,
    uint32_t mask_pages) {
    for (uint32_t kv_chunk_index = 0; kv_chunk_index < kv_chunk_count; ++kv_chunk_index) {
        cb_key.reserve_back(kv_pages);
        cb_value.reserve_back(kv_pages);
        cb_mask.reserve_back(mask_pages);
        cb_key.push_back(kv_pages);
        cb_value.push_back(kv_pages);
        cb_mask.push_back(mask_pages);
    }
    cb_gather_origin.push_back(1);
    cb_gather_origin.pop_front(1);
}

void kernel_main() {
    constexpr uint32_t head_count = get_compile_time_arg_val(kernel_args::reader_arg::head_count);
    constexpr uint32_t brick_count = get_compile_time_arg_val(kernel_args::reader_arg::brick_count);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(kernel_args::reader_arg::head_dim_tiles);
    constexpr uint32_t bricks_per_query_chunk =
        get_compile_time_arg_val(kernel_args::reader_arg::bricks_per_query_chunk);
    constexpr kernel_args::AxisExtents query_chunk_bricks{
        get_compile_time_arg_val(kernel_args::reader_arg::query_chunk_bricks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::query_chunk_bricks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::query_chunk_bricks_width)};
    constexpr kernel_args::AxisExtents volume_chunks{
        get_compile_time_arg_val(kernel_args::reader_arg::volume_chunks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::volume_chunks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::volume_chunks_width)};
    constexpr uint32_t chunk_count = volume_chunks.time * volume_chunks.height * volume_chunks.width;
    constexpr uint32_t tiles_per_kv_chunk = get_compile_time_arg_val(kernel_args::reader_arg::tiles_per_kv_chunk);
    constexpr uint32_t kv_chunk_count = get_compile_time_arg_val(kernel_args::reader_arg::kv_chunk_count);
    constexpr uint32_t gather_brick_count = get_compile_time_arg_val(kernel_args::reader_arg::gather_brick_count);

    constexpr kernel_args::AxisExtents volume_bricks{
        get_compile_time_arg_val(kernel_args::reader_arg::volume_bricks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::volume_bricks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::volume_bricks_width)};
    // Q lives on the query grid; K, V and the gather live on the resident grid above. Equal
    // unless the host asked for a query sub-region.
    constexpr kernel_args::AxisExtents query_bricks{
        get_compile_time_arg_val(kernel_args::reader_arg::query_bricks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::query_bricks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::query_bricks_width)};
    constexpr uint32_t query_brick_count = get_compile_time_arg_val(kernel_args::reader_arg::query_brick_count);
    constexpr kernel_args::AxisExtents query_origin_bricks{
        get_compile_time_arg_val(kernel_args::reader_arg::query_origin_bricks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::query_origin_bricks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::query_origin_bricks_width)};
    constexpr kernel_args::AxisExtents gather_bricks{
        get_compile_time_arg_val(kernel_args::reader_arg::gather_bricks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::gather_bricks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::gather_bricks_width)};
    // Not constexpr: `shard_origin` is filled in per chunk from the gather origin table, because
    // it is the one geometric value that differs per device and the mesh runs one program.
    // Everything else here is a compile-time constant and still folds.
    kernel_args::NeighborhoodExtents extents{
        {get_compile_time_arg_val(kernel_args::reader_arg::brick_sites_time),
         get_compile_time_arg_val(kernel_args::reader_arg::brick_sites_height),
         get_compile_time_arg_val(kernel_args::reader_arg::brick_sites_width)},
        {get_compile_time_arg_val(kernel_args::reader_arg::context_window_time),
         get_compile_time_arg_val(kernel_args::reader_arg::context_window_height),
         get_compile_time_arg_val(kernel_args::reader_arg::context_window_width)},
        {get_compile_time_arg_val(kernel_args::reader_arg::stride_time),
         get_compile_time_arg_val(kernel_args::reader_arg::stride_height),
         get_compile_time_arg_val(kernel_args::reader_arg::stride_width)},
        {get_compile_time_arg_val(kernel_args::reader_arg::volume_time),
         get_compile_time_arg_val(kernel_args::reader_arg::volume_height),
         get_compile_time_arg_val(kernel_args::reader_arg::volume_width)},
        {get_compile_time_arg_val(kernel_args::reader_arg::query_chunk_bricks_time) *
             get_compile_time_arg_val(kernel_args::reader_arg::brick_sites_time),
         get_compile_time_arg_val(kernel_args::reader_arg::query_chunk_bricks_height) *
             get_compile_time_arg_val(kernel_args::reader_arg::brick_sites_height),
         get_compile_time_arg_val(kernel_args::reader_arg::query_chunk_bricks_width) *
             get_compile_time_arg_val(kernel_args::reader_arg::brick_sites_width)},
        {0, 0, 0},  // shard_origin: filled from the table below
        {get_compile_time_arg_val(kernel_args::reader_arg::resident_time),
         get_compile_time_arg_val(kernel_args::reader_arg::resident_height),
         get_compile_time_arg_val(kernel_args::reader_arg::resident_width)}};

    constexpr auto query_accessor_args = TensorAccessorArgs<kernel_args::reader_arg::COUNT>();
    constexpr auto key_accessor_args = TensorAccessorArgs<query_accessor_args.next_compile_time_args_offset()>();
    constexpr auto value_accessor_args = TensorAccessorArgs<key_accessor_args.next_compile_time_args_offset()>();
    constexpr uint32_t has_interior_mask = get_compile_time_arg_val(kernel_args::reader_arg::has_interior_mask);
    // Per-brick masks: one tile per (query brick, gather slot) rather than one per slot shared by
    // the chunk. Needed once the chunk is wider than the stride, because then each brick centres a
    // DIFFERENT window and a shared mask would attend to the wrong one.
    constexpr uint32_t per_brick_mask = get_compile_time_arg_val(kernel_args::reader_arg::per_brick_mask);
    constexpr uint32_t mask_memset_only = get_compile_time_arg_val(kernel_args::reader_arg::mask_memset_only);
    constexpr uint32_t skip_kv = get_compile_time_arg_val(kernel_args::reader_arg::skip_kv);
    constexpr uint32_t skip_unowned = get_compile_time_arg_val(kernel_args::reader_arg::skip_unowned);
    constexpr uint32_t skip_if_bit = get_compile_time_arg_val(kernel_args::reader_arg::skip_if_bit);
    constexpr uint32_t relative_mask = get_compile_time_arg_val(kernel_args::reader_arg::relative_mask);
    constexpr uint32_t table_always = get_compile_time_arg_val(kernel_args::reader_arg::table_always);
    constexpr auto origin_accessor_args = TensorAccessorArgs<value_accessor_args.next_compile_time_args_offset()>();
    constexpr auto interior_mask_args = TensorAccessorArgs<origin_accessor_args.next_compile_time_args_offset()>();

    // Which regime's set is currently sitting in cb_resident_mask, so a run of chunks that share
    // a regime -- which is nearly all of them, the interior being one regime -- fetches once.
    uint32_t resident_regime = NO_REGIME;
    CircularBuffer cb_resident_mask(kernel_args::cb_resident_mask);

    // ---- the interior mask, WRITTEN ONCE and then left alone ----
    //
    // Every unclamped query brick has the same mask, tile for tile. Its window sits exactly half
    // a window below it, so the planner's gather origin sits at a CONSTANT brick offset from the
    // brick (floor((F - half) / B) - F / B, which does not depend on F because F is brick
    // aligned), and the relative table is keyed on that same difference. So slot -> table page is
    // one fixed permutation for the whole plan, and the 175 tiles it selects are the entire mask
    // for ~75% of the bricks at 1080p.
    //
    // In persistent_mask mode cb_mask holds a whole work item, so its pages cycle back to the
    // same addresses every item: once those tiles ARE the table, the next unclamped brick needs
    // no writes at all. Only leaving an edge shell dirties them, which at 1080p happens a few
    // dozen times per core against 458 work items.
    //
    // Worth 15.2 s against 15.6 s at 145 frames. Less than the 350 KB per work item it saves
    // would suggest, because what the op is actually bound by is the number of SCORE TILES the
    // compute kernel walks -- see FINDINGS section 10 -- but it is free and it is real.
    //
    // This predicate is ALSO what the program factory sizes cb_mask by, so the two cannot drift:
    // there is no compile arg for the mode. Adding one is not free either -- the reader's five
    // TensorAccessorArgs chain off reader_arg::COUNT, and moving it by one ran the last accessor
    // off the end of the compile-arg vector.
    constexpr bool interior_table_supported = relative_mask != 0 && has_interior_mask != 0 && per_brick_mask == 0;
    constexpr uint32_t path_mode = get_compile_time_arg_val(kernel_args::reader_arg::path_mode);
    // Classify and the tight gather MUST NOT share a binary: that I-cache mix is 64 ns/slot.
    // NA_PATH_KIND is the wrapper TU / factory define, not a path_mode compile-arg compare --
    // those were DCE'd and both programs compiled the same loop.
#if NA_PATH_KIND == 2
    constexpr bool compile_interior = false;
#elif NA_PATH_KIND == 1
    constexpr bool compile_interior = true;
#else
    // path 0 matches the old single program (interior-only when a relative table is in play).
    constexpr bool compile_interior = interior_table_supported && path_mode != 2;
#endif
    constexpr bool compile_edge = !compile_interior;
    bool mask_pages_hold_table = false;
    constexpr uint32_t mask_tiles_per_kv_chunk =
        per_brick_mask != 0 ? bricks_per_query_chunk * tiles_per_kv_chunk : tiles_per_kv_chunk;

    uint32_t argument_index = 0;
    const uint32_t query_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t key_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t value_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t gather_origin_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t interior_mask_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t work_item_start = get_arg_val<uint32_t>(argument_index++);
    const uint32_t work_item_count = get_arg_val<uint32_t>(argument_index++);
    const uint32_t tile_and_skip = get_arg_val<uint32_t>(argument_index++);
    const uint32_t tile_bytes = tile_and_skip & 0xffffu;
    const uint32_t skip_if_runtime = tile_and_skip >> 16;

    const auto query_reader = TensorAccessor(query_accessor_args, query_address);
    const auto key_reader = TensorAccessor(key_accessor_args, key_address);
    const auto value_reader = TensorAccessor(value_accessor_args, value_address);
    const auto origin_reader = TensorAccessor(origin_accessor_args, gather_origin_address);
    const auto interior_mask_reader = TensorAccessor(interior_mask_args, interior_mask_address);

    CircularBuffer cb_query(kernel_args::cb_query);
    CircularBuffer cb_key(kernel_args::cb_key);
    CircularBuffer cb_value(kernel_args::cb_value);
    CircularBuffer cb_mask(kernel_args::cb_mask);
    CircularBuffer cb_gather_origin(kernel_args::cb_gather_origin);

    Noc noc;

    for (uint32_t work_item = work_item_start; work_item < work_item_start + work_item_count; ++work_item) {
        // A work item is one (batch, head, query brick). Bricks vary fastest so that
        // neighbouring bricks -- which share most of their context window -- land on the same
        // core, which is what later lets their K/V stay resident.
        const uint32_t chunk_index = work_item % chunk_count;
        const uint32_t head_index = (work_item / chunk_count) % head_count;
        const uint32_t batch_index = work_item / (chunk_count * head_count);

        const layout::BrickCoordinate chunk_origin =
            layout::chunk_origin_brick(chunk_index, volume_chunks, query_chunk_bricks);

        // ---- Q: one tile row per brick in the chunk ----
        cb_query.reserve_back(head_dim_tiles * bricks_per_query_chunk);
        uint32_t query_write_pointer = cb_query.get_write_ptr();
        for (uint32_t brick_in_chunk = 0; brick_in_chunk < bricks_per_query_chunk; ++brick_in_chunk) {
            const layout::BrickCoordinate brick =
                layout::brick_within_chunk(brick_in_chunk, chunk_origin, query_chunk_bricks);
            // A chunk on the far edge can hang off the volume. Those rows have no queries to
            // read; the writer drops them again, and flash rows are independent so whatever
            // stale L1 they carry cannot reach a real query.
            if (!layout::brick_is_inside(brick, query_bricks)) {
                query_write_pointer += head_dim_tiles * tile_bytes;
                continue;
            }
            const uint32_t first_tile = layout::tile_offset(
                batch_index,
                layout::brick_index(brick, query_bricks),
                head_index,
                query_brick_count,
                head_count,
                head_dim_tiles);
            for (uint32_t head_dim_tile = 0; head_dim_tile < head_dim_tiles; ++head_dim_tile) {
                noc.async_read(
                    query_reader,
                    CoreLocalMem<uint32_t>(query_write_pointer),
                    tile_bytes,
                    {.page_id = first_tile + head_dim_tile},
                    {});
                query_write_pointer += tile_bytes;
            }
        }

        // ---- where this brick's context window starts, from the host-built plan ----
        cb_gather_origin.reserve_back(1);
        const uint32_t origin_write_pointer = cb_gather_origin.get_write_ptr();
        noc.async_read(
            origin_reader,
            CoreLocalMem<uint32_t>(origin_write_pointer),
            kernel_args::GATHER_ORIGIN_ROW_BYTES,
            {.page_id = chunk_index},
            {});
        noc.async_read_barrier();
#ifdef NA_HAS_PATH_SKIP
        // Same L1 dest the NOC just filled. A later origin_row[2] load ran after a1 was reused,
        // so the skip compared the wrong word and both programs walked (~530 ms).
        {
            CoreLocalMem<volatile uint32_t> origin_mem(origin_write_pointer);
            volatile uint32_t edge_token = origin_mem[kernel_args::gather_origin_column::skip_edge_token];
            if constexpr (compile_interior) {
                if (edge_token == 0xFFFFFFFFu) {
                    cb_query.push_back(head_dim_tiles * bricks_per_query_chunk);
                    handshake_skip_work_item(
                        cb_key,
                        cb_value,
                        cb_mask,
                        cb_gather_origin,
                        kv_chunk_count,
                        tiles_per_kv_chunk * head_dim_tiles,
                        mask_tiles_per_kv_chunk);
                    mask_pages_hold_table = false;
                    continue;
                }
            } else {
                if (edge_token != 0xFFFFFFFFu) {
                    cb_query.push_back(head_dim_tiles * bricks_per_query_chunk);
                    handshake_skip_work_item(
                        cb_key,
                        cb_value,
                        cb_mask,
                        cb_gather_origin,
                        kv_chunk_count,
                        tiles_per_kv_chunk * head_dim_tiles,
                        mask_tiles_per_kv_chunk);
                    mask_pages_hold_table = false;
                    continue;
                }
            }
        }
#endif
        cb_query.push_back(head_dim_tiles * bricks_per_query_chunk);

        const volatile tt_l1_ptr uint32_t* origin_row =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(origin_write_pointer);
        namespace column = kernel_args::gather_origin_column;
        const uint32_t gather_width_packed = origin_row[column::gather_width];
        // Origins are rounded down to a brick boundary by the planner, so this division is exact.
        // Low 31 bits of gather_width are the origin; the high bit is the interior flag.
        const BrickCoordinate gather_origin_brick{
            origin_row[column::gather_time] / extents.brick_sites.time,
            origin_row[column::gather_height] / extents.brick_sites.height,
            (gather_width_packed & 0x7fffffffu) / extents.brick_sites.width};

        // Where this device sits in the global volume. Addressing above is LOCAL; window
        // placement below is GLOBAL, and this is what converts between them.
        extents.shard_origin = kernel_args::SignedAxisOffsets{
            static_cast<int32_t>(origin_row[column::shard_origin_time]),
            static_cast<int32_t>(origin_row[column::shard_origin_height]),
            static_cast<int32_t>(origin_row[column::shard_origin_width])};

        // RESIDENT-local, like query_origin_site below and like the gather table's key origins.
        const mask_gen::SiteInBrick chunk_origin_site{
            (chunk_origin.time + query_origin_bricks.time) * extents.brick_sites.time,
            (chunk_origin.height + query_origin_bricks.height) * extents.brick_sites.height,
            (chunk_origin.width + query_origin_bricks.width) * extents.brick_sites.width};

        // The relative table needs no regime: it is keyed on (key_brick - query_brick), which is
        // defined for every chunk. Only the per-brick clamping test below gates it.
        const uint32_t regime =
            (has_interior_mask != 0 && relative_mask == 0) ? chunk_regime(chunk_origin_site, extents) : NO_REGIME;
        const bool use_uploaded_mask = (relative_mask != 0) ? (has_interior_mask != 0) : (regime != NO_REGIME);

        // Resolved ONCE per chunk, not once per slot. It depends only on the query brick, and
        // asking it per slot put a branch to fill_mask_tile next to the table read in the same
        // loop body -- the instruction-cache mix the mask loops are split three ways to avoid.
        // That alone was 32.3 s against 15.6 s at 145 frames, on a gate that admits 75% of bricks
        // and a plan where only 20% of mask tiles ever generated.
        (void)table_always;
        (void)skip_unowned;
        (void)skip_if_bit;
        (void)skip_if_runtime;
        (void)compile_edge;
        // The pages already hold exactly these tiles, so there is nothing to write.
        const bool refill_mask = compile_interior && !mask_pages_hold_table;

        uint32_t resident_mask_pointer = 0;
        if (use_uploaded_mask && relative_mask == 0) {
            resident_mask_pointer = cb_resident_mask.get_write_ptr();
            if (regime != resident_regime) {
                for (uint32_t slot = 0; slot < gather_brick_count; ++slot) {
                    noc.async_read(
                        interior_mask_reader,
                        CoreLocalMem<uint32_t>(resident_mask_pointer + slot * tile_bytes),
                        tile_bytes,
                        {.page_id = regime * gather_brick_count + slot},
                        {});
                }
                noc.async_read_barrier();
                resident_regime = regime;
            }
        }

        // ---- K, V and the mask, one flash chunk at a time ----
        for (uint32_t kv_chunk_index = 0; kv_chunk_index < kv_chunk_count; ++kv_chunk_index) {
            cb_key.reserve_back(tiles_per_kv_chunk * head_dim_tiles);
            cb_value.reserve_back(tiles_per_kv_chunk * head_dim_tiles);
            cb_mask.reserve_back(mask_tiles_per_kv_chunk);

            // Probe 7/8: keep the CB handshake (reserve/push the same counts) but skip classify,
            // tile_offset, mask fill and K/V noc. Drain/qk could not isolate this walk because they
            // still waited on these CBs after the reader filled them.
            if (skip_kv == 2) {
                cb_key.push_back(tiles_per_kv_chunk * head_dim_tiles);
                cb_value.push_back(tiles_per_kv_chunk * head_dim_tiles);
                cb_mask.push_back(mask_tiles_per_kv_chunk);
                continue;
            }

            // Interior: the relative mask is already in cb_mask (or about to be refilled).
            // Classify / key_origins / coverage are dead. Keep this loop a handful of
            // async_reads -- sharing a body with fill_mask_tile is what made the 147-slot
            // walk 64 ns/slot (skip_slots is 26 ms for the same handshake + compute).
#if NA_PATH_KIND != 2
            if constexpr (compile_interior) {
                if (skip_kv == 0) {
                    constexpr uint32_t page_stride = head_count * head_dim_tiles;
                    const uint32_t key_base_pointer = cb_key.get_write_ptr();
                    uint32_t value_write_pointer = cb_value.get_write_ptr();
                    if constexpr (
                        tiles_per_kv_chunk == gather_bricks.width &&
                        kv_chunk_count == gather_bricks.time * gather_bricks.height) {
                        const BrickCoordinate row{
                            gather_origin_brick.time + kv_chunk_index / gather_bricks.height,
                            gather_origin_brick.height + kv_chunk_index % gather_bricks.height,
                            gather_origin_brick.width};
                        uint32_t page = layout::tile_offset(
                            batch_index,
                            layout::brick_index(row, volume_bricks),
                            head_index,
                            brick_count,
                            head_count,
                            head_dim_tiles);
                        for (uint32_t slot = 0; slot < tiles_per_kv_chunk; ++slot) {
                            for (uint32_t head_dim_tile = 0; head_dim_tile < head_dim_tiles; ++head_dim_tile) {
                                noc.async_read(
                                    key_reader,
                                    CoreLocalMem<uint32_t>(
                                        key_base_pointer + (head_dim_tile * tiles_per_kv_chunk + slot) * tile_bytes),
                                    tile_bytes,
                                    {.page_id = page + head_dim_tile},
                                    {});
                                noc.async_read(
                                    value_reader,
                                    CoreLocalMem<uint32_t>(value_write_pointer),
                                    tile_bytes,
                                    {.page_id = page + head_dim_tile},
                                    {});
                                value_write_pointer += tile_bytes;
                            }
                            page += page_stride;
                        }
                    } else {
                        const uint32_t start_slot = kv_chunk_index * tiles_per_kv_chunk;
                        BrickCoordinate key_brick =
                            start_slot >= gather_brick_count
                                ? gather_origin_brick
                                : gather_slot_brick(gather_origin_brick, start_slot, gather_bricks);
                        uint32_t key_brick_index = layout::brick_index(key_brick, volume_bricks);
                        uint32_t page = layout::tile_offset(
                            batch_index, key_brick_index, head_index, brick_count, head_count, head_dim_tiles);
                        for (uint32_t slot = 0; slot < tiles_per_kv_chunk; ++slot) {
                            if (kv_chunk_index * tiles_per_kv_chunk + slot < gather_brick_count) {
                                for (uint32_t head_dim_tile = 0; head_dim_tile < head_dim_tiles; ++head_dim_tile) {
                                    noc.async_read(
                                        key_reader,
                                        CoreLocalMem<uint32_t>(
                                            key_base_pointer +
                                            (head_dim_tile * tiles_per_kv_chunk + slot) * tile_bytes),
                                        tile_bytes,
                                        {.page_id = page + head_dim_tile},
                                        {});
                                    noc.async_read(
                                        value_reader,
                                        CoreLocalMem<uint32_t>(value_write_pointer),
                                        tile_bytes,
                                        {.page_id = page + head_dim_tile},
                                        {});
                                    value_write_pointer += tile_bytes;
                                }
                            }
                            if (slot + 1 < tiles_per_kv_chunk) {
                                advance_gather_raster(
                                    key_brick, key_brick_index, gather_origin_brick, gather_bricks, volume_bricks);
                                page = layout::tile_offset(
                                    batch_index, key_brick_index, head_index, brick_count, head_count, head_dim_tiles);
                            }
                        }
                    }
                }
                const uint32_t mask_write_pointer = cb_mask.get_write_ptr();
                if (refill_mask) {
                    for (uint32_t slot = 0; slot < tiles_per_kv_chunk; ++slot) {
                        const uint32_t gather_slot = kv_chunk_index * tiles_per_kv_chunk + slot;
                        const uint32_t destination_address = mask_write_pointer + slot * tile_bytes;
                        if (gather_slot < gather_brick_count) {
                            noc.async_read(
                                interior_mask_reader,
                                CoreLocalMem<uint32_t>(destination_address),
                                tile_bytes,
                                {.page_id = gather_slot},
                                {});
                            continue;
                        }
                        volatile tt_l1_ptr uint32_t* destination =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_address);
                        for (uint32_t word = 0; word < tile_bytes / sizeof(uint32_t); ++word) {
                            destination[word] = 0xFF80FF80u;
                        }
                    }
                }
                noc.async_read_barrier();
                cb_key.push_back(tiles_per_kv_chunk * head_dim_tiles);
                cb_value.push_back(tiles_per_kv_chunk * head_dim_tiles);
                cb_mask.push_back(tiles_per_kv_chunk);
                continue;
            }
#endif
#if NA_PATH_KIND != 1
            if constexpr (compile_edge) {
                gather_edge_flash_chunk(
                    noc,
                    key_reader,
                    value_reader,
                    interior_mask_reader,
                    extents,
                    gather_origin_brick,
                    chunk_origin,
                    chunk_origin_site,
                    gather_bricks,
                    volume_bricks,
                    query_chunk_bricks,
                    query_origin_bricks,
                    cb_key,
                    cb_value,
                    cb_mask,
                    kv_chunk_index,
                    batch_index,
                    head_index,
                    brick_count,
                    head_count,
                    head_dim_tiles,
                    tiles_per_kv_chunk,
                    kv_chunk_count,
                    gather_brick_count,
                    bricks_per_query_chunk,
                    tile_bytes,
                    skip_kv,
                    per_brick_mask,
                    mask_memset_only,
                    relative_mask,
                    table_always,
                    use_uploaded_mask ? 1u : 0u,
                    regime);
            }
#endif
        }

        // An edge brick wrote generated tiles over the pages, so the next unclamped one must
        // put the table back.
        mask_pages_hold_table = compile_interior;

        cb_gather_origin.push_back(1);
        cb_gather_origin.pop_front(1);
    }
}
