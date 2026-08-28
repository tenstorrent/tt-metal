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

using ttnn::transformer::neighborhood::ALL_AXES;
using ttnn::transformer::neighborhood::Axis;
using ttnn::transformer::neighborhood::BrickPoint;
using ttnn::transformer::neighborhood::ChunkShapeInBricks;
using ttnn::transformer::neighborhood::containing_brick;
using ttnn::transformer::neighborhood::first_site_of;
using ttnn::transformer::neighborhood::ShapeInBricks;
using ttnn::transformer::neighborhood::ShapeInChunks;
using ttnn::transformer::neighborhood::Site;
using ttnn::transformer::neighborhood::SiteOffset;
using ttnn::transformer::neighborhood::Unit;

namespace {

// tiles_per_kv_chunk is bounded by DST capacity (8), so the per-chunk scratch is fixed size.
constexpr uint32_t MAX_TILES_PER_KV_CHUNK = 8;

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
    const Site& query_origin_site, const Site& key_origin_site, const kernel_args::NeighborhoodExtents& extents) {
    // Shapes into 3-word locals -- see classify_brick in neighborhood_mask_gen.hpp: reading them
    // through `extents` per axis measured 9% slower at the stage-5 band.
    const auto brick_sites = extents.brick_sites;
    const auto context_window = extents.context_window;
    const BrickPoint key_brick = containing_brick(key_origin_site, extents.brick_sites);
    const BrickPoint query_brick = containing_brick(query_origin_site, extents.brick_sites);

    uint32_t index = 0;
    for (Axis axis : ALL_AXES) {
        const int32_t relative = static_cast<int32_t>(key_brick[axis]) - static_cast<int32_t>(query_brick[axis]);
        const int32_t low = relative_span_low(context_window[axis], brick_sites[axis]);
        const int32_t high = relative_span_high(context_window[axis], brick_sites[axis]);
        if (relative < low || relative > high) {
            return NO_REGIME;
        }
        index = index * static_cast<uint32_t>(high - low + 1) + static_cast<uint32_t>(relative - low);
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
    const BrickPoint& gather_origin_brick,
    const Site& query_origin_site,
    ShapeInBricks gather_bricks,
    const kernel_args::NeighborhoodExtents& extents) {
    // Shapes into 3-word locals -- see classify_brick in neighborhood_mask_gen.hpp: reading them
    // through `extents` per axis measured 9% slower at the stage-5 band.
    const auto brick_sites = extents.brick_sites;
    const auto context_window = extents.context_window;
    const BrickPoint query_brick = containing_brick(query_origin_site, extents.brick_sites);

    for (Axis axis : ALL_AXES) {
        const int32_t low = relative_span_low(context_window[axis], brick_sites[axis]);
        const int32_t high = relative_span_high(context_window[axis], brick_sites[axis]);
        if (static_cast<int32_t>(gather_origin_brick[axis]) - static_cast<int32_t>(query_brick[axis]) != low) {
            return false;
        }
        if (gather_bricks[axis] != static_cast<uint32_t>(high - low + 1)) {
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
    const Site& query_origin_site, const kernel_args::NeighborhoodExtents& extents) {
    // Shapes into 3-word locals -- see classify_brick in neighborhood_mask_gen.hpp: reading them
    // through `extents` per axis measured 9% slower at the stage-5 band.
    const auto brick_sites = extents.brick_sites;
    const auto volume = extents.volume;
    const auto context_window = extents.context_window;
    const auto resident = extents.resident;
    const auto shard_origin = extents.shard_origin;
    for (Axis axis : ALL_AXES) {
        const uint32_t window = context_window[axis] < volume[axis] ? context_window[axis] : volume[axis];
        if (window >= volume[axis]) {
            return false;
        }
        // A brick hanging off what this shard holds carries ghost rows, which the table's
        // always-visible interior pattern does not describe.
        if (query_origin_site[axis] + brick_sites[axis] > resident[axis]) {
            return false;
        }
        const int32_t first = static_cast<int32_t>(query_origin_site[axis]) + shard_origin[axis];
        const int32_t last = first + static_cast<int32_t>(brick_sites[axis]) - 1;
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
FORCE_INLINE uint32_t chunk_regime(const Site& chunk_origin_site, const kernel_args::NeighborhoodExtents& extents) {
    // Shapes into 3-word locals -- see classify_brick in neighborhood_mask_gen.hpp: reading them
    // through `extents` per axis measured 9% slower at the stage-5 band.
    const auto brick_sites = extents.brick_sites;
    const auto stride = extents.stride;
    const auto volume = extents.volume;
    const auto context_window = extents.context_window;
    const auto resident = extents.resident;
    const auto shard_origin = extents.shard_origin;
    const auto query_chunk = extents.query_chunk;
    uint32_t regime = 0;
    for (Axis axis : ALL_AXES) {
        const uint32_t window = context_window[axis] < volume[axis] ? context_window[axis] : volume[axis];
        if (window >= volume[axis]) {
            return NO_REGIME;
        }
        // A chunk that overhangs what is resident carries ghosts, which the shared patterns do
        // not describe.
        if (chunk_origin_site[axis] + query_chunk[axis] > resident[axis]) {
            return NO_REGIME;
        }
        const uint32_t snap = ttnn::transformer::neighborhood::snap_extent_on_axis(stride[axis], brick_sites[axis]);
        const uint32_t highest = volume[axis] - window;

        bool all_low = true;
        bool all_high = true;
        bool all_centred = true;
        for (uint32_t offset = 0; offset < query_chunk[axis]; ++offset) {
            // A brick in a low-edge halo sits below the volume; it has no window and is never
            // read, so clamping here just keeps the arithmetic in range.
            const int32_t signed_global = static_cast<int32_t>(chunk_origin_site[axis] + offset) + shard_origin[axis];
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

}  // namespace

void kernel_main() {
    constexpr uint32_t head_count = get_compile_time_arg_val(kernel_args::reader_arg::head_count);
    constexpr uint32_t brick_count = get_compile_time_arg_val(kernel_args::reader_arg::brick_count);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(kernel_args::reader_arg::head_dim_tiles);
    constexpr uint32_t bricks_per_query_chunk =
        get_compile_time_arg_val(kernel_args::reader_arg::bricks_per_query_chunk);
    // A ratio, not a size: bricks per chunk, which is what scales a chunk coordinate to bricks.
    constexpr ChunkShapeInBricks query_chunk_bricks = ChunkShapeInBricks::of(
        get_compile_time_arg_val(kernel_args::reader_arg::query_chunk_bricks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::query_chunk_bricks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::query_chunk_bricks_width));
    constexpr ShapeInChunks volume_chunks = ShapeInChunks::of(
        get_compile_time_arg_val(kernel_args::reader_arg::volume_chunks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::volume_chunks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::volume_chunks_width));
    constexpr uint32_t chunk_count = volume_chunks.time() * volume_chunks.height() * volume_chunks.width();
    constexpr uint32_t tiles_per_kv_chunk = get_compile_time_arg_val(kernel_args::reader_arg::tiles_per_kv_chunk);
    constexpr uint32_t kv_chunk_count = get_compile_time_arg_val(kernel_args::reader_arg::kv_chunk_count);
    constexpr uint32_t gather_brick_count = get_compile_time_arg_val(kernel_args::reader_arg::gather_brick_count);

    constexpr ShapeInBricks volume_bricks = ShapeInBricks::of(
        get_compile_time_arg_val(kernel_args::reader_arg::volume_bricks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::volume_bricks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::volume_bricks_width));
    // Q lives on the query grid; K, V and the gather live on the resident grid above. Equal
    // unless the host asked for a query sub-region.
    constexpr ShapeInBricks query_bricks = ShapeInBricks::of(
        get_compile_time_arg_val(kernel_args::reader_arg::query_bricks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::query_bricks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::query_bricks_width));
    constexpr uint32_t query_brick_count = get_compile_time_arg_val(kernel_args::reader_arg::query_brick_count);
    // A position, not a size: where the query grid starts inside the resident brick grid.
    constexpr BrickPoint query_origin_bricks = BrickPoint::at(
        get_compile_time_arg_val(kernel_args::reader_arg::query_origin_bricks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::query_origin_bricks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::query_origin_bricks_width));
    constexpr ShapeInBricks gather_bricks = ShapeInBricks::of(
        get_compile_time_arg_val(kernel_args::reader_arg::gather_bricks_time),
        get_compile_time_arg_val(kernel_args::reader_arg::gather_bricks_height),
        get_compile_time_arg_val(kernel_args::reader_arg::gather_bricks_width));
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
    const bool interior_table_supported = relative_mask != 0 && has_interior_mask != 0 && per_brick_mask == 0;
    bool mask_pages_hold_table = false;

    uint32_t argument_index = 0;
    const uint32_t query_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t key_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t value_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t gather_origin_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t interior_mask_address = get_arg_val<uint32_t>(argument_index++);
    const uint32_t work_item_start = get_arg_val<uint32_t>(argument_index++);
    const uint32_t work_item_count = get_arg_val<uint32_t>(argument_index++);
    const uint32_t tile_bytes = get_arg_val<uint32_t>(argument_index++);

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

        const BrickPoint chunk_origin = layout::chunk_origin_brick(chunk_index, volume_chunks, query_chunk_bricks);

        // ---- Q: one tile row per brick in the chunk ----
        cb_query.reserve_back(head_dim_tiles * bricks_per_query_chunk);
        uint32_t query_write_pointer = cb_query.get_write_ptr();
        for (uint32_t brick_in_chunk = 0; brick_in_chunk < bricks_per_query_chunk; ++brick_in_chunk) {
            const BrickPoint brick = layout::brick_within_chunk(brick_in_chunk, chunk_origin, query_chunk_bricks);
            // A chunk on the far edge can hang off the volume. Those rows have no queries to
            // read; the writer drops them again, and flash rows are independent so whatever
            // stale L1 they carry cannot reach a real query.
            if (!layout::point3_is_inside(brick, query_bricks)) {
                query_write_pointer += head_dim_tiles * tile_bytes;
                continue;
            }
            const uint32_t first_tile = layout::tile_offset(
                batch_index,
                layout::point3_to_linear(brick, query_bricks),
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
        cb_query.push_back(head_dim_tiles * bricks_per_query_chunk);

        const volatile tt_l1_ptr uint32_t* origin_row =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(origin_write_pointer);
        namespace column = kernel_args::gather_origin_column;
        // Origins are rounded down to a brick boundary by the planner, so this division is exact.
        const BrickPoint gather_origin_brick = containing_brick(
            Site::at(
                origin_row[column::gather_time], origin_row[column::gather_height], origin_row[column::gather_width]),
            extents.brick_sites);

        // Where this device sits in the global volume. Addressing above is LOCAL; window
        // placement below is GLOBAL, and this is what converts between them.
        extents.shard_origin = SiteOffset::at(
            static_cast<int32_t>(origin_row[column::shard_origin_time]),
            static_cast<int32_t>(origin_row[column::shard_origin_height]),
            static_cast<int32_t>(origin_row[column::shard_origin_width]));

        // RESIDENT-local, like query_origin_site below and like the gather table's key origins.
        const Site chunk_origin_site = first_site_of(chunk_origin + query_origin_bricks, extents.brick_sites);

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
        const bool use_interior_table =
            interior_table_supported &&
            gather_is_canonical(gather_origin_brick, chunk_origin_site, gather_bricks, extents) &&
            (table_always != 0 || brick_window_is_unclamped(chunk_origin_site, extents));
        // The pages already hold exactly these tiles, so there is nothing to write.
        const bool refill_mask = use_interior_table && !mask_pages_hold_table;

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
            constexpr uint32_t mask_tiles_per_kv_chunk =
                per_brick_mask != 0 ? bricks_per_query_chunk * tiles_per_kv_chunk : tiles_per_kv_chunk;
            cb_mask.reserve_back(mask_tiles_per_kv_chunk);

            // K and V are laid out DIFFERENTLY in their circular buffers, and it matters.
            //
            // matmul_blocks walks in1 as `in1_index += N` over the inner dimension, so in1 is
            // always a [K, N] grid of tiles -- the `transpose` flag transposes each TILE, not the
            // grid. For QK^T that means K must be stored head-dim-major, [head_dim_tiles][slots];
            // for the PV matmul the inner dimension is the slot, so V is [slots][head_dim_tiles],
            // which is the order the gather naturally produces.
            //
            // At head_dim_tiles == 1 the two layouts are the same buffer, which is why a wrong K
            // layout survived every test until one used a 64-wide head.
            const uint32_t key_base_pointer = cb_key.get_write_ptr();
            uint32_t value_write_pointer = cb_value.get_write_ptr();
            uint32_t mask_write_pointer = cb_mask.get_write_ptr();

            // Three passes, deliberately. Mixing the constant fills with fill_mask_tile in one
            // loop body puts a large function (nested loops, divisions) next to a memset in the
            // same instruction cache and measured WORSE than either alone -- 7498 ms against
            // 2761 ms for generating everywhere. Keeping each loop tight avoids that.
            mask_gen::BrickCoverage coverage[MAX_TILES_PER_KV_CHUNK];
            Site key_origins[MAX_TILES_PER_KV_CHUNK];

            for (uint32_t slot = 0; slot < tiles_per_kv_chunk; ++slot) {
                const uint32_t gather_slot = kv_chunk_index * tiles_per_kv_chunk + slot;
                const bool slot_is_padding = gather_slot >= gather_brick_count;
                // Gather slots are linearised over the gather box the same way everything else is
                // linearised, so this is the shared decode rather than a fourth transcription of
                // row-major. A ragged slot past the gather has no brick; it reads slot 0 and is
                // masked out below.
                const BrickPoint slot_offset =
                    slot_is_padding ? BrickPoint{} : layout::linear_to_point3<Unit::Bricks>(gather_slot, gather_bricks);
                const BrickPoint key_brick = gather_origin_brick + slot_offset;

                key_origins[slot] = first_site_of(key_brick, extents.brick_sites);
                // The resident table already holds the right tile for every slot, uniform ones
                // included, so classifying is 175 divisions per chunk spent on an answer nothing
                // reads. `coverage` is dead on that path.
                coverage[slot] = (slot_is_padding || use_interior_table)
                                     ? mask_gen::BrickCoverage::NoneVisible
                                     : mask_gen::classify_brick(chunk_origin_site, key_origins[slot], extents);

                const uint32_t key_first_tile = layout::tile_offset(
                    batch_index,
                    layout::point3_to_linear(key_brick, volume_bricks),
                    head_index,
                    brick_count,
                    head_count,
                    head_dim_tiles);
                // DIFFVAE_NA_SKIP_KV: issue no K/V reads at all, leaving whatever the buffers
                // held. WRONG OUTPUT -- it exists to split the gather's DMA cost from the compute
                // kernel's, which no other probe here separates, and which decides whether a
                // bigger query chunk (fewer slots per query, more matmul per slot) can pay.
                if (skip_kv != 0) {
                    value_write_pointer += head_dim_tiles * tile_bytes;
                    continue;
                }
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
            }

            // Per-brick masks: every query brick in the chunk gets its own tile per slot, laid
            // out [brick][slot] so the compute kernel can advance by tiles_per_kv_chunk per in0
            // subblock. Generated rather than copied from the uploaded set: the upload is keyed on
            // the CHUNK's window regime, which is the wrong window for every brick but the first.
            if (per_brick_mask != 0) {
                for (uint32_t brick_in_chunk = 0; brick_in_chunk < bricks_per_query_chunk; ++brick_in_chunk) {
                    const BrickPoint query_brick =
                        layout::brick_within_chunk(brick_in_chunk, chunk_origin, query_chunk_bricks);
                    // Into RESIDENT-local sites: the key origins this is compared against come
                    // from the gather table, which addresses the resident tensor. Without the
                    // shift a query sub-region would place every window a halo too low.
                    const Site query_origin_site =
                        first_site_of(query_brick + query_origin_bricks, extents.brick_sites);
                    const uint32_t brick_base = mask_write_pointer + brick_in_chunk * tiles_per_kv_chunk * tile_bytes;
                    // Resolved per brick, not per slot: the table describes a window that centres
                    // on its query, which stops being true once the window clamps at a volume edge.
                    const bool brick_takes_table =
                        relative_mask != 0 && use_uploaded_mask &&
                        (table_always != 0 || brick_window_is_unclamped(query_origin_site, extents));
                    for (uint32_t slot = 0; slot < tiles_per_kv_chunk; ++slot) {
                        const uint32_t gather_slot = kv_chunk_index * tiles_per_kv_chunk + slot;
                        // DIFFVAE_NA_MASK_MEMSET_ONLY: write every tile as a constant, skipping
                        // classify_brick AND fill_mask_tile. WRONG OUTPUT -- it exists to split
                        // "writing N tiles costs X" from "deciding what is in them costs X", which
                        // no other experiment here separates.
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
                            // The uploaded RELATIVE tile, fetched by DMA, replaces ~1024 elements
                            // of window arithmetic per tile. Everything about the pattern is in
                            // (key_brick - query_brick), so no gather origin or shard origin
                            // enters and one table serves every chunk on every shard.
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
                        const uint32_t fill =
                            brick_coverage == mask_gen::BrickCoverage::AllVisible ? 0x00000000u : 0xFF80FF80u;
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
                cb_mask.push_back(mask_tiles_per_kv_chunk);
                continue;
            }

            // An unclamped brick reads the relative table straight into cb_mask -- but only when
            // the pages do not already hold it, which after the first such brick in a run they
            // do. Refills are rare enough that the per-slot page arithmetic and the fallback
            // below cost nothing, so neither is hoisted or cached.
            if (use_interior_table) {
                if (refill_mask) {
                    for (uint32_t slot = 0; slot < tiles_per_kv_chunk; ++slot) {
                        const uint32_t gather_slot = kv_chunk_index * tiles_per_kv_chunk + slot;
                        const uint32_t destination_address = mask_write_pointer + slot * tile_bytes;
                        if (gather_slot < gather_brick_count) {
                            // Canonical gather, so the table page IS the slot.
                            noc.async_read(
                                interior_mask_reader,
                                CoreLocalMem<uint32_t>(destination_address),
                                tile_bytes,
                                {.page_id = gather_slot},
                                {});
                            continue;
                        }
                        // A ragged slot past the gather has no keys; mask it out rather than
                        // leaving whatever the pages held.
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

            // Uniform bricks: one word repeated, no window arithmetic at all.
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

            // Bricks the window boundary cuts through, on a chunk the resident table does not
            // describe: either a GNA regime (uploaded, keyed on the chunk's window) or a stride-1
            // brick whose window clamps at a volume edge, which generates.
            //
            // The relative table is NOT consulted here. It only ever applies to an unclamped
            // brick, and those took the resident path above -- asking again would put a DRAM read
            // and fill_mask_tile in one loop body, and that instruction-cache mix is exactly what
            // made the gated run (32.3 s) cost as much as generating everywhere (34.1 s) when only
            // 20% of its tiles ever generated.
            if (use_uploaded_mask && relative_mask == 0) {
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

        // An edge brick wrote generated tiles over the pages, so the next unclamped one must
        // put the table back.
        mask_pages_hold_table = use_interior_table;

        cb_gather_origin.push_back(1);
        cb_gather_origin.pop_front(1);
    }
}
