// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared host/device contract. All extents are tiles. Source IDs are tensor ranks,
// never transport ranks; the caller translates the resolved route before packing.
// Live source width controls packing, while allocated source width controls addresses.
// A partial K chunk retains the configured full CB stride in every consumer.

// One contiguous sequence interval in a packed K chunk. Column starts are implicit:
// zero for the first run, then the preceding run's exclusive column_end.
struct PackedKVMaskRun {
    uint32_t global_start_tile;
    uint32_t column_end;
};

// Preconditions checked by the host: nonzero source/chunk/region sizes, source_tiles a
// whole number of regions, group size <= 32, products representable in uint32_t, source
// IDs in the resolved tensor-rank range. Chunk and stream indices passed to addressing
// helpers must be in range.
//
// A packed attention pass streams the heads of its group's sources (every slab but the
// newest, source by source). Causal and logical-tail masks reach only the newest slab, so
// newest slabs are regrouped by row instead: a row is source_count consecutive tensor ranks,
// whose newest regions form one contiguous global range, chunked like the classic ring's
// diagonal chunk for that Q shard. Each pass appends the rows its group completes (see
// packed_kv_pass_rows) in row order, which keeps the passes balanced and makes each row's
// masked columns a suffix the compute can drop. A single-slab cache has no head to defer and
// streams whole sources. All sizes are in tiles; the final chunk may be partial, but retains
// the full CB stride.
struct PackedKVGroupPlan {
    uint32_t source_tiles;
    uint32_t source_count;
    uint32_t chunk_tiles;
    uint32_t region_tiles;
    // Rows whose newest slabs follow the heads in this pass, one bit per row.
    uint32_t newest_rows = 0;

    struct Location {
        uint32_t rank;
        uint32_t local;
    };

    constexpr bool defers_newest() const { return source_tiles > region_tiles; }
    constexpr uint32_t head_tiles() const { return defers_newest() ? source_tiles - region_tiles : source_tiles; }
    constexpr uint32_t head_stream_tiles() const { return head_tiles() * source_count; }
    constexpr uint32_t row_tiles() const { return source_count * region_tiles; }
    constexpr uint32_t tile_count() const {
        return head_stream_tiles() + static_cast<uint32_t>(__builtin_popcount(newest_rows)) * row_tiles();
    }
    // The k-th appended row.
    constexpr uint32_t newest_row(uint32_t k) const {
        uint32_t rows = newest_rows;
        for (; k > 0; --k) {
            rows &= rows - 1;
        }
        return static_cast<uint32_t>(__builtin_ctz(rows));
    }
    constexpr uint32_t chunk_count() const { return tile_count() / chunk_tiles + (tile_count() % chunk_tiles != 0); }
    constexpr uint32_t valid_tiles(uint32_t chunk) const {
        const uint32_t start = chunk * chunk_tiles;
        if (start >= tile_count()) {
            return 0;
        }
        const uint32_t remaining = tile_count() - start;
        return remaining < chunk_tiles ? remaining : chunk_tiles;
    }
    // Source tensor rank and its local tile; head tiles resolve through the group's IDs.
    constexpr Location locate(uint32_t stream_tile, const uint32_t* source_ids) const {
        if (stream_tile < head_stream_tiles()) {
            return {source_ids[stream_tile / head_tiles()], stream_tile % head_tiles()};
        }
        const uint32_t newest = stream_tile - head_stream_tiles();
        const uint32_t within_row = newest % row_tiles();
        return {
            newest_row(newest / row_tiles()) * source_count + within_row / region_tiles,
            head_tiles() + within_row % region_tiles};
    }
    // Largest group member the chunk reads. Members become ready in index order, and the
    // newest slabs follow every member's head.
    constexpr uint32_t last_source(uint32_t chunk) const {
        const uint32_t last = chunk * chunk_tiles + valid_tiles(chunk) - 1;
        return last < head_stream_tiles() ? last / head_tiles() : source_count - 1;
    }
    // Tiles that stay contiguous in both the stream and one source.
    constexpr uint32_t segment_tiles(uint32_t chunk, uint32_t destination_offset) const {
        const uint32_t stream_tile = chunk * chunk_tiles + destination_offset;
        const uint32_t remaining = valid_tiles(chunk) - destination_offset;
        const uint32_t contiguous = stream_tile < head_stream_tiles()
                                        ? head_tiles() - stream_tile % head_tiles()
                                        : region_tiles - (stream_tile - head_stream_tiles()) % region_tiles;
        return remaining < contiguous ? remaining : contiguous;
    }
    // Largest local slab index among the chunk's valid tiles.
    constexpr uint32_t max_slab(uint32_t chunk) const {
        const uint32_t first = chunk * chunk_tiles;
        const uint32_t last = first + valid_tiles(chunk) - 1;
        if (last >= head_stream_tiles()) {
            return (source_tiles - 1) / region_tiles;
        }
        // A head chunk that crosses into a later source contains the earlier head's last slab.
        if (first / head_tiles() != last / head_tiles()) {
            return (head_tiles() - 1) / region_tiles;
        }
        return last % head_tiles() / region_tiles;
    }
    // Emit at most chunk_tiles runs of globally contiguous columns. Head runs split at
    // source and block-cyclic slab boundaries; each appended row is one run.
    // The caller supplies chunk_tiles entries, including for partial chunks. Only chunks
    // reaching a masked slab call this, so it locates each run directly.
    constexpr uint32_t mask_runs(
        uint32_t chunk, const uint32_t* source_ids, uint32_t global_chunk_tiles, PackedKVMaskRun* runs) const {
        const uint32_t valid = valid_tiles(chunk);
        uint32_t count = 0;
        for (uint32_t column = 0; column < valid;) {
            const uint32_t stream_tile = chunk * chunk_tiles + column;
            uint32_t length = valid - column;
            uint32_t global_start = 0;
            if (stream_tile < head_stream_tiles()) {
                const Location at = locate(stream_tile, source_ids);
                const uint32_t slab = at.local / region_tiles;
                const uint32_t region_offset = at.local - slab * region_tiles;
                length = length < region_tiles - region_offset ? length : region_tiles - region_offset;
                global_start = slab * global_chunk_tiles + at.rank * region_tiles + region_offset;
            } else {
                const uint32_t newest = stream_tile - head_stream_tiles();
                const uint32_t within_row = newest % row_tiles();
                length = length < row_tiles() - within_row ? length : row_tiles() - within_row;
                global_start = head_tiles() / region_tiles * global_chunk_tiles +
                               newest_row(newest / row_tiles()) * row_tiles() + within_row;
            }
            runs[count++] = {global_start, column + length};
            column += length;
        }
        return count;
    }
};

// Rows appended by each pass: a row joins the pass in which its last rank arrives, so its
// newest slabs are ready behind that pass's heads. arrival(i) returns the tensor rank the
// route delivers i-th; pass p covers arrivals [p * group, (p + 1) * group).
template <typename Arrival>
constexpr void packed_kv_pass_rows(uint32_t ring_size, uint32_t group, Arrival&& arrival, uint32_t* pass_rows) {
    const uint32_t passes = ring_size / group;
    uint32_t row_pass[32] = {};
    for (uint32_t i = 0; i < ring_size; ++i) {
        const uint32_t row = arrival(i) / group;
        const uint32_t pass = i / group;
        row_pass[row] = pass > row_pass[row] ? pass : row_pass[row];
    }
    for (uint32_t pass = 0; pass < passes; ++pass) {
        pass_rows[pass] = 0;
    }
    for (uint32_t row = 0; row < passes; ++row) {
        pass_rows[row_pass[row]] |= 1u << row;
    }
}

// The plan for one pass, appending the newest slabs of `rows`.
constexpr PackedKVGroupPlan packed_kv_pass_plan(PackedKVGroupPlan plan, uint32_t rows) {
    plan.newest_rows = plan.defers_newest() ? rows : 0;
    return plan;
}

// Readiness advances once per source in sequencer order, and is reused across Q
// chunks in the same group. The callback must complete its wait before returning.
// Drain also covers cores with no Q work, preserving the receiver's signal cadence.
struct PackedKVSourceReadiness {
    uint32_t ready_sources = 0;

    template <typename WaitSource>
    void drain(uint32_t required_sources, WaitSource&& wait_source) {
        while (ready_sources < required_sources) {
            wait_source(ready_sources);
            ++ready_sources;
        }
    }

    template <typename WaitSource>
    void wait_for_chunk(const PackedKVGroupPlan& plan, uint32_t chunk, WaitSource&& wait_source) {
        drain(plan.last_source(chunk) + 1, wait_source);
    }
};

// Each source owns one region per global prefill chunk. Clip an oversized
// input cache to the slabs touched by this invocation; keep the physical source
// stride separate so remote reads still address the persistent gather correctly.
constexpr uint32_t packed_kv_source_tiles(
    uint32_t capacity_tiles, uint32_t logical_tiles, uint32_t region_tiles, uint32_t ring_size) {
    if (region_tiles == 0 || ring_size == 0) {
        return capacity_tiles;
    }
    const uint32_t global_chunk_tiles = region_tiles * ring_size;
    const uint32_t slabs = logical_tiles / global_chunk_tiles + (logical_tiles % global_chunk_tiles != 0);
    const uint32_t valid_tiles = slabs * region_tiles;
    return valid_tiles < capacity_tiles ? valid_tiles : capacity_tiles;
}

// All sources must participate in the group; configurations with inactive sources
// retain their per-source traversal. This is a scheduling predicate, not a proof
// of mask elision: uniform live slabs can contain globally invalid tail tiles.
constexpr uint32_t packed_kv_source_group_size(
    uint32_t configured, uint32_t ring_size, uint32_t source_tiles, uint32_t logical_tiles, uint32_t active_mask) {
    if (ring_size == 0 || ring_size > 32 || source_tiles == 0 || configured <= 1 || ring_size % configured != 0) {
        return 1;
    }
    const uint32_t all_sources = ~uint32_t{0} >> (32 - ring_size);
    return logical_tiles > 0 && logical_tiles <= source_tiles * ring_size && active_mask == all_sources ? configured
                                                                                                        : 1;
}
