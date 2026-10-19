// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_mapped_plan.hpp"

#include <algorithm>
#include <array>
#include <list>
#include <map>
#include <mutex>
#include <utility>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_shape.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

namespace {

constexpr uint64_t kTileDim = 32;
constexpr uint64_t kFaceDim = 16;
constexpr uint32_t kTileElems = kTileDim * kTileDim;
constexpr uint64_t kEntryBytes = kMappedEntryWords * sizeof(uint32_t);
constexpr uint64_t kMaxMapBytes = 16 * 1024 * 1024;
// Planning walks every output tile row on the host; above this many tiles it costs seconds.
constexpr uint64_t kMaxOutTiles = 128 * 1024;
// Slice ranges start on 16-byte boundaries so each core's partial page write keeps L1 and DRAM offsets congruent.
constexpr uint32_t kSliceGranuleBytes = 16;
// First-order device-time model (ns) deciding whether to slice: launch floor, per-row overhead, per source-tile read,
// per-row dispatch, and the smallest modeled gain worth the extra map rows. Measured on silicon.
constexpr uint64_t kLaunchNs = 1000;
constexpr uint64_t kRowNs = 600;
constexpr uint64_t kReadNs = 155;
constexpr uint64_t kRowLaunchNs = 34;
constexpr uint64_t kMinGainNs = 1500;
constexpr std::array<uint32_t, 2> kPipelinedDepths = {8, 4};
// Distinct reshapes memoized; a miss re-walks up to kMaxOutTiles output tiles.
constexpr size_t kLayoutCacheEntries = 1024;

uint64_t tile_pages(const Shape3& s) { return s.c * tt::div_up(s.h, kTileDim) * tt::div_up(s.w, kTileDim); }

uint32_t faced_offset(uint64_t row, uint64_t col) {
    const uint64_t face = (row / kFaceDim) * 2 + col / kFaceDim;
    return static_cast<uint32_t>(face * kFaceDim * kFaceDim + (row % kFaceDim) * kFaceDim + col % kFaceDim);
}

struct Segment {
    uint32_t in_page = 0;
    uint32_t in_offset = 0;
    uint32_t out_offset = 0;
    uint32_t num = 0;
};

bool extends(const Segment& head, const Segment& next) {
    return head.in_page == next.in_page && head.in_offset + head.num == next.in_offset &&
           head.out_offset + head.num == next.out_offset;
}

// Segments of one output tile, grouped by input tile in ascending order. A segment stays inside one face row on
// both sides, so it is contiguous in both faced tiles.
void map_output_page(const Shape3& in, const Shape3& out, uint64_t page, std::vector<Segment>& segments) {
    const uint64_t ht_i = tt::div_up(in.h, kTileDim);
    const uint64_t wt_i = tt::div_up(in.w, kTileDim);
    const uint64_t ht_o = tt::div_up(out.h, kTileDim);
    const uint64_t wt_o = tt::div_up(out.w, kTileDim);
    const uint64_t channel = page / (ht_o * wt_o);
    const uint64_t rem = page % (ht_o * wt_o);
    const uint64_t h0 = (rem / wt_o) * kTileDim;
    const uint64_t w0 = (rem % wt_o) * kTileDim;
    const uint64_t valid_h = std::min(kTileDim, out.h - h0);
    const uint64_t valid_w = std::min(kTileDim, out.w - w0);

    segments.clear();
    for (uint64_t r = 0; r < valid_h; ++r) {
        const uint64_t flat = (channel * out.h + h0 + r) * out.w + w0;
        const uint64_t row_in = flat / in.w;
        uint64_t w_in = flat % in.w;
        uint64_t h_in = row_in % in.h;
        uint64_t c_in = row_in / in.h;
        uint64_t col = 0;
        while (col < valid_w) {
            const uint64_t len =
                std::min({valid_w - col, kFaceDim - col % kFaceDim, in.w - w_in, kFaceDim - w_in % kFaceDim});
            const Segment segment{
                .in_page = static_cast<uint32_t>(c_in * ht_i * wt_i + (h_in / kTileDim) * wt_i + w_in / kTileDim),
                .in_offset = faced_offset(h_in % kTileDim, w_in % kTileDim),
                .out_offset = faced_offset(r, col),
                .num = static_cast<uint32_t>(len)};
            if (!segments.empty() && extends(segments.back(), segment)) {
                segments.back().num += segment.num;
            } else {
                segments.push_back(segment);
            }
            col += len;
            w_in += len;
            if (w_in == in.w) {
                w_in = 0;
                if (++h_in == in.h) {
                    h_in = 0;
                    ++c_in;
                }
            }
        }
    }

    std::stable_sort(
        segments.begin(), segments.end(), [](const Segment& a, const Segment& b) { return a.in_page < b.in_page; });
    size_t kept = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (kept > 0 && extends(segments[kept - 1], segments[i])) {
            segments[kept - 1].num += segments[i].num;
        } else {
            segments[kept++] = segments[i];
        }
    }
    segments.resize(kept);
}

// Source-tile reads of one map row: the reader loads one tile per change of input page.
uint32_t row_reads(const std::vector<Segment>& segments) {
    uint32_t reads = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i == 0 || segments[i].in_page != segments[i - 1].in_page) {
            ++reads;
        }
    }
    return reads;
}

uint64_t predicted_ns(uint64_t pages, uint64_t max_reads, uint64_t slices, uint64_t cores) {
    const uint64_t rows = pages * slices;
    const uint64_t used = std::min(rows, cores);
    const uint64_t rows_per_core = tt::div_up(rows, used);
    const uint64_t reads_per_core = rows_per_core * tt::div_up(max_reads, slices);
    return kLaunchNs + rows_per_core * kRowNs + reads_per_core * kReadNs + rows * kRowLaunchNs;
}

// Slices per output tile; 1 keeps one row per tile. Slicing only engages when output tiles fill less than half the
// cores, each row's read chain exceeds the launch floor, and the modeled gain clears the model's error band.
uint32_t choose_slices(uint64_t pages, uint32_t max_reads, uint32_t num_cores, uint32_t tile_bytes) {
    if (pages == 0 || num_cores <= 1 || 2 * pages > num_cores || max_reads * kReadNs <= kLaunchNs) {
        return 1;
    }
    const uint64_t cap = std::min<uint64_t>(tile_bytes / kSliceGranuleBytes, tt::div_up(num_cores, pages));
    if (cap < 2) {
        return 1;
    }
    uint64_t best = 1;
    uint64_t best_ns = predicted_ns(pages, max_reads, 1, num_cores);
    const uint64_t unsliced_ns = best_ns;
    for (uint64_t slices = 2; slices <= cap; ++slices) {
        const uint64_t ns = predicted_ns(pages, max_reads, slices, num_cores);
        if (ns < best_ns) {
            best = slices;
            best_ns = ns;
        }
    }
    return unsliced_ns - best_ns <= kMinGainNs ? 1 : static_cast<uint32_t>(best);
}

// Ascending in-tile datum cuts balancing segment counts across `slices` ranges. Each cut rounds up to the granule so
// the segment that crosses a balance point stays in the finished range; duplicate or out-of-tile cuts are dropped.
std::vector<uint32_t> range_cuts(const std::vector<Segment>& segments, uint32_t slices, uint32_t granule_elems) {
    std::vector<uint32_t> cuts;
    if (segments.empty() || slices < 2) {
        return cuts;
    }
    std::vector<Segment> ordered = segments;
    std::stable_sort(
        ordered.begin(), ordered.end(), [](const Segment& a, const Segment& b) { return a.out_offset < b.out_offset; });
    const uint64_t total = ordered.size();
    uint32_t previous = 0;
    uint64_t boundary = 1;
    uint64_t cumulative = 0;
    for (const Segment& segment : ordered) {
        ++cumulative;
        while (boundary < slices && cumulative * slices >= total * boundary) {
            const uint32_t cut = tt::round_up(segment.out_offset + segment.num, granule_elems);
            if (previous < cut && cut < kTileElems) {
                cuts.push_back(cut);
                previous = cut;
            }
            ++boundary;
        }
    }
    return cuts;
}

// Segments of one output tile split at `cuts` and bucketed by range; bucket i covers [bounds[i], bounds[i + 1]).
std::vector<std::vector<Segment>> slice_page(
    const std::vector<Segment>& segments, const std::vector<uint32_t>& bounds) {
    std::vector<std::vector<Segment>> buckets(bounds.size() - 1);
    for (const Segment& segment : segments) {
        Segment rest = segment;
        while (true) {
            const auto upper = std::upper_bound(bounds.begin(), bounds.end(), rest.out_offset);
            const size_t index = static_cast<size_t>(upper - bounds.begin()) - 1;
            const uint32_t end = bounds[index + 1];
            if (rest.out_offset + rest.num <= end) {
                buckets[index].push_back(rest);
                break;
            }
            const uint32_t head = end - rest.out_offset;
            buckets[index].push_back(Segment{rest.in_page, rest.in_offset, rest.out_offset, head});
            rest = Segment{rest.in_page, rest.in_offset + head, rest.out_offset + head, rest.num - head};
        }
    }
    return buckets;
}

std::vector<uint32_t> slice_bounds(const std::vector<Segment>& segments, uint32_t slices, uint32_t element_bytes) {
    std::vector<uint32_t> bounds{0};
    const auto cuts = range_cuts(segments, slices, kSliceGranuleBytes / element_bytes);
    bounds.insert(bounds.end(), cuts.begin(), cuts.end());
    bounds.push_back(kTileElems);
    return bounds;
}

uint32_t entries_alignment() { return std::max<uint32_t>(1, tt::tt_metal::hal::get_dram_alignment() / kEntryBytes); }

struct Candidates {
    std::optional<TileMappedLayout> unsliced;
    std::optional<TileMappedLayout> sliced;
    uint32_t slices = 1;
};

Candidates plan_candidates(
    const Shape3& in, const Shape3& out, uint32_t element_bytes, uint32_t tile_bytes, uint32_t num_cores) {
    Candidates candidates;
    const uint64_t pages = tile_pages(out);
    if (pages == 0 || pages > kMaxOutTiles) {
        return candidates;
    }
    const uint32_t alignment = entries_alignment();
    std::vector<Segment> segments;
    uint32_t max_entries = 1;
    uint32_t max_reads = 0;
    for (uint64_t page = 0; page < pages; ++page) {
        map_output_page(in, out, page, segments);
        max_entries = std::max<uint32_t>(max_entries, static_cast<uint32_t>(segments.size()));
        max_reads = std::max(max_reads, row_reads(segments));
        // Fail as soon as the widest row so far, repeated over every page, already exceeds the cap.
        if (pages * tt::round_up(max_entries, alignment) * kEntryBytes > kMaxMapBytes) {
            return candidates;
        }
    }
    candidates.unsliced = TileMappedLayout{
        .num_rows = static_cast<uint32_t>(pages),
        .entries_per_page = tt::round_up(max_entries, alignment),
        .num_output_pages = static_cast<uint32_t>(pages)};

    const uint32_t slices = choose_slices(pages, max_reads, num_cores, tile_bytes);
    if (slices < 2 || kSliceGranuleBytes % element_bytes != 0) {
        return candidates;
    }
    uint64_t num_rows = 0;
    uint32_t max_row_entries = 1;
    for (uint64_t page = 0; page < pages; ++page) {
        map_output_page(in, out, page, segments);
        const auto buckets = slice_page(segments, slice_bounds(segments, slices, element_bytes));
        num_rows += buckets.size();
        for (const auto& bucket : buckets) {
            max_row_entries = std::max<uint32_t>(max_row_entries, static_cast<uint32_t>(bucket.size()) + 1);
        }
    }
    const uint32_t entries = tt::round_up(max_row_entries, alignment);
    if (num_rows * entries * kEntryBytes > kMaxMapBytes) {
        return candidates;
    }
    candidates.sliced = TileMappedLayout{
        .num_rows = static_cast<uint32_t>(num_rows),
        .entries_per_page = entries,
        .num_output_pages = static_cast<uint32_t>(pages),
        .sliced = true};
    candidates.slices = slices;
    return candidates;
}

std::optional<std::pair<Shape3, Shape3>> mapped_shapes(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape) {
    const auto in = normalize(in_shape);
    const auto out = normalize(out_shape);
    if (!in || !out || in->c * in->h * in->w != out->c * out->h * out->w) {
        return std::nullopt;
    }
    return std::make_pair(*in, *out);
}

Candidates cached_candidates(
    const Shape3& in, const Shape3& out, uint32_t element_bytes, uint32_t tile_bytes, uint32_t num_cores) {
    using Key = std::array<uint64_t, 10>;
    using Entry = std::pair<Key, Candidates>;
    static std::mutex mutex;
    static std::list<Entry> recency;
    static std::map<Key, std::list<Entry>::iterator> cache;

    const Key key{in.c, in.h, in.w, out.c, out.h, out.w, element_bytes, tile_bytes, num_cores, entries_alignment()};
    {
        std::lock_guard lock(mutex);
        if (auto it = cache.find(key); it != cache.end()) {
            recency.splice(recency.begin(), recency, it->second);
            return it->second->second;
        }
    }
    auto candidates = plan_candidates(in, out, element_bytes, tile_bytes, num_cores);
    std::lock_guard lock(mutex);
    if (!cache.contains(key)) {
        recency.emplace_front(key, candidates);
        cache.emplace(key, recency.begin());
        if (recency.size() > kLayoutCacheEntries) {
            cache.erase(recency.back().first);
            recency.pop_back();
        }
    }
    return candidates;
}

// Upper bound on the segments of one output tile. When both widths are multiples of 16, each output face row is one
// run inside one input face row, so a tile has at most two segments per row; otherwise at most one per datum.
uint64_t max_segments_per_page(const Shape3& in, const Shape3& out) {
    return in.w % kFaceDim == 0 && out.w % kFaceDim == 0 ? 2 * kTileDim : kTileElems;
}

void write_entry(uint32_t* entry, const Segment& segment) {
    entry[0] = segment.in_page;
    entry[1] = segment.in_offset;
    entry[2] = segment.out_offset;
    entry[3] = segment.num;
}

}  // namespace

std::optional<TileMappedLayout> tile_mapped_layout(
    const ttnn::Shape& in_shape,
    const ttnn::Shape& out_shape,
    uint32_t element_bytes,
    uint32_t tile_bytes,
    uint32_t num_cores,
    uint64_t l1_budget) {
    const auto shapes = mapped_shapes(in_shape, out_shape);
    if (!shapes) {
        return std::nullopt;
    }
    const auto candidates = cached_candidates(shapes->first, shapes->second, element_bytes, tile_bytes, num_cores);
    // The sliced map page can be smaller than the per-tile one, so it may fit where the per-tile map does not.
    if (candidates.sliced && tile_mapped_input_depth(*candidates.sliced, tile_bytes, l1_budget) > 0) {
        return candidates.sliced;
    }
    if (candidates.unsliced && tile_mapped_input_depth(*candidates.unsliced, tile_bytes, l1_budget) > 0) {
        return candidates.unsliced;
    }
    return std::nullopt;
}

bool tile_mapped_within_caps(
    const ttnn::Shape& in_shape,
    const ttnn::Shape& out_shape,
    uint32_t element_bytes,
    uint32_t tile_bytes,
    uint32_t num_cores) {
    const auto shapes = mapped_shapes(in_shape, out_shape);
    if (!shapes) {
        return false;
    }
    const auto& [in, out] = *shapes;
    const uint64_t pages = tile_pages(out);
    if (pages == 0 || pages > kMaxOutTiles) {
        return false;
    }
    if (pages * tt::round_up(max_segments_per_page(in, out), entries_alignment()) * kEntryBytes <= kMaxMapBytes) {
        return true;
    }
    return cached_candidates(in, out, element_bytes, tile_bytes, num_cores).unsliced.has_value();
}

TileMappedPlan build_tile_mapped_plan(
    const ttnn::Shape& in_shape,
    const ttnn::Shape& out_shape,
    uint32_t element_bytes,
    uint32_t tile_bytes,
    uint32_t num_cores,
    bool sliced) {
    const auto shapes = mapped_shapes(in_shape, out_shape);
    TT_FATAL(shapes.has_value(), "Reshape shapes differ in volume");
    const auto& [in, out] = *shapes;
    const auto candidates = cached_candidates(in, out, element_bytes, tile_bytes, num_cores);
    const auto& layout = sliced ? candidates.sliced : candidates.unsliced;
    TT_FATAL(layout.has_value(), "Reshape tile map exceeds its size caps");

    TileMappedPlan plan{.layout = *layout};
    plan.table.assign(static_cast<size_t>(layout->num_rows) * layout->page_words(), 0);
    std::vector<Segment> segments;
    size_t row = 0;
    for (uint64_t page = 0; page < layout->num_output_pages; ++page) {
        map_output_page(in, out, page, segments);
        if (!sliced) {
            uint32_t* entry = plan.table.data() + row++ * layout->page_words();
            for (const Segment& segment : segments) {
                write_entry(entry, segment);
                entry += kMappedEntryWords;
            }
            continue;
        }
        const auto bounds = slice_bounds(segments, candidates.slices, element_bytes);
        const auto buckets = slice_page(segments, bounds);
        for (size_t i = 0; i < buckets.size(); ++i) {
            uint32_t* entry = plan.table.data() + row++ * layout->page_words();
            write_entry(
                entry,
                Segment{
                    static_cast<uint32_t>(page),
                    bounds[i] * element_bytes,
                    (bounds[i + 1] - bounds[i]) * element_bytes,
                    0});
            for (const Segment& segment : buckets[i]) {
                entry += kMappedEntryWords;
                write_entry(entry, segment);
            }
        }
    }
    TT_FATAL(row == layout->num_rows, "Reshape tile map has {} rows, its layout {}", row, layout->num_rows);
    return plan;
}

uint32_t tile_mapped_input_depth(const TileMappedLayout& layout, uint32_t tile_bytes, uint64_t l1_budget) {
    // Map, input and scratch CBs, each starting at a DRAM-aligned address. At depth 1 the unsliced map needs no more
    // L1 than native's tiled reshape, which holds one map page, one input tile and one output tile, each also
    // DRAM-aligned: native's map page holds every segment it splits the output tile into, and this map's segments are
    // those same segments, merged further once grouped by input tile.
    const auto footprint = [](uint64_t bytes) { return tt::round_up(bytes, tt::tt_metal::hal::get_dram_alignment()); };
    const uint64_t fixed = footprint(layout.page_bytes()) + footprint(tile_bytes);
    for (const uint32_t depth : kPipelinedDepths) {
        if (fixed + footprint(static_cast<uint64_t>(depth) * tile_bytes) <= l1_budget) {
            return depth;
        }
    }
    return fixed + footprint(tile_bytes) <= l1_budget ? 1 : 0;
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
