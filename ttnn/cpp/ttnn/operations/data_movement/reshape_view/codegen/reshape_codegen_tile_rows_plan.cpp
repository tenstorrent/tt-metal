// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_rows_plan.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::data_movement::reshape_codegen {

namespace {

constexpr uint64_t kTileDim = 32;
constexpr uint64_t kFaceRows = 16;
// One output tile can need 32 source tiles, so 32 is the smallest source cap.
constexpr uint32_t kMaxIn = 32;
// Smaller output batches until there are enough batches to spread over cores; small batches re-read shared
// sources, so stop at 8.
constexpr std::array<uint32_t, 3> kMaxOutSteps = {32, 16, 8};
constexpr uint32_t kMinBatches = 32;
// Planning walks every output tile row on the host; above this many tiles it costs seconds and hundreds of MB.
constexpr uint64_t kMaxOutTiles = 128 * 1024;
constexpr uint64_t kMaxTableBytes = 32 * 1024 * 1024;
// Every output tile has at least one run in each 16-row half.
constexpr uint64_t kMinRunsPerTile = 2;
// Native reshapes one output tile per core per round and reads every source tile it touches. Tile rows puts a batch
// of at least 8 output tiles on one core, so it needs at least 8x fewer reads to win (measured on Blackhole).
constexpr uint64_t kTileRowsMinReadCut = 8;
constexpr size_t kLayoutCacheEntries = 64;
constexpr int64_t kNoSource = -1;

// Physical (C, H, W) model of a TILE shape: leading dims fold into C.
struct Shape3 {
    uint64_t c = 1;
    uint64_t h = 1;
    uint64_t w = 1;
};

std::optional<Shape3> normalize(const ttnn::Shape& shape) {
    if (shape.rank() == 0) {
        return std::nullopt;
    }
    for (size_t i = 0; i < shape.rank(); ++i) {
        if (shape[i] == 0) {
            return std::nullopt;
        }
    }
    Shape3 out;
    out.w = shape[-1];
    if (shape.rank() >= 2) {
        out.h = shape[-2];
    }
    for (size_t i = 0; i + 2 < shape.rank(); ++i) {
        out.c *= shape[i];
    }
    return out;
}

uint64_t tile_pages(const Shape3& s) { return s.c * tt::div_up(s.h, kTileDim) * (s.w / kTileDim); }

// Source (input tile, row) of every row of one output tile; padding rows have tile kNoSource and keep their own row.
struct TileRowSources {
    std::array<int64_t, kTileDim> tile{};
    std::array<uint32_t, kTileDim> row{};
};

TileRowSources row_sources(const Shape3& in, const Shape3& out, uint64_t out_tile) {
    const uint64_t ht_i = tt::div_up(in.h, kTileDim);
    const uint64_t wt_i = in.w / kTileDim;
    const uint64_t ht_o = tt::div_up(out.h, kTileDim);
    const uint64_t wt_o = out.w / kTileDim;
    const uint64_t channel = out_tile / (ht_o * wt_o);
    const uint64_t rem = out_tile % (ht_o * wt_o);
    const uint64_t ht = rem / wt_o;
    const uint64_t wt = rem % wt_o;

    TileRowSources s;
    for (uint64_t r = 0; r < kTileDim; ++r) {
        const uint64_t h = ht * kTileDim + r;
        if (h >= out.h) {
            s.tile[r] = kNoSource;
            s.row[r] = static_cast<uint32_t>(r);
            continue;
        }
        const uint64_t flat = (channel * out.h + h) * out.w + wt * kTileDim;
        const uint64_t row_in = flat / in.w;
        const uint64_t col_in = flat % in.w;
        const uint64_t channel_in = row_in / in.h;
        const uint64_t h_in = row_in % in.h;
        s.tile[r] = static_cast<int64_t>(channel_in * ht_i * wt_i + (h_in / kTileDim) * wt_i + col_in / kTileDim);
        s.row[r] = static_cast<uint32_t>(h_in % kTileDim);
    }
    return s;
}

// A run is a maximal block of rows copied as one face-row span: same source tile, consecutive source rows, and
// neither side crossing the face boundary at row 16. Returns the first row of each run.
std::vector<uint32_t> run_starts(const TileRowSources& s) {
    std::vector<uint32_t> starts;
    for (uint32_t r = 0; r < kTileDim; ++r) {
        const bool start = r == 0 || s.tile[r] != s.tile[r - 1] || s.row[r] != s.row[r - 1] + 1 || r % kFaceRows == 0 ||
                           s.row[r] % kFaceRows == 0;
        if (start) {
            starts.push_back(r);
        }
    }
    return starts;
}

// Distinct real source tiles of one output tile, ascending.
std::vector<int64_t> distinct_sources(const TileRowSources& s) {
    std::vector<int64_t> sources;
    for (const int64_t tile : s.tile) {
        if (tile != kNoSource) {
            sources.push_back(tile);
        }
    }
    std::sort(sources.begin(), sources.end());
    sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
    return sources;
}

struct Batch {
    std::vector<uint64_t> outs;
    std::vector<int64_t> ins;  // ascending
};

struct PlanInputs {
    Shape3 in;
    Shape3 out;
    uint64_t out_pages = 0;
    uint32_t max_runs = 0;
    // Runs over real rows, and source tiles native reads: each output tile reads its distinct sources once.
    uint64_t runs = 0;
    uint64_t native_reads = 0;
    // Output tiles ordered by their smallest source tile, ties by index.
    std::vector<uint64_t> order;
};

PlanInputs prepare(const Shape3& in, const Shape3& out) {
    PlanInputs p{.in = in, .out = out, .out_pages = tile_pages(out)};
    std::vector<int64_t> min_source(p.out_pages);
    for (uint64_t t = 0; t < p.out_pages; ++t) {
        const auto s = row_sources(in, out, t);
        const auto starts = run_starts(s);
        p.max_runs = std::max<uint32_t>(p.max_runs, static_cast<uint32_t>(starts.size()));
        p.runs += std::count_if(starts.begin(), starts.end(), [&](uint32_t r) { return s.tile[r] != kNoSource; });
        p.native_reads += distinct_sources(s).size();
        int64_t lowest = std::numeric_limits<int64_t>::max();
        for (const int64_t tile : s.tile) {
            if (tile != kNoSource) {
                lowest = std::min(lowest, tile);
            }
        }
        min_source[t] = lowest;
    }
    // Ordering by smallest source puts output tiles that share sources next to each other. It is a heuristic:
    // sharing tiles can still split across batches, which costs re-reads but not correctness.
    p.order.resize(p.out_pages);
    std::iota(p.order.begin(), p.order.end(), 0);
    std::stable_sort(
        p.order.begin(), p.order.end(), [&](uint64_t a, uint64_t b) { return min_source[a] < min_source[b]; });
    return p;
}

// Greedy batches of output tiles in source order; each holds at most max_out tiles and max_in distinct sources.
std::vector<Batch> make_batches(const PlanInputs& p, uint32_t max_out) {
    std::vector<Batch> batches;
    Batch current;
    for (const uint64_t out_page : p.order) {
        const auto sources = distinct_sources(row_sources(p.in, p.out, out_page));
        std::vector<int64_t> merged;
        std::set_union(
            current.ins.begin(), current.ins.end(), sources.begin(), sources.end(), std::back_inserter(merged));
        if (!current.outs.empty() && (current.outs.size() == max_out || merged.size() > kMaxIn)) {
            batches.push_back(std::move(current));
            current = Batch{};
            merged = sources;
        }
        current.outs.push_back(out_page);
        current.ins = std::move(merged);
    }
    if (!current.outs.empty()) {
        batches.push_back(std::move(current));
    }
    return batches;
}

uint32_t page_align_words() { return std::max<uint32_t>(16, tt::tt_metal::hal::get_dram_alignment() / 4); }

TileRowsLayout layout_of(const PlanInputs& p, const std::vector<Batch>& batches) {
    TileRowsLayout layout;
    for (const auto& b : batches) {
        layout.in_slots = std::max<uint32_t>(layout.in_slots, static_cast<uint32_t>(b.ins.size()));
        layout.out_slots = std::max<uint32_t>(layout.out_slots, static_cast<uint32_t>(b.outs.size()));
    }
    layout.max_runs = p.max_runs;
    layout.num_batches = static_cast<uint32_t>(batches.size());
    const uint32_t words = kTileRowsHeaderWords + layout.in_slots + layout.out_slots * layout.entry_words();
    layout.page_words = tt::round_up(words, page_align_words());
    layout.num_input_pages = static_cast<uint32_t>(tile_pages(p.in));
    layout.num_output_pages = static_cast<uint32_t>(p.out_pages);
    return layout;
}

struct Choice {
    TileRowsLayout layout;
    uint32_t max_out = 0;
    uint64_t runs = 0;
    uint64_t native_reads = 0;
    uint64_t batch_reads = 0;
};

std::optional<Choice> choose(const Shape3& in, const Shape3& out) {
    const uint64_t out_pages = tile_pages(out);
    const uint64_t lower_bound_bytes = out_pages * (kTileRowsEntryHeaderWords + kMinRunsPerTile) * sizeof(uint32_t);
    if (out_pages == 0 || out_pages > kMaxOutTiles || lower_bound_bytes > kMaxTableBytes) {
        return std::nullopt;
    }
    const PlanInputs p = prepare(in, out);
    Choice choice;
    for (const uint32_t max_out : kMaxOutSteps) {
        const auto batches = make_batches(p, max_out);
        uint64_t batch_reads = 0;
        for (const auto& b : batches) {
            batch_reads += b.ins.size();
        }
        choice = Choice{
            .layout = layout_of(p, batches),
            .max_out = max_out,
            .runs = p.runs,
            .native_reads = p.native_reads,
            .batch_reads = batch_reads};
        if (choice.layout.num_batches >= kMinBatches) {
            break;
        }
    }
    // Per-batch source slots, which dominate when the source cap forces small batches, only show once planned.
    const uint64_t table_bytes = static_cast<uint64_t>(choice.layout.num_batches) * choice.layout.page_bytes();
    if (table_bytes > kMaxTableBytes) {
        return std::nullopt;
    }
    return choice;
}

std::optional<std::pair<Shape3, Shape3>> permutation_shapes(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape) {
    const auto in = normalize(in_shape);
    const auto out = normalize(out_shape);
    if (!in || !out || in->w % kTileDim != 0 || out->w % kTileDim != 0 ||
        in->c * in->h * in->w != out->c * out->h * out->w) {
        return std::nullopt;
    }
    return std::make_pair(*in, *out);
}

std::optional<Choice> cached_choice(const Shape3& in, const Shape3& out) {
    using Key = std::array<uint64_t, 7>;
    static std::mutex mutex;
    static std::map<Key, std::optional<Choice>> cache;
    static std::deque<Key> insertion_order;

    const Key key{in.c, in.h, in.w, out.c, out.h, out.w, page_align_words()};
    {
        std::lock_guard lock(mutex);
        if (auto it = cache.find(key); it != cache.end()) {
            return it->second;
        }
    }
    auto choice = choose(in, out);
    std::lock_guard lock(mutex);
    if (cache.emplace(key, choice).second) {
        insertion_order.push_back(key);
        if (insertion_order.size() > kLayoutCacheEntries) {
            cache.erase(insertion_order.front());
            insertion_order.pop_front();
        }
    }
    return choice;
}

}  // namespace

bool is_tile_row_permutation(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape) {
    return permutation_shapes(in_shape, out_shape).has_value();
}

std::optional<TileRowsLayout> tile_rows_layout(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape) {
    const auto shapes = permutation_shapes(in_shape, out_shape);
    if (!shapes) {
        return std::nullopt;
    }
    const auto choice = cached_choice(shapes->first, shapes->second);
    if (!choice) {
        return std::nullopt;
    }
    return choice->layout;
}

TileRowsPlan build_tile_rows_plan(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape) {
    const auto shapes = permutation_shapes(in_shape, out_shape);
    TT_FATAL(shapes.has_value(), "Reshape is not a tile-row permutation");
    const auto choice = cached_choice(shapes->first, shapes->second);
    TT_FATAL(choice.has_value(), "Reshape tile-row plan exceeds its size caps");

    const PlanInputs p = prepare(shapes->first, shapes->second);
    const auto batches = make_batches(p, choice->max_out);
    TileRowsPlan plan{.layout = layout_of(p, batches)};
    const TileRowsLayout& layout = plan.layout;
    plan.table.assign(static_cast<size_t>(layout.num_batches) * layout.page_words, 0);

    const uint32_t entries_base = kTileRowsHeaderWords + layout.in_slots;
    for (size_t index = 0; index < batches.size(); ++index) {
        const Batch& batch = batches[index];
        uint32_t* row = plan.table.data() + index * layout.page_words;
        row[0] = static_cast<uint32_t>(batch.ins.size());
        row[1] = static_cast<uint32_t>(batch.outs.size());
        std::copy(batch.ins.begin(), batch.ins.end(), row + kTileRowsHeaderWords);
        for (size_t j = 0; j < batch.outs.size(); ++j) {
            const uint64_t out_page = batch.outs[j];
            const auto s = row_sources(p.in, p.out, out_page);
            const auto starts = run_starts(s);
            uint32_t* entry = row + entries_base + j * layout.entry_words();
            entry[0] = static_cast<uint32_t>(out_page);
            entry[1] = static_cast<uint32_t>(starts.size());
            for (size_t r = 0; r < starts.size(); ++r) {
                const uint32_t first = starts[r];
                const uint32_t rows = (r + 1 < starts.size() ? starts[r + 1] : kTileDim) - first;
                uint32_t slot = kZeroSlot;
                if (s.tile[first] != kNoSource) {
                    slot = static_cast<uint32_t>(
                        std::lower_bound(batch.ins.begin(), batch.ins.end(), s.tile[first]) - batch.ins.begin());
                }
                entry[kTileRowsEntryHeaderWords + r] =
                    (slot << kRunSlotShift) | (s.row[first] << kRunSrcRowShift) | (first << kRunDstRowShift) | rows;
            }
        }
    }
    return plan;
}

uint64_t tile_rows_l1_bytes(const TileRowsLayout& layout, uint32_t tile_bytes) {
    // Plan CB double-buffers one page, the input CB two batches of slots, plus the write ring and a zero tile.
    return 2ull * layout.page_bytes() +
           (2ull * layout.in_slots + kTileRowsWriteGroup + 1) * static_cast<uint64_t>(tile_bytes);
}

bool tile_rows_beats_native(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape, uint32_t num_cores) {
    const auto shapes = permutation_shapes(in_shape, out_shape);
    const auto choice = shapes ? cached_choice(shapes->first, shapes->second) : std::nullopt;
    if (!choice) {
        return false;
    }
    // A width change copies one row per run and loses below the read cut; a refold of whole face halves wins once
    // native needs more than one round.
    return choice->native_reads >= kTileRowsMinReadCut * choice->batch_reads ||
           (choice->runs <= kMinRunsPerTile * choice->layout.num_output_pages &&
            choice->layout.num_output_pages > num_cores);
}

bool tile_rows_fits(const Tensor& input, const ttnn::Shape& logical_output_shape, uint64_t l1_budget) {
    const uint32_t tile_bytes =
        input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
    const auto rows_layout = tile_rows_layout(input.logical_shape(), logical_output_shape);
    return rows_layout.has_value() && tile_rows_l1_bytes(*rows_layout, tile_bytes) <= l1_budget;
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
