// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_rows_plan.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <numeric>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_plan.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// Named rather than anonymous: TileRowsSchedule, declared in the header, holds these types.
namespace tile_rows_planner {

// Source caps tried, largest first, when the full plan does not fit.
constexpr std::array<uint32_t, 6> kMaxInSteps = {kTileRowsMaxIn, 16, 8, 4, 2, 1};
// Smaller output batches until there are enough batches to spread over cores; small batches re-read shared
// sources, so stop at 8.
constexpr std::array<uint32_t, 3> kMaxOutSteps = {32, 16, 8};
constexpr uint32_t kMinBatches = 32;
constexpr uint64_t kMaxTableBytes = 32 * 1024 * 1024;
// Every output tile has at least one run in each 16-row half.
constexpr uint64_t kMinRunsPerTile = 2;
constexpr size_t kChoiceMemoEntries = 64;
constexpr uint32_t kNoSource = std::numeric_limits<uint32_t>::max();

// Source (input tile, row) of every row of one output tile; padding rows have tile kNoSource and keep their own row.
struct TileRowSources {
    std::array<uint32_t, kTileDim> tile{};
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
        s.tile[r] = static_cast<uint32_t>(channel_in * ht_i * wt_i + (h_in / kTileDim) * wt_i + col_in / kTileDim);
        s.row[r] = static_cast<uint32_t>(h_in % kTileDim);
    }
    return s;
}

// What the planner needs of every output tile, from one row_sources walk per tile. Both lists are flattened: tile
// t owns [begin[t], begin[t + 1]).
struct TileFacts {
    // Distinct real source tiles, ascending.
    std::vector<uint32_t> source_begin{0};
    std::vector<uint32_t> sources;
    // A run is a maximal block of rows copied as one face-row span: same source tile, consecutive source rows, and
    // neither side crossing the face boundary at row 16. Its source tile (kNoSource for padding) and its run word
    // without the slot, which depends on the batch.
    std::vector<uint32_t> run_begin{0};
    std::vector<uint32_t> run_source;
    std::vector<uint32_t> run_word;
};

struct Batch {
    std::vector<uint32_t> outs;
    std::vector<uint32_t> ins;  // ascending
};

struct PlanInputs {
    Shape3 in;
    Shape3 out;
    uint64_t out_pages = 0;
    uint32_t max_runs = 0;
    uint64_t runs = 0;
    uint64_t native_reads = 0;
    TileFacts tiles;
    // Output tiles ordered by their smallest source tile, ties by index.
    std::vector<uint32_t> order;
};

void add_tile(PlanInputs& p, const TileRowSources& s) {
    TileFacts& f = p.tiles;
    const size_t first_source = f.sources.size();
    for (const uint32_t tile : s.tile) {
        if (tile != kNoSource) {
            f.sources.push_back(tile);
        }
    }
    std::sort(f.sources.begin() + first_source, f.sources.end());
    f.sources.erase(std::unique(f.sources.begin() + first_source, f.sources.end()), f.sources.end());
    f.source_begin.push_back(static_cast<uint32_t>(f.sources.size()));
    p.native_reads += f.sources.size() - first_source;

    const size_t first_run = f.run_word.size();
    for (uint32_t r = 0; r < kTileDim; ++r) {
        const bool start = r == 0 || s.tile[r] != s.tile[r - 1] || s.row[r] != s.row[r - 1] + 1 || r % kFaceDim == 0 ||
                           s.row[r] % kFaceDim == 0;
        if (start) {
            f.run_source.push_back(s.tile[r]);
            f.run_word.push_back((s.row[r] << kRunSrcRowShift) | (r << kRunDstRowShift));
            p.runs += s.tile[r] != kNoSource ? 1 : 0;
        }
        ++f.run_word.back();  // one more row in the current run
    }
    f.run_begin.push_back(static_cast<uint32_t>(f.run_word.size()));
    p.max_runs = std::max<uint32_t>(p.max_runs, static_cast<uint32_t>(f.run_word.size() - first_run));
}

PlanInputs prepare(const Shape3& in, const Shape3& out) {
    PlanInputs p{.in = in, .out = out, .out_pages = tile_pages(out)};
    for (uint64_t t = 0; t < p.out_pages; ++t) {
        add_tile(p, row_sources(in, out, t));
    }
    // Heuristic: tiles that share sources land next to each other, but can still split across batches (re-reads).
    const auto& begin = p.tiles.source_begin;
    const auto& sources = p.tiles.sources;
    p.order.resize(p.out_pages);
    std::iota(p.order.begin(), p.order.end(), 0);
    std::stable_sort(
        p.order.begin(), p.order.end(), [&](uint32_t a, uint32_t b) { return sources[begin[a]] < sources[begin[b]]; });
    return p;
}

// Greedy batches of output tiles in source order; each holds at most max_out tiles and at most max_in distinct
// sources, unless one output tile alone needs more.
std::vector<Batch> make_batches(const PlanInputs& p, uint32_t max_out, uint32_t max_in) {
    const auto& begin = p.tiles.source_begin;
    const auto& all_sources = p.tiles.sources;
    std::vector<Batch> batches;
    Batch current;
    for (const uint32_t out_page : p.order) {
        const auto first = all_sources.begin() + begin[out_page];
        const auto last = all_sources.begin() + begin[out_page + 1];
        std::vector<uint32_t> merged;
        std::set_union(current.ins.begin(), current.ins.end(), first, last, std::back_inserter(merged));
        if (!current.outs.empty() && (current.outs.size() == max_out || merged.size() > max_in)) {
            batches.push_back(std::move(current));
            current = Batch{};
            merged.assign(first, last);
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

// Caps checked from the shapes alone, so an oversized reshape never reaches the planner.
bool exceeds_caps(const Shape3& out) {
    const uint64_t out_pages = tile_pages(out);
    const uint64_t lower_bound_bytes = out_pages * (kTileRowsEntryHeaderWords + kMinRunsPerTile) * sizeof(uint32_t);
    return out_pages == 0 || out_pages > kMaxPlannedOutTiles || lower_bound_bytes > kMaxTableBytes;
}

std::optional<std::pair<Shape3, Shape3>> permutation_shapes(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape) {
    auto shapes = normalize_pair(in_shape, out_shape);
    if (shapes && (shapes->first.w % kTileDim != 0 || shapes->second.w % kTileDim != 0)) {
        return std::nullopt;
    }
    return shapes;
}

}  // namespace tile_rows_planner

struct TileRowsSchedule {
    // Shared by the plans one shape pair gets under different source caps.
    std::shared_ptr<const tile_rows_planner::PlanInputs> inputs;
    std::vector<tile_rows_planner::Batch> batches;
};

namespace tile_rows_planner {

std::optional<TileRowsChoice> choose(std::shared_ptr<const PlanInputs> inputs, uint32_t max_in) {
    auto schedule = std::make_shared<TileRowsSchedule>();
    schedule->inputs = std::move(inputs);
    const PlanInputs& p = *schedule->inputs;
    TileRowsChoice choice;
    for (const uint32_t max_out : kMaxOutSteps) {
        schedule->batches = make_batches(p, max_out, max_in);
        uint64_t batch_reads = 0;
        for (const auto& b : schedule->batches) {
            batch_reads += b.ins.size();
        }
        choice = TileRowsChoice{
            .layout = layout_of(p, schedule->batches),
            .max_out = max_out,
            .max_in = max_in,
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
    choice.schedule = std::move(schedule);
    return choice;
}

// Process-wide: routing asks on every call, before any device or program cache is involved, and the answer
// depends only on the shapes, the source cap and the DRAM alignment. Entries drop their schedule to stay small.
// `prepared` carries the per-shape walk across calls for the same shapes; it is filled only on a miss.
std::optional<TileRowsChoice> memoized_choice(
    const Shape3& in, const Shape3& out, uint32_t max_in, std::shared_ptr<const PlanInputs>& prepared) {
    using Key = std::array<uint64_t, 8>;
    static PlanMemo<Key, std::optional<TileRowsChoice>> memo(kChoiceMemoEntries);

    const Key key{in.c, in.h, in.w, out.c, out.h, out.w, max_in, page_align_words()};
    if (auto hit = memo.find(key)) {
        return std::move(*hit);
    }
    if (!prepared) {
        prepared = std::make_shared<const PlanInputs>(prepare(in, out));
    }
    auto choice = choose(prepared, max_in);
    auto entry = choice;
    if (entry) {
        entry->schedule.reset();
    }
    memo.insert(key, std::move(entry));
    return choice;
}

}  // namespace tile_rows_planner

std::optional<TileRowsChoice> plan_tile_rows(
    const ttnn::Shape& in_shape, const ttnn::Shape& out_shape, uint32_t max_in) {
    using namespace tile_rows_planner;
    const auto shapes = permutation_shapes(in_shape, out_shape);
    if (!shapes || exceeds_caps(shapes->second)) {
        return std::nullopt;
    }
    std::shared_ptr<const PlanInputs> prepared;
    return memoized_choice(shapes->first, shapes->second, max_in, prepared);
}

std::optional<TileRowsChoice> plan_tile_rows_fitting(
    const ttnn::Shape& in_shape, const ttnn::Shape& out_shape, const std::function<bool(const TileRowsChoice&)>& fits) {
    using namespace tile_rows_planner;
    const auto shapes = permutation_shapes(in_shape, out_shape);
    if (!shapes || exceeds_caps(shapes->second)) {
        return std::nullopt;
    }
    std::shared_ptr<const PlanInputs> prepared;
    for (const uint32_t staged_batches : {kTileRowsStagedBatches, 1u}) {
        for (const uint32_t max_in : kMaxInSteps) {
            auto choice = memoized_choice(shapes->first, shapes->second, max_in, prepared);
            if (!choice) {
                continue;
            }
            choice->staged_batches = staged_batches;
            if (fits(*choice)) {
                return choice;
            }
        }
    }
    return std::nullopt;
}

TileRowsPlan build_tile_rows_plan(
    const ttnn::Shape& in_shape, const ttnn::Shape& out_shape, const TileRowsChoice& choice) {
    using namespace tile_rows_planner;
    std::shared_ptr<const TileRowsSchedule> schedule = choice.schedule;
    if (!schedule) {
        const auto shapes = permutation_shapes(in_shape, out_shape);
        TT_FATAL(shapes.has_value(), "Reshape is not a tile-row permutation");
        auto planned = std::make_shared<TileRowsSchedule>();
        planned->inputs = std::make_shared<const PlanInputs>(prepare(shapes->first, shapes->second));
        planned->batches = make_batches(*planned->inputs, choice.max_out, choice.max_in);
        schedule = std::move(planned);
    }
    const PlanInputs& p = *schedule->inputs;
    const auto& batches = schedule->batches;
    const TileFacts& tiles = p.tiles;

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
            const uint32_t out_page = batch.outs[j];
            const uint32_t first_run = tiles.run_begin[out_page];
            const uint32_t n_runs = tiles.run_begin[out_page + 1] - first_run;
            uint32_t* entry = row + entries_base + j * layout.entry_words();
            entry[0] = out_page;
            entry[1] = n_runs;
            for (uint32_t r = 0; r < n_runs; ++r) {
                const uint32_t source = tiles.run_source[first_run + r];
                uint32_t slot = kZeroSlot;
                if (source != kNoSource) {
                    slot = static_cast<uint32_t>(
                        std::lower_bound(batch.ins.begin(), batch.ins.end(), source) - batch.ins.begin());
                }
                entry[kTileRowsEntryHeaderWords + r] = (slot << kRunSlotShift) | tiles.run_word[first_run + r];
            }
        }
    }
    return plan;
}

uint64_t tile_rows_l1_bytes(const TileRowsChoice& choice, uint32_t tile_bytes) {
    // Plan CB double-buffers one page, the input CB holds staged_batches batches of slots, plus the write ring and a
    // zero tile.
    const TileRowsLayout& layout = choice.layout;
    return 2ull * layout.page_bytes() +
           (static_cast<uint64_t>(choice.staged_batches) * layout.in_slots + kTileRowsWriteGroup + 1) *
               static_cast<uint64_t>(tile_bytes);
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
