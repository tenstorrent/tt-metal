// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "head_plan.hpp"
#include "ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>

using moe_block_tiles::kBf16TileBytes;
using moe_block_tiles::kFp32TileBytes;

namespace ttnn::operations::experimental::moe_block::moe_head {

// The cost model's arithmetic with every operation rounded on its own, no fused multiply-add: the ranking of close
// layouts does not depend on what the compiler contracts.
#if defined(__clang__)
#define HEAD_PLAN_NO_FP_CONTRACT _Pragma("clang fp contract(off)")
#else
#define HEAD_PLAN_NO_FP_CONTRACT
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

namespace head_plan_detail {

constexpr uint32_t kPortGroup = 4;     // the x gather: a port sender's packet group
constexpr uint32_t kHeaders = 12;      // fabric packet headers a RISC
constexpr uint32_t kRouterArgs = 341;  // a router core's runtime-arg bound
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kEngineLogitsOverhead =
    3.0;  // the logits on the matmul cores: their matmul's cost over the router cores'

using head_math::align_to_64;
using head_math::ceil_div;
std::string kilobytes(double bytes) { return format_fixed(bytes / 1024, 0); }
std::string to_text(int64_t value) { return std::to_string(value); }

int64_t round_us(double value) { return static_cast<int64_t>(std::nearbyint(value)); }

enum class Cells : uint8_t { None, Column, Row, Rows };

// router_cells: where the router cores go.
// router_cores: the router cores (a row tile each of the chip's owned rows of one router iteration).
Cells router_cells(const HeadShape& shape, uint32_t engine_rows, uint32_t engine_columns, uint32_t router_cores) {
    const uint32_t grid_rows = shape.grid[1];
    if (router_cores + 2 * shape.links + 1 <= grid_rows) {
        return Cells::Column;
    }
    if (engine_rows < grid_rows && router_cores <= engine_columns) {
        return Cells::Row;
    }
    if (engine_rows < grid_rows && router_cores <= engine_columns * (grid_rows - engine_rows)) {
        return Cells::Rows;
    }
    return Cells::None;
}

uint32_t latent_destinations(uint32_t tp_size, bool ring) { return ring && tp_size > 2 ? tp_size / 2 : tp_size - 1; }

int64_t region_of(const Regions& regions, const std::string& name) {
    for (const auto& [region_name, bytes] : regions) {
        if (region_name == name) {
            return bytes;
        }
    }
    return 0;
}

// cores_check: every per-core requirement past the matmul cores' arena; "" or why not. The ports' buffers
// (logits sub-blocks, routing exchange packets, x-gather staging) and the watcher's grow the arena up to arena_cap, as
// the program grows it (head_program: the arena is one size on every core).
std::string cores_check(
    const HeadShape& shape,
    const HeadLayout& layout,
    const ReduceScatterPlan& reduce_scatter,
    int64_t& arena,
    int64_t arena_cap) {
    const uint32_t tp_size = shape.tp_size, links = shape.links, grid_rows = shape.grid[1];
    const bool ring = shape.ring && tp_size > 2;
    if (reduce_scatter.core_block_tiles >= 256 || (tp_size > 4 && reduce_scatter.scatter != -2)) {
        return "reduce-scatter counts word: blocks of < 256 tiles (P " + to_text(reduce_scatter.core_block_tiles) +
               "), past TP 4 the rows scatter only";
    }
    if (kPortGroup + 1 > kHeaders) {
        return "packet headers";
    }
    if (shape.latent) {
        const uint32_t num_destinations = latent_destinations(tp_size, ring);
        if (num_destinations > 3 || kPortGroup + 1 + num_destinations > kHeaders) {
            return "latent all-to-all: " + to_text(num_destinations) +
                   " destination blocks a port > 3 (packet headers: " + to_text(kHeaders) + " a RISC)";
        }
    }
    const uint32_t num_router_cores =
        shape.rows / 32 / tp_size / layout.router_iterations;  // (the router cores: one router iteration's row tiles)
    if (layout.engine_columns * layout.engine_rows / links > 128) {
        return "a port's reduce-scatter cores: " + to_text(layout.engine_columns * layout.engine_rows / links) +
               " > 128 (the port reader's rdc / seen tables)";
    }
    if (3 * links > grid_rows) {
        return "mover column: " + to_text(2 * links) + " ports + " + to_text(links) + " copy cores > " +
               to_text(grid_rows) + " cells";
    }
    if (num_router_cores + 2 * links + 1 > grid_rows) {
        const int64_t free_cells = (int64_t(grid_rows) - layout.engine_rows) * layout.engine_columns;
        if (layout.engine_rows >= grid_rows ||
            !(num_router_cores <= layout.engine_columns || num_router_cores <= free_cells)) {
            return "router cores: " + to_text(num_router_cores) +
                   " (a row tile each of a chip's owned rows of a router iteration) fit neither the mover column (" +
                   to_text(grid_rows) + " cells less " + to_text(2 * links) + " ports and the watcher) nor the " +
                   to_text(free_cells) + " free cells under the matmul cores (" + to_text(layout.engine_columns) + "x" +
                   to_text(layout.engine_rows) + ")";
        }
    }
    const uint32_t num_planner_only_cores = planner_only_cores(
        shape.row_blocks,
        head_free_cells(
            shape.grid[0],
            grid_rows,
            layout.engine_columns,
            layout.engine_rows,
            num_router_cores + 2 * links + 1 > grid_rows ? num_router_cores : 0)
            .size());
    const uint32_t expert_tiles = shape.router_candidates / 32;
    const auto [router_offsets, router_bytes] =
        router_layout(tp_size, expert_tiles, shape.top_k, expert_tiles, shape.row_blocks > 1);
    if (router_bytes > arena) {
        return "router cores: " + kilobytes(router_bytes) + " KB > the arena's " + kilobytes(arena) + " KB";
    }
    const int64_t router_cores_per_link = ceil_div(num_router_cores, links);
    const int64_t logits_partials_bytes = int64_t(tp_size - 1) * router_cores_per_link * expert_tiles * kFp32TileBytes;
    if (logits_partials_bytes > arena_cap) {
        return "the ports' logits sub-blocks: " + kilobytes(logits_partials_bytes) + " KB > the arena's " +
               kilobytes(arena_cap) + " KB";
    }
    // (the planner and the exchange: the whole chunk's tokens - every row block's)
    const uint32_t num_chips = shape.num_chips(), tokens_per_chip = shape.tokens_per_chip_all_blocks(),
                   top_k = shape.top_k, experts_per_chip = shape.experts_per_chip(), accumulators = shape.accumulators;
    if (num_chips * experts_per_chip > 1024 || tokens_per_chip > 2048 || experts_per_chip > 64 ||
        shape.tokens_per_chip() % 32) {
        return "routing planner: N EPC " + to_text(num_chips * experts_per_chip) + " <= 1024, S " +
               to_text(tokens_per_chip) + " <= 2048 tokens a chip, EPC " + to_text(experts_per_chip) + " <= 64";
    }
    // (the last row block's planner - or the only one's - over the router's dead buffers [0, bias) where it fits)
    const auto placement = place_planner(
        num_chips,
        tokens_per_chip,
        top_k,
        experts_per_chip,
        accumulators,
        num_router_cores,
        region_of(router_offsets, "bias"),
        router_bytes,
        arena,
        0,
        num_planner_only_cores);
    const uint32_t planner_instances = placement.planner_instances;
    if (!placement.ok) {
        const int64_t regions_per_core = ceil_div(std::max<uint32_t>(planner_instances, 1), num_router_cores);
        const std::string what = !placement.sizes.reason.empty()
                                     ? placement.sizes.reason
                                     : to_text(regions_per_core) + " region(s) of " + kilobytes(placement.step) +
                                           " KB past " + kilobytes(placement.router_region_start) + " KB";
        return "routing planner: " + what + " > the arena's " + kilobytes(arena) + " KB";
    }
    if (shape.row_blocks >
        1) {  // (an earlier block's: the planner_instances instances planning no token, past the router's bytes)
        const auto idle_placement = place_planner(
            num_chips,
            kPlannerIdleTokens,
            top_k,
            experts_per_chip,
            accumulators,
            num_router_cores,
            region_of(router_offsets, "arr"),
            router_bytes,
            arena,
            planner_instances,
            num_planner_only_cores);
        if (!idle_placement.ok) {
            return "routing planner: an earlier row block's " + to_text(planner_instances) +
                   " instances past the router's " + kilobytes(router_bytes) + " KB > the arena's " + kilobytes(arena) +
                   " KB";
        }
    }
    const auto paths = routing_paths(shape.sp_rows, tp_size, ring, links, shape.first_hops);
    if (std::holds_alternative<std::string>(paths)) {
        return "routing exchange: " + std::get<std::string>(paths);
    }
    const auto [max_x_packets, max_y_packets, max_senders] = std::get<RoutingPaths>(paths);
    const int64_t slot_bytes = routing_slot_bytes(tokens_per_chip, top_k, experts_per_chip, planner_instances);
    const int64_t x_gather_bytes =
        port_cb_bytes(layout.k_step_tiles, shape.payload, kPortGroup);  // (the x-gather's staging, after them)
    const int64_t port_bytes =
        align_to_64(align_to_64(logits_partials_bytes) + int64_t(max_x_packets) * slot_bytes) + x_gather_bytes;
    if (port_bytes > arena_cap) {
        return "routing exchange: a port's buffers " + kilobytes(port_bytes) + " KB (logits " +
               kilobytes(logits_partials_bytes) + " + " + to_text(max_x_packets) + " packets of " +
               format_fixed(slot_bytes / 1024.0, 1) + " KB + x-gather staging " + kilobytes(x_gather_bytes) +
               " KB) > the arena's " + kilobytes(arena_cap) + " KB";
    }
    if (2 * int64_t(max_y_packets) * slot_bytes > arena_cap) {
        return "routing exchange: the watcher's Y buffers " + kilobytes(2 * int64_t(max_y_packets) * slot_bytes) +
               " KB > the arena's " + kilobytes(arena_cap) + " KB";
    }
    arena = std::max({arena, align_to_64(port_bytes), align_to_64(2 * int64_t(max_y_packets) * slot_bytes)});
    const int64_t planner_args_length =
        26 + 5 * int64_t(planner_instances) + 3 * int64_t(num_chips) + 3 * int64_t(max_senders);
    const std::vector<std::pair<std::string, int64_t>> runtime_args_counts = {
        {"BRISC", 1 + 10 + 3 + 2 * int64_t(num_router_cores) + planner_args_length},
        {"NCRISC", 2 * int64_t(tp_size) + 10 + (num_router_cores < planner_instances ? 2 + planner_args_length : 2)},
        {"compute", tp_size + 9},
    };
    std::string too_many;
    for (const auto& [risc, count] : runtime_args_counts) {
        if (count > kRouterArgs) {
            too_many += (too_many.empty() ? "" : ", ") + ("'" + risc + "': ") + to_text(count);
        }
    }
    if (!too_many.empty()) {
        return "router cores' runtime args: {" + too_many + "} > " + to_text(kRouterArgs) + " (" +
               to_text(num_router_cores) + " router cores, " + to_text(planner_instances) + " planner instances, N " +
               to_text(num_chips) + ")";
    }
    return "";
}

bool is_l1(const HeadPlanned& planned) {
    return planned.note.rfind("arena", 0) == 0 || planned.note.rfind("router cores' arena", 0) == 0 ||
           planned.note.find("of L1") != std::string::npos;
}

// summary: why no candidate fits.
std::string summary(const std::vector<HeadPlanned>& unfit, const ArenaMax& arena_max) {
    std::vector<const HeadPlanned*> l1_refused, other_refused;
    for (const auto& planned : unfit) {
        (is_l1(planned) ? l1_refused : other_refused).push_back(&planned);
    }
    std::vector<std::string> out;
    if (!l1_refused.empty()) {
        const HeadPlanned* least = nullptr;
        int64_t least_excess = 0;
        for (const auto* planned : l1_refused) {
            int64_t excess = (planned->arena ? planned->arena : kUnlimited) - arena_max(planned->layout.k_step_tiles);
            if (!least || excess < least_excess) {
                least = planned;
                least_excess = excess;
            }
        }
        std::set<uint32_t> tried_k_steps, tried_row_chunks;
        for (const auto* planned : l1_refused) {
            tried_k_steps.insert(planned->layout.k_step_tiles);
            tried_row_chunks.insert(planned->layout.row_chunks);
        }
        out.push_back(
            "L1: the least arena any layout needs is " + kilobytes(least->arena) + " KB (" +
            describe_layout(least->layout) + "), a core has " + kilobytes(arena_max(least->layout.k_step_tiles)) +
            " KB; tried K steps " + list_string({tried_k_steps.begin(), tried_k_steps.end()}) + ", row chunks " +
            list_string({tried_row_chunks.begin(), tried_row_chunks.end()}) + ", W rings off / 5 / 4 / 3");
    }
    if (!other_refused.empty()) {
        auto sorted = other_refused;
        std::stable_sort(sorted.begin(), sorted.end(), [](const HeadPlanned* left, const HeadPlanned* right) {
            return std::make_pair(left->layout.row_chunks, -int64_t(left->layout.k_step_tiles)) <
                   std::make_pair(right->layout.row_chunks, -int64_t(right->layout.k_step_tiles));
        });
        std::vector<std::string> reasons;
        const HeadPlanned* first = nullptr;
        for (const auto* planned : sorted) {
            std::string reason = planned->note.substr(0, planned->note.find(':'));
            if (std::find(reasons.begin(), reasons.end(), reason) == reasons.end()) {
                reasons.push_back(reason);
                if (!first) {
                    first = planned;
                }
            }
        }
        std::string more =
            reasons.size() > 1 ? " (and " + to_text(reasons.size() - 1) + " other reason(s))" : std::string();
        out.push_back(first->note + " at " + describe_layout(first->layout) + more);
    }
    std::string joined;
    for (size_t i = 0; i < out.size(); ++i) {
        joined += (i ? "; " : "") + out[i];
    }
    return joined;
}

// degradation: what L1 forced.
std::string degradation(const HeadPlanned& chosen, const std::optional<HeadPlanned>& unlimited) {
    HEAD_PLAN_NO_FP_CONTRACT
    if (!unlimited || chosen.predicted_us <= unlimited->predicted_us * 1.005) {
        return "";
    }
    const auto &chosen_layout = chosen.layout, &unlimited_layout = unlimited->layout;
    std::vector<std::string> changes;
    if (chosen_layout.k_step_tiles != unlimited_layout.k_step_tiles) {
        changes.push_back(
            "K step " + to_text(unlimited_layout.k_step_tiles) + " -> " + to_text(chosen_layout.k_step_tiles));
    }
    if (chosen_layout.row_chunks != unlimited_layout.row_chunks) {
        changes.push_back(
            "row chunks " + to_text(unlimited_layout.row_chunks) + " -> " + to_text(chosen_layout.row_chunks));
    }
    if (chosen_layout.engine_rows != unlimited_layout.engine_rows ||
        chosen_layout.row_tiles_per_core != unlimited_layout.row_tiles_per_core) {
        changes.push_back(
            "matmul core rows x row_tiles_per_core " + to_text(unlimited_layout.engine_rows) + "x" +
            to_text(unlimited_layout.row_tiles_per_core) + " -> " + to_text(chosen_layout.engine_rows) + "x" +
            to_text(chosen_layout.row_tiles_per_core));
    }
    if (chosen_layout.compact_reduce_scatter_zones != unlimited_layout.compact_reduce_scatter_zones) {
        changes.push_back("compact RS zones");
    }
    for (const auto& [name, value] : chosen.parts) {
        if (name == "wring" && value) {
            changes.push_back("W streamed from DRAM every chunk (a ring of " + to_text(value) + " slots)");
        }
    }
    std::string joined;
    for (size_t i = 0; i < changes.size(); ++i) {
        joined += (i ? ", " : "") + changes[i];
    }
    return "L1-limited: " + (joined.empty() ? std::string("another layout") : joined) + " - predicted " +
           format_fixed(chosen.predicted_us, 0) + " us vs " + format_fixed(unlimited->predicted_us, 0) +
           " us with unlimited L1 (+" + format_fixed(chosen.predicted_us - unlimited->predicted_us, 0) + " us, +" +
           format_fixed(100 * (chosen.predicted_us / unlimited->predicted_us - 1), 0) + "%)";
}

}  // namespace head_plan_detail

using namespace head_plan_detail;

ArenaMax arena_max_of(int64_t l1_free, uint32_t payload, int64_t cb_floor) {
    (void)payload;
    return [l1_free, cb_floor](uint32_t) { return arena_cap_of(l1_free, cb_floor); };
}

ArenaMax arena_max_fixed(int64_t bytes) {
    return [bytes](uint32_t) { return bytes; };
}

uint32_t engine_width(const HeadShape& shape) {
    int64_t engine_columns = std::min<int64_t>(int64_t(shape.grid[0]) - 1, shape.shared_expert_dim / 32);
    if (shape.latent) {
        engine_columns = std::min<int64_t>(engine_columns, *shape.latent / 32 / shape.tp_size);
    }
    return engine_columns < 0 ? 0 : uint32_t(engine_columns);
}

std::vector<std::array<uint32_t, 3>> head_layouts(const HeadShape& shape, uint32_t max_row_tiles_per_core) {
    const uint32_t tp_size = shape.tp_size, grid_rows = shape.grid[1];
    const uint32_t engine_columns = engine_width(shape);
    const uint32_t row_tiles = shape.rows / 32;
    std::vector<std::array<uint32_t, 3>> out;
    if (shape.rows % 32 || row_tiles % tp_size) {
        return out;
    }
    const uint32_t owned_row_tiles = row_tiles / tp_size;
    for (uint32_t row_chunks = 1; row_chunks <= owned_row_tiles; ++row_chunks) {
        if (owned_row_tiles % row_chunks) {
            continue;
        }
        const uint32_t chunk_row_tiles = row_tiles / row_chunks;
        for (uint32_t engine_rows = grid_rows; engine_rows > 0; --engine_rows) {
            if (chunk_row_tiles % engine_rows || chunk_row_tiles / engine_rows > max_row_tiles_per_core) {
                continue;
            }
            bool router_fits = false;  // (the router cores fit for some router iteration count)
            for (uint32_t iterations = 1; iterations <= row_chunks && !router_fits; ++iterations) {
                router_fits =
                    row_chunks % iterations == 0 &&
                    router_cells(shape, engine_rows, engine_columns, owned_row_tiles / iterations) != Cells::None;
            }
            if (!router_fits) {
                continue;
            }
            out.push_back({engine_rows, chunk_row_tiles / engine_rows, row_chunks});
        }
    }
    return out;
}

std::vector<uint32_t> k_steps(const HeadShape& shape) {
    const uint32_t k_shard_tiles = shape.hidden_dim / 32 / shape.tp_size;
    std::vector<uint32_t> out;
    for (uint32_t step = 16; step >= 1; --step) {
        if (k_shard_tiles % step == 0 && (k_shard_tiles / step) % 2 == 0) {
            out.push_back(step);
        }
    }
    return out;
}

std::string describe_layout(const HeadLayout& layout) {
    return "matmul cores " + to_text(layout.engine_columns) + "x" + to_text(layout.engine_rows) +
           " row_tiles_per_core " + to_text(layout.row_tiles_per_core) + ", " + to_text(layout.row_chunks) +
           " row chunk" + (layout.row_chunks > 1 ? "s" : "") + ", K step " + to_text(layout.k_step_tiles) +
           (layout.router_iterations > 1 ? ", " + to_text(layout.router_iterations) + " router iterations" : "");
}

HeadSized head_size(const HeadShape& shape, const HeadLayout& layout, int64_t arena_cap) {
    std::optional<ReduceScatterPlan> reduce_scatter;
    std::optional<SharedExpertSizing> sizing;
    try {
        reduce_scatter.emplace(
            shape.tp_size,
            shape.ring && shape.tp_size > 2,
            layout.engine_columns,
            layout.engine_rows,
            layout.row_tiles_per_core,
            shape.hidden_dim / 32,
            shape.reduce_scatter_dim(),
            shape.links,
            kDestTiles / layout.row_tiles_per_core,
            layout.compact_reduce_scatter_zones);
        SharedExpertArgs sizing_args;
        sizing_args.rows = shape.rows;
        sizing_args.hidden_dim = shape.hidden_dim;
        sizing_args.shared_expert_dim = shape.shared_expert_dim;
        sizing_args.engine_columns = layout.engine_columns;
        sizing_args.engine_rows = layout.engine_rows;
        sizing_args.row_tiles_per_core = layout.row_tiles_per_core;
        sizing_args.k_step_tiles = layout.k_step_tiles;
        sizing_args.shards = shape.tp_size;
        sizing_args.reduce_scatter = &*reduce_scatter;
        if (shape.latent) {
            sizing_args.latent = *shape.latent / 32 / shape.tp_size;
        }
        sizing_args.router = std::make_pair(
            shape.router_candidates / 32, shape.hidden_dim / shape.tp_size / 32);  // (the logits on the matmul cores)
        sizing_args.router_iterations = layout.router_iterations;
        sizing_args.arena_max = arena_cap;
        sizing_args.w_dtype = shape.w_dtype;
        sizing.emplace(sizing_args);
    } catch (const ArenaTooLarge& error) {
        return {
            std::nullopt,
            error.need,
            {},
            "arena " + kilobytes(error.need) + " KB > " + kilobytes(arena_cap) + " KB (W ring at the least)"};
    } catch (const SizingError& error) {
        const std::string why = error.what();
        return {std::nullopt, 0, {}, why.empty() ? "sizing failed" : why};
    }
    const int64_t need = align_to_64(router_min_bytes(shape.tp_size, shape.router_candidates, shape.top_k));
    if (need > arena_cap) {
        return {
            std::nullopt,
            need,
            sizing->sizes,
            "router cores' arena " + kilobytes(need) + " KB > " + kilobytes(arena_cap) + " KB"};
    }
    int64_t arena = std::max(sizing->arena_bytes, need);
    std::string why = cores_check(shape, layout, *reduce_scatter, arena, arena_cap);
    if (!why.empty()) {
        return {std::nullopt, arena, sizing->sizes, why};
    }
    Regions sizes = sizing->sizes;
    return {std::move(sizing), arena, std::move(sizes), ""};
}

std::pair<double, Parts> head_cost(
    const HeadShape& shape, const HeadLayout& layout, const SharedExpertSizing& sizing, const HeadCost& cost) {
    HEAD_PLAN_NO_FP_CONTRACT
    const int64_t tp_size = shape.tp_size, row_tiles_per_core = layout.row_tiles_per_core,
                  row_chunks = layout.row_chunks;
    const int64_t hidden_tiles = shape.hidden_dim / 32, shared_expert_tiles = shape.shared_expert_dim / 32;
    int64_t gate_up_width = 0;
    for (uint32_t pairs : sizing.column_pairs) {
        gate_up_width = std::max<int64_t>(gate_up_width, 2 * pairs);
    }
    gate_up_width += shape.latent ? sizing.max_latent_column_tiles : 0;
    int64_t down_width = 0;
    for (const auto& pass_widths : sizing.down_pass_widths) {
        int64_t sum = 0;
        for (uint32_t width : pass_widths) {
            sum += width;
        }
        down_width = std::max(down_width, sum);
    }
    const double gate_up_us =
        double(row_chunks * row_tiles_per_core * hidden_tiles * gate_up_width) * cost.tile_cycles / cost.clock_mhz +
        double(row_chunks * sizing.gate_up_passes * sizing.num_k_steps) * cost.round_us;
    const double down_us =
        double(row_chunks * row_tiles_per_core * shared_expert_tiles * down_width) * cost.tile_cycles / cost.clock_mhz +
        double(row_chunks * sizing.down_passes * sizing.down_k_steps) * cost.round_us;
    const bool ring_hops = shape.ring && tp_size > 2;
    const double rate = double(shape.links) * cost.link_gbs * 1e3;
    double x_us, latent_us = 0.0;
    if (ring_hops) {
        const double hops = double(tp_size - 1) / 2;
        x_us = std::floor(hops * shape.rows * shape.hidden_dim / double(tp_size)) * 2 / rate;
        if (shape.latent) {
            latent_us = std::floor(std::floor(hops * shape.rows * *shape.latent / double(tp_size)) / double(tp_size)) *
                        2 / rate;
        }
    } else {
        const int64_t hops = tp_size - 1;
        x_us = double(hops * shape.rows * shape.hidden_dim / tp_size * 2) / rate;
        if (shape.latent) {
            latent_us = double(hops * shape.rows * int64_t(*shape.latent) / tp_size / tp_size * 2) / rate;
        }
    }
    const double reduce_scatter_us = x_us;
    const int64_t w_bytes = sizing.dram_w_bytes();
    const double w_us = double(w_bytes * (sizing.w_ring_slots ? row_chunks : 1)) / (cost.dram_gbs * 1e3);
    const double w_noc_us = double(row_chunks * w_bytes) / double(layout.engine_columns) / (cost.noc_gbs * 1e3);
    const double math_us = gate_up_us + down_us;
    const double x_multicast_us =
        double(row_chunks * sizing.gate_up_passes) * (double(hidden_tiles) / 2) * double(row_tiles_per_core) *
        kBf16TileBytes / double(std::min(sizing.family0_size, sizing.family1_size)) / (cost.x_multicast_gbs * 1e3);
    const double compute_us =
        std::max({math_us, w_noc_us, x_multicast_us}) / double(row_chunks) + cost.chunk_us * (row_chunks > 1 ? 1 : 0);
    const double steady_us =
        std::max({compute_us, (x_us + latent_us + reduce_scatter_us) / double(row_chunks), w_us / double(row_chunks)});
    double shared_expert_us = (x_us + latent_us) / double(row_chunks) + double(row_chunks - 1) * steady_us +
                              compute_us + reduce_scatter_us / double(row_chunks);
    if (row_chunks == 1) {
        shared_expert_us = std::max(shared_expert_us, w_us + down_us + reduce_scatter_us);
    }
    const int64_t expert_tiles = shape.router_candidates / 32;
    const double router_chain_us =
        cost.chain_us + cost.chain_us_per_expert_tile_k * double(expert_tiles) * double(shape.top_k);
    const int64_t expert_tiles_per_column = ceil_div(expert_tiles, layout.engine_columns);
    const double engine_logits_us = kEngineLogitsOverhead * double(row_chunks) * double(row_tiles_per_core) *
                                    double(expert_tiles_per_column) * double(hidden_tiles / tp_size) *
                                    cost.router_tile_cycles / cost.clock_mhz;
    shared_expert_us += engine_logits_us;  // (the logits on the matmul cores)
    // (router iterations: the router cores' chain once an iteration, a barrier across the TP ring between them)
    const double router_stage_us = engine_logits_us + double(layout.router_iterations) * router_chain_us +
                                   double(layout.router_iterations - 1) * cost.router_barrier_us;
    const double total_us = cost.base_us + std::max(shared_expert_us, router_stage_us);
    Parts parts = {
        {"gu", round_us(gate_up_us)},
        {"down", round_us(down_us)},
        {"x", round_us(x_us)},
        {"rs", round_us(reduce_scatter_us)},
        {"w", round_us(w_us)},
        {"wnoc", round_us(w_noc_us)},
        {"xmc", round_us(x_multicast_us)},
        {"owners", int64_t(std::min(sizing.family0_size, sizing.family1_size))},
        {"se", round_us(shared_expert_us)},
        {"chain", round_us(router_stage_us)},
        {"wring", int64_t(sizing.w_ring_slots)},
    };
    return {total_us, parts};
}

std::vector<HeadPlanned> head_plan_layouts(
    const HeadShape& shape, const ArenaMax& arena_max, bool keep_unfit, uint32_t min_router_iterations) {
    const uint32_t engine_columns = engine_width(shape);
    // each layout with the fewest router iterations (from min_router_iterations) that fit - for one layout more
    // iterations only add router chains and barriers - and the layouts ranked by cost: a layout that needs router
    // iterations wins where its matmul cores outweigh them (e.g. 10x10 row_tiles_per_core 2 at 1280 rows on a 10-wide
    // grid over 10x8 row_tiles_per_core 1 in 5 chunks)
    std::vector<HeadPlanned> out;
    for (const auto& [engine_rows, row_tiles_per_core, row_chunks] : head_layouts(shape)) {
        for (uint32_t k_step_tiles : k_steps(shape)) {
            // (the fewest router iterations that fit: router_iterations divides the row chunks)
            for (uint32_t iterations = min_router_iterations; iterations <= row_chunks; ++iterations) {
                if (row_chunks % iterations) {
                    continue;
                }
                const std::vector<bool> compacts = shape.reduce_scatter_dim() == -2 && shape.tp_size <= 4
                                                       ? std::vector<bool>{false, true}
                                                       : std::vector<bool>{false};
                HeadLayout layout;
                HeadSized sized;
                for (bool compact : compacts) {
                    layout = HeadLayout{
                        .engine_rows = engine_rows,
                        .row_tiles_per_core = row_tiles_per_core,
                        .row_chunks = row_chunks,
                        .k_step_tiles = k_step_tiles,
                        .engine_columns = engine_columns,
                        .compact_reduce_scatter_zones = compact,
                        .router_iterations = iterations};
                    sized = head_size(shape, layout, arena_max(k_step_tiles));
                    if (sized.sizing) {
                        break;
                    }
                }
                if (!sized.sizing) {
                    if (keep_unfit) {
                        out.push_back(HeadPlanned{layout, sized.arena, sized.sizes, kInfinity, {}, sized.reason});
                    }
                    continue;
                }
                auto [predicted_us, parts] = head_cost(shape, layout, *sized.sizing);
                out.push_back(HeadPlanned{layout, sized.arena, sized.sizes, predicted_us, parts, ""});
                break;
            }
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const HeadPlanned& left, const HeadPlanned& right) {
        return std::make_pair(left.predicted_us, -int64_t(left.layout.k_step_tiles)) <
               std::make_pair(right.predicted_us, -int64_t(right.layout.k_step_tiles));
    });
    return out;
}

std::string shape_checks(const HeadShape& shape) {
    const uint32_t tp_size = shape.tp_size, hidden_dim = shape.hidden_dim;
    if (tp_size < 2) {
        return "TP " + to_text(tp_size) + ": the MoE head gathers x over the TP axis (TP >= 2)";
    }
    if (tp_size > 4 && shape.latent) {
        return "TP " + to_text(tp_size) +
               " > 4 with a latent: the latent all-to-all sends a port's destination blocks under one packet header "
               "each - at most 3 (a port's latent block) - a mesh with TP 2 or 4";
    }
    if (tp_size > 8) {
        return "TP " + to_text(tp_size) + " > 8: the routing / combine port tables (a mesh with TP <= 8)";
    }
    if (shape.grid[0] < 2 || shape.grid[0] > 12) {
        return "a " + to_text(shape.grid[0]) +
               "-column compute grid: the head is at most 11 matmul-core columns and the " +
               "movers' column (2 to 12 columns)";
    }
    if (shape.links != 1 && shape.links != 2) {
        return to_text(shape.links) + " links: 1 or 2 (a port per direction and link)";
    }
    if (hidden_dim % (32 * tp_size)) {
        return "H " + to_text(hidden_dim) + " is not whole tiles over TP " + to_text(tp_size) + " (a multiple of " +
               to_text(32 * tp_size) + ")";
    }
    const uint32_t k_shard_tiles = hidden_dim / 32 / tp_size;
    if (k_steps(shape).empty()) {
        return "H / 32 / TP = " + to_text(k_shard_tiles) +
               " tiles a shard: no K step <= 16 gives an even count a shard (an even shard)";
    }
    if (shape.shared_expert_dim < 32 || shape.shared_expert_dim % 32) {
        return "the shared expert's I / TP = " + to_text(shape.shared_expert_dim) +
               ": whole tiles, at least one (I_s a multiple of " + to_text(32 * tp_size) + ")";
    }
    if (shape.latent && (*shape.latent % (32 * tp_size))) {
        return "latent " + to_text(*shape.latent) + " is not whole tiles over TP " + to_text(tp_size) +
               " (a multiple of " + to_text(32 * tp_size) + ")";
    }
    if (shape.top_k % 2 || shape.top_k > 16) {
        return "top-K " + to_text(shape.top_k) +
               ": even and <= 16 (joint routing rows - K ids + K weights in 64 B - dispatch prep, routing exchange)";
    }
    if (shape.router_candidates % 32) {
        return to_text(shape.router_candidates) + " router candidates: whole tiles (a multiple of 32)";
    }
    if (shape.rows % (32 * tp_size)) {
        const int64_t nearest = ceil_div(shape.rows, 32 * tp_size) * 32 * tp_size;
        return "rows a mesh row " + to_text(shape.rows) + " must be a multiple of 32 x TP = " + to_text(32 * tp_size) +
               "; nearest valid " + to_text(nearest);
    }
    const uint32_t num_chips = shape.num_chips(), tokens_per_chip = shape.tokens_per_chip_all_blocks(),
                   experts_per_chip = shape.experts_per_chip();
    if (num_chips > moe_block_plan_words::kMaxChips || shape.accumulators > moe_block_plan_words::kAccMax) {
        return "routing planner: N " + to_text(num_chips) + " <= 32 chips, accumulators " +
               to_text(shape.accumulators) + " <= " + to_text(moe_block_plan_words::kAccMax);
    }
    if (tokens_per_chip > 2048) {
        return "routing planner: " + to_text(tokens_per_chip) +
               " tokens a chip > 2048 (an entry's token field); chunk <= " + to_text(2048 * num_chips) + " tokens";
    }
    if (experts_per_chip > 64 || num_chips * experts_per_chip > 1024) {
        return "routing planner: " + to_text(experts_per_chip) + " experts a chip <= 64 and N x EPC " +
               to_text(num_chips * experts_per_chip) + " <= 1024 (fold the router)";
    }
    if (engine_width(shape) < 1) {
        return "no matmul core column: I / TP " + to_text(shape.shared_expert_dim) + " or latent tiles a chip";
    }
    return "";
}

HeadFit head_plan_fit(const HeadShape& shape, const HeadFitOptions& options) {
    return head_plan_fit(
        shape, arena_max_of(options.l1_free.value_or(kL1Free - kL1Margin), shape.payload, options.cb_floor), options);
}

HeadFit head_plan_fit(const HeadShape& shape, const ArenaMax& arena_max, const HeadFitOptions& options) {
    HeadFit fit;
    fit.reason = shape_checks(shape);
    if (!fit.reason.empty()) {
        return fit;
    }
    if (head_layouts(shape).empty()) {
        const uint32_t owned_row_tiles = shape.rows / 32 / shape.tp_size;
        fit.reason =
            "no matmul-core layout for " + to_text(shape.rows) + " rows a mesh row (" + to_text(owned_row_tiles) +
            " row tiles an owner: row chunks of whole owner blocks, row_tiles_per_core <= 2, the router cores' cells)";
        return fit;
    }
    const auto candidates = head_plan_layouts(shape, arena_max, true, options.min_router_iterations);
    std::vector<HeadPlanned> fitting;
    for (const auto& planned : candidates) {
        if (planned.predicted_us != kInfinity) {
            fitting.push_back(planned);
        }
    }
    if (fitting.empty()) {
        std::vector<HeadPlanned> unfit;
        for (const auto& planned : candidates) {
            if (planned.predicted_us == kInfinity) {
                unfit.push_back(planned);
            }
        }
        if (unfit.empty()) {
            fit.reason = "no layout";
            return fit;
        }
        std::string why = summary(unfit, arena_max);
        if (options.largest && why.find("L1:") != std::string::npos) {
            const uint32_t step = 32 * shape.tp_size;
            int64_t low = 0, high = int64_t(shape.rows / step) - 1;
            HeadFitOptions sub_options = options;
            sub_options.largest = false;
            sub_options.explain = false;
            while (low < high) {
                const int64_t middle = (low + high + 1) / 2;
                HeadShape smaller = shape;
                smaller.rows = uint32_t(middle * step);
                if (head_plan_fit(smaller, arena_max, sub_options).ok) {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            if (low) {
                why += "; the largest rows a mesh row that fit: " + to_text(low * step) + " (chunk " +
                       to_text(low * step * shape.sp_rows) + ")";
            }
        }
        fit.reason = why;
        return fit;
    }
    fit.ok = true;
    fit.planned = fitting;
    if (!options.explain) {
        return fit;
    }
    try {
        const auto unlimited = head_plan_layouts(shape, arena_max_fixed(kUnlimited), false);
        if (!unlimited.empty()) {
            fit.unconstrained = unlimited.front();
        }
    } catch (const std::exception&) {  // (the unconstrained pick only explains the note)
        fit.unconstrained.reset();
    }
    fit.note = degradation(fit.planned[0], fit.unconstrained);
    return fit;
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head

#if !defined(__clang__)
#pragma GCC pop_options
#endif
#undef HEAD_PLAN_NO_FP_CONTRACT
