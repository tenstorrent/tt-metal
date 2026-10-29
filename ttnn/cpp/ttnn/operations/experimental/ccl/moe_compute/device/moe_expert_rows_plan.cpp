// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_expert_rows_plan.hpp"

#include <algorithm>
#include <array>
#include <set>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_align.hpp>

#include "kernels/moe_ring_common.h"
#include "moe_compute_program_factory.hpp"

namespace ttnn::experimental::prim::expert_rows_host {

namespace {

constexpr uint32_t kTileRowBytesBf16 = tt::constants::TILE_WIDTH * kBf16Bytes;
constexpr std::array<uint32_t, 3> kBlockSlots = {8, 6, 4};  // weight blocks per reader core, tried in order
constexpr uint32_t kBlockBytes = 16 * 1024;                 // weight bytes per read block
constexpr uint32_t kTwoReadersCoresPerBank = 8;  // worker cores per DRAM bank from which a bank gets 2 readers
// a reader keeps one weight block in flight when its bank feeds this many cores or more, else two
constexpr uint32_t kOneBlockInFlightBankCores = 3;
// Assignment of a bank's further cores: the grid distance to its optimal reader core, plus a cost for another row, a
// smaller x, a smaller y, and a smaller x than an optimal core off grid column 0.
constexpr double kAssignOtherRowCost = 0.1;
constexpr double kAssignSmallerXCost = 0.3;
constexpr double kAssignSmallerYCost = 0.05;
constexpr double kAssignSmallerXOffColumnZeroCost = 6.0;
// (x slots, intermediate slots) tried in order; without credits at least three intermediate slots (rows_writer.cpp)
constexpr std::array<std::pair<uint32_t, uint32_t>, 3> kBuffering = {{{2, 4}, {2, 3}, {1, 3}}};
// Several row tiles per job: two x chunk slots (a chunk moves while the previous one computes) and one intermediate
// slot, both handed back by credits (rows_writer.cpp). The x chunks keep the group cores within a few chunks of each
// other, so the intermediate slot is free again by the time it is needed; its L1 goes to larger x chunks (fewer partial
// reloads).
constexpr uint32_t kChunkXSlots = 2;
constexpr uint32_t kChunkIntermediateSlots = 1;
constexpr uint32_t kActivatedSlots = 2;  // cb_activated: the activated tiles of two jobs
constexpr uint32_t kRowsSlots = 2;       // cb_rows: two output groups of one row tile

uint32_t align_to_part(uint32_t bytes) { return tt::align(bytes, expert_rows::part_alignment); }

// Consecutive parts of a layout, each at a multiple of part_alignment.
struct LayoutCursor {
    uint32_t end = 0;
    uint32_t take(uint32_t bytes) {
        const uint32_t offset = end;
        end = align_to_part(end + bytes);
        return offset;
    }
};

RoutingLayout routing_layout(
    const MoEExpertRowsShape& shape, uint32_t program_cores, uint32_t sources_per_core, uint32_t expert_groups) {
    const uint32_t tokens_per_core = tt::div_up(shape.tokens, program_cores);
    const uint32_t area_bytes =
        expert_rows::entry_area_bytes(expert_rows::entry_capacity(shape.tokens, shape.top_k, program_cores));
    const auto block = expert_rows::table_layout(shape.tokens, shape.top_k, shape.local_experts);
    LayoutCursor cursor;
    RoutingLayout layout;
    layout.own_mapping = cursor.take(shape.global_experts * kBf16Bytes);
    layout.source_mappings = cursor.take(sources_per_core * shape.mapping_page_bytes);
    layout.token_ids = cursor.take(tokens_per_core * shape.index_page_bytes);
    layout.token_scores = cursor.take(tokens_per_core * shape.index_page_bytes);
    layout.ctl = cursor.take(expert_rows::ctl_bytes(expert_groups));
    layout.active_experts = cursor.take(shape.local_experts * kBf16Bytes);
    layout.expert_rows = cursor.take(shape.local_experts * kBf16Bytes);
    layout.expert_first_rows = cursor.take(shape.local_experts * kBf16Bytes);
    layout.table = cursor.take(block.bytes);
    layout.row_tokens = layout.table + block.row_tokens;
    layout.jobs = cursor.take(shape.tokens * shape.top_k * sizeof(expert_rows::JobRecord));
    layout.entry_areas = cursor.take(program_cores * area_bytes);
    layout.size = cursor.end;
    return layout;
}

// The writer's scratch (cb_scratch): the x staging (tile_rows rows per row tile, each the core's piece of an x chunk of
// at most piece_max tiles), that piece as tiles, and the staging of one row tile of output rows (one output group).
uint64_t writer_scratch_bytes(uint32_t row_tiles, uint32_t piece_max) {
    return uint64_t{expert_rows::tile_rows} * row_tiles * piece_max * kTileRowBytesBf16 +
           uint64_t{piece_max} * row_tiles * kTileBytesBf16 + expert_rows::tile_rows * kGroupTiles * kTileRowBytesBf16;
}

// Weight blocks of cb_w when jobs hold units (several row tiles): whole chunk units (none wraps around the CB), at
// least two, at least block_slots.
uint32_t weight_cb_blocks(uint32_t chunk_blocks, uint32_t block_slots) {
    return chunk_blocks * std::max(2u, tt::div_up(block_slots, chunk_blocks));
}

// cb_ctl: u32 job count, then with several row tiles per job the row tiles of each of the group's jobs
uint32_t ctl_cb_bytes(uint32_t words) { return tt::align(kWordBytes * words, expert_rows::part_alignment); }

// The CBs of a buffering choice, when they fit cb_budget bytes per core.
std::optional<CbSizes> cb_sizes(
    const MoEExpertRowsShape& shape,
    const Buffering& buffering,
    uint32_t activated_tiles,
    uint32_t block_tiles,
    uint32_t routing_bytes,
    uint32_t partial_tile_bytes,
    uint32_t cb_budget) {
    const uint32_t hidden_tiles = shape.hidden_tiles(), intermediate_tiles = shape.intermediate_tiles();
    const bool spill = buffering.chunk_tiles < hidden_tiles || buffering.w2_chunk_rows < intermediate_tiles;
    const uint64_t x_chunks =
        uint64_t{buffering.x_slots} * buffering.chunk_tiles * buffering.row_tiles * kTileBytesBf16;
    const uint64_t weights = uint64_t{buffering.block_slots} * block_tiles * tt::tile_size(tt::DataFormat::Bfp4_b);
    const uint64_t activated = uint64_t{kActivatedSlots} * buffering.row_tiles * activated_tiles * kTileBytesBf16;
    const uint64_t intermediate =
        uint64_t{buffering.intermediate_slots} * buffering.row_tiles * intermediate_tiles * kTileBytesBf16;
    const uint64_t rows = kRowsSlots * kGroupTiles * kTileBytesBf16;
    const uint64_t scratch = writer_scratch_bytes(buffering.row_tiles, buffering.chunk_piece_max);
    // a 4-tile slot per (column group, row tile), also for a half group (rows_compute.cpp)
    const uint64_t partials = spill ? uint64_t{buffering.row_tiles} * std::max(activated_tiles, kColumnsPerGroup) *
                                          kGroupTiles / kColumnsPerGroup * partial_tile_bytes
                                    : 0;
    const uint64_t ones = shape.has_bias ? kTileBytesBf16 : 0;
    const uint64_t total = x_chunks + weights + activated + intermediate + rows + buffering.ctl_bytes + routing_bytes +
                           scratch + partials + ones;
    if (total > cb_budget) {
        return std::nullopt;
    }
    return CbSizes{
        .x_chunks = static_cast<uint32_t>(x_chunks),
        .weights = static_cast<uint32_t>(weights),
        .activated = static_cast<uint32_t>(activated),
        .intermediate = static_cast<uint32_t>(intermediate),
        .rows = static_cast<uint32_t>(rows),
        .ctl = buffering.ctl_bytes,
        .routing = routing_bytes,
        .scratch = static_cast<uint32_t>(scratch),
        .ones = static_cast<uint32_t>(ones),
        .partials = static_cast<uint32_t>(partials)};
}

// The cores of one expert group: each ring position's column groups split over up to readers_per_bank cores (earlier
// cores take the remainder), and its output groups over the same cores.
struct CoreSplit {
    uint32_t readers_per_bank = 0, block_tiles = 0, block_packets = 0;
    uint32_t activated_tiles = 0;
    std::vector<ExpertRowsCore> members;
};

// Each bank's cores: its NOC-0-optimal DRAM reader core first, then the nearest free cores (round-robin over banks).
std::vector<std::vector<CoreCoord>> assign_cores(
    const std::vector<CoreCoord>& optimal, const std::vector<uint32_t>& counts, CoreCoord grid) {
    std::set<std::pair<uint32_t, uint32_t>> taken;
    std::vector<std::vector<CoreCoord>> out(optimal.size());
    for (uint32_t bank = 0; bank < optimal.size(); ++bank) {
        if (counts[bank]) {
            out[bank].push_back(optimal[bank]);
            taken.insert({optimal[bank].x, optimal[bank].y});
        }
    }
    const uint32_t most = *std::max_element(counts.begin(), counts.end());
    for (uint32_t step = 1; step < most; ++step) {
        for (uint32_t bank = 0; bank < optimal.size(); ++bank) {
            if (out[bank].size() >= counts[bank]) {
                continue;
            }
            const int optimal_x = optimal[bank].x, optimal_y = optimal[bank].y;
            double best = std::numeric_limits<double>::max();
            CoreCoord pick{0, 0};
            for (uint32_t x = 0; x < grid.x; ++x) {
                for (uint32_t y = 0; y < grid.y; ++y) {
                    if (taken.contains({x, y})) {
                        continue;
                    }
                    const int delta_x = static_cast<int>(x) - optimal_x, delta_y = static_cast<int>(y) - optimal_y;
                    double cost = std::abs(delta_x) + std::abs(delta_y) + (delta_y != 0 ? kAssignOtherRowCost : 0.0) +
                                  (delta_x < 0 ? kAssignSmallerXCost : 0.0) +
                                  (delta_y < 0 ? kAssignSmallerYCost : 0.0) +
                                  (optimal_x > 0 && delta_x < 0 ? kAssignSmallerXOffColumnZeroCost : 0.0);
                    if (cost < best) {
                        best = cost;
                        pick = CoreCoord(x, y);
                    }
                }
            }
            TT_FATAL(
                best < std::numeric_limits<double>::max(),
                "moe_compute expert rows: the {}x{} worker grid is full; bank {} needs {} cores, has {}",
                grid.x,
                grid.y,
                bank,
                counts[bank],
                out[bank].size());
            out[bank].push_back(pick);
            taken.insert({pick.x, pick.y});
        }
    }
    return out;
}

// W2 output groups per member: contiguous runs in member order on top of its gate/up tiles, the busiest member least
// loaded, then the least sum of squared loads; earlier members take more on a tie (with equal runs, the
// least-loaded-member greedy deal). best[member_end][group_end]: (busiest, sum of squares) of members [0, member_end)
// taking groups [0, group_end), the last of them groups [run_begin, group_end); first[member_end][group_end]: that
// run_begin.
std::vector<uint32_t> split_output_groups(
    const std::vector<uint64_t>& gate_up_tiles, const std::vector<uint64_t>& output_prefix) {
    const uint32_t members = gate_up_tiles.size(), groups = output_prefix.size() - 1;
    using Cost = std::pair<uint64_t, uint64_t>;
    constexpr Cost kNone = {std::numeric_limits<uint64_t>::max(), 0};
    std::vector<std::vector<Cost>> best(members + 1, std::vector<Cost>(groups + 1, kNone));
    std::vector<std::vector<uint32_t>> first(members + 1, std::vector<uint32_t>(groups + 1, 0));
    best[0][0] = {0, 0};
    for (uint32_t member_end = 1; member_end <= members; ++member_end) {
        for (uint32_t group_end = 0; group_end <= groups; ++group_end) {
            for (uint32_t run_begin = 0; run_begin <= group_end; ++run_begin) {
                if (best[member_end - 1][run_begin] == kNone) {
                    continue;
                }
                const uint64_t load =
                    gate_up_tiles[member_end - 1] + output_prefix[group_end] - output_prefix[run_begin];
                const Cost cost = {
                    std::max(best[member_end - 1][run_begin].first, load),
                    best[member_end - 1][run_begin].second + load * load};
                if (cost <= best[member_end][group_end]) {
                    best[member_end][group_end] = cost;
                    first[member_end][group_end] = run_begin;
                }
            }
        }
    }
    std::vector<uint32_t> outputs(members);
    for (uint32_t member_end = members, group_end = groups; member_end > 0; --member_end) {
        outputs[member_end - 1] = group_end - first[member_end][group_end];
        group_end = first[member_end][group_end];
    }
    return outputs;
}

// The plan search of one call: the core splits, their placement and their buffering.
struct Planner {
    const MoEExpertRowsShape& shape;
    const std::vector<uint32_t>& shard_banks;  // DRAM bank of the prepared weights' shard r (= ring position r)
    const std::vector<CoreCoord>& optimal;     // NOC-0-optimal DRAM reader core of each bank
    CoreCoord grid;
    uint32_t cb_budget = 0;
    uint32_t partial_tile_bytes = 0;
    const std::vector<ShardLayout>& layouts;
    const PreparedLayout& prepared;

    uint32_t grid_cores() const { return grid.x * grid.y; }

    // sources one core's tokens can come from: its tokens over the tokens per source, plus one where they straddle two
    uint32_t sources_per_core(uint32_t program_cores) const {
        return tt::div_up(tt::div_up(shape.tokens, program_cores), shape.tokens / shape.num_sources) + 1;
    }

    CoreSplit split_cores(uint32_t readers_per_bank, uint32_t block_tiles, uint32_t block_packets) const {
        CoreSplit split{
            .readers_per_bank = readers_per_bank, .block_tiles = block_tiles, .block_packets = block_packets};
        const uint32_t w0_w1_k_rows = shape.hidden_tiles() + shape.bias_rows();
        const uint32_t w2_k_rows = shape.intermediate_tiles() + shape.bias_rows();
        for (uint32_t position = 0; position < shard_banks.size(); ++position) {
            const ShardLayout& shard = layouts[position];
            const GroupRun columns = shard.column_groups();
            const uint32_t readers = std::min(readers_per_bank, columns.count), output_groups = shard.output_groups();
            std::vector<uint32_t> member_groups(readers);
            std::vector<uint64_t> gate_up_tiles(readers), output_prefix(output_groups + 1, 0);
            for (uint32_t reader = 0, column_group = 0; reader < readers; ++reader) {
                member_groups[reader] = columns.count / readers + (reader < columns.count % readers ? 1 : 0);
                for (const uint32_t end = column_group + member_groups[reader]; column_group < end; ++column_group) {
                    gate_up_tiles[reader] += columns.width(column_group) * w0_w1_k_rows;
                }
            }
            for (uint32_t output_group = 0; output_group < output_groups; ++output_group) {
                output_prefix[output_group + 1] =
                    output_prefix[output_group] + prepared.w2_groups.width(output_group) * w2_k_rows;
            }
            const std::vector<uint32_t> member_outputs = split_output_groups(gate_up_tiles, output_prefix);
            uint32_t first_group = 0, first_output = 0;
            for (uint32_t reader = 0; reader < readers; ++reader) {
                ExpertRowsCore member;
                member.ring_position = position;
                member.bank = shard_banks[position];
                member.first_column_group = first_group;
                member.column_groups = member_groups[reader];
                member.half_column = shard.half_column && first_group + member_groups[reader] == columns.count;
                member.first_column = shard.first_column + kColumnsPerGroup * first_group;
                member.activated_columns =
                    std::min(shard.columns - kColumnsPerGroup * first_group, kColumnsPerGroup * member_groups[reader]);
                member.first_output_group = first_output;
                member.output_groups = member_outputs[reader];
                member.half_output = prepared.w2_groups.half_last && member_outputs[reader] > 0 &&
                                     first_output + member_outputs[reader] == prepared.w2_groups.count;
                member.first_output_tile = shard.first_output_tile + kGroupTiles * first_output;
                member.output_tiles = member_outputs[reader] ? std::min(
                                                                   shard.output_tiles - kGroupTiles * first_output,
                                                                   kGroupTiles * member_outputs[reader])
                                                             : 0;
                split.members.push_back(member);
                first_group += member_groups[reader];
                first_output += member_outputs[reader];
            }
        }
        for (const auto& member : split.members) {
            split.activated_tiles = std::max(split.activated_tiles, kColumnsPerGroup * member.column_groups);
        }
        return split;
    }

    // expert_groups copies of the split, each core near its bank's optimal reader core, group-major.
    void place_groups(ExpertRowsPlan& plan, const CoreSplit& split, uint32_t expert_groups) const {
        const uint32_t num_shards = shard_banks.size();
        std::vector<uint32_t> counts(num_shards, 0);
        for (const auto& member : split.members) {
            counts[member.ring_position] += expert_groups;
        }
        std::vector<uint32_t> bank_counts(optimal.size(), 0);
        for (uint32_t position = 0; position < num_shards; ++position) {
            bank_counts[shard_banks[position]] = counts[position];
        }
        const auto assigned = assign_cores(optimal, bank_counts, grid);
        std::vector<uint32_t> first_member(num_shards, 0);
        for (uint32_t member_index = split.members.size(); member_index-- > 0;) {
            first_member[split.members[member_index].ring_position] = member_index;
        }
        plan.readers_per_bank = split.readers_per_bank;
        plan.expert_groups = expert_groups;
        plan.group_cores = split.members.size();
        plan.activated_tiles = split.activated_tiles;
        plan.block_tiles = split.block_tiles;
        plan.block_packets = split.block_packets;
        plan.blocks_in_flight =
            *std::max_element(bank_counts.begin(), bank_counts.end()) >= kOneBlockInFlightBankCores ? 1 : 2;
        for (uint32_t expert_group = 0; expert_group < expert_groups; ++expert_group) {
            for (uint32_t member_index = 0; member_index < split.members.size(); ++member_index) {
                ExpertRowsCore member = split.members[member_index];
                member.expert_group = expert_group;
                member.core =
                    assigned[member.bank]
                            [(member_index - first_member[member.ring_position]) * expert_groups + expert_group];
                plan.cores.push_back(member);
            }
        }
    }

    // The plan of `buffering` over expert_groups copies of the split, when its CBs fit cb_budget.
    std::optional<ExpertRowsPlan> make_plan(
        const CoreSplit& split,
        const Buffering& buffering,
        uint32_t expert_groups,
        const RoutingLayout& routing) const {
        const auto cbs = cb_sizes(
            shape, buffering, split.activated_tiles, split.block_tiles, routing.size, partial_tile_bytes, cb_budget);
        if (!cbs.has_value()) {
            return std::nullopt;
        }
        ExpertRowsPlan plan;
        static_cast<Buffering&>(plan) = buffering;
        plan.cbs = *cbs;
        plan.routing = routing;
        place_groups(plan, split, expert_groups);
        return plan;
    }

    // One row tile per job: x in one chunk (one slot per job), W2 in whole runs, nothing spilled; the intermediate
    // exchange orders slot reuse (rows_writer.cpp), so x / intermediate slots come before expert groups.
    std::optional<ExpertRowsPlan> plan_one_row_tile(const CoreSplit& split) const {
        const uint32_t group_cores = split.members.size();
        const uint32_t hidden_tiles = shape.hidden_tiles();
        const uint32_t piece_max = tt::div_up(hidden_tiles, group_cores);
        for (const uint32_t block_slots : kBlockSlots) {
            for (const auto& [x_slots, intermediate_slots] : kBuffering) {
                for (uint32_t expert_groups = grid_cores() / group_cores; expert_groups >= 1; --expert_groups) {
                    const uint32_t program_cores = expert_groups * group_cores;
                    const RoutingLayout routing =
                        routing_layout(shape, program_cores, sources_per_core(program_cores), expert_groups);
                    const Buffering buffering{
                        .row_tiles = 1,
                        .x_slots = x_slots,
                        .intermediate_slots = intermediate_slots,
                        .block_slots = block_slots,
                        .chunk_tiles = hidden_tiles,
                        .chunk_piece_max = piece_max,
                        .chunk_blocks =
                            expert_rows::run_blocks(kGroupTiles, hidden_tiles + shape.bias_rows(), split.block_tiles),
                        .w2_chunk_rows = shape.intermediate_tiles(),
                        .ctl_bytes = ctl_cb_bytes(1)};
                    auto plan = make_plan(split, buffering, expert_groups, routing);
                    if (plan.has_value()) {
                        return plan;
                    }
                }
            }
        }
        return std::nullopt;
    }

    // Several row tiles per job: x moves in chunks of chunk_tiles hidden tiles (whole K rows of a column group per
    // weight block where possible; the largest chunk that fits, at most all of them), W2 in chunks that fill the same
    // number of weight blocks.
    std::optional<ExpertRowsPlan> plan_row_tiles(const CoreSplit& split, uint32_t row_tiles) const {
        const uint32_t group_cores = split.members.size();
        const uint32_t hidden_tiles = shape.hidden_tiles();
        const uint32_t chunk_unit = split.block_tiles / kGroupTiles;
        std::vector<uint32_t> chunks = {hidden_tiles};
        for (uint32_t chunk = tt::round_down(hidden_tiles - 1, chunk_unit); chunk >= chunk_unit; chunk -= chunk_unit) {
            chunks.push_back(chunk);
        }
        // jobs of at most 32 row_tiles rows of one expert: at most one partial job per expert plus the full ones
        const uint32_t jobs_max =
            shape.local_experts + shape.tokens * shape.top_k / (expert_rows::tile_rows * row_tiles);
        for (const uint32_t block_slots : kBlockSlots) {
            for (uint32_t expert_groups = grid_cores() / group_cores; expert_groups >= 1; --expert_groups) {
                const uint32_t program_cores = expert_groups * group_cores;
                const RoutingLayout routing =
                    routing_layout(shape, program_cores, sources_per_core(program_cores), expert_groups);
                const uint32_t ctl_bytes = ctl_cb_bytes(1 + tt::div_up(jobs_max, expert_groups));
                for (const uint32_t chunk : chunks) {
                    const uint32_t chunk_blocks =
                        expert_rows::run_blocks(kGroupTiles, chunk + shape.bias_rows(), split.block_tiles);
                    const Buffering buffering{
                        .row_tiles = row_tiles,
                        .x_slots = kChunkXSlots,
                        .intermediate_slots = kChunkIntermediateSlots,
                        .block_slots = weight_cb_blocks(chunk_blocks, block_slots),
                        .chunk_tiles = chunk,
                        .chunk_piece_max = tt::div_up(chunk, group_cores),
                        .chunk_blocks = chunk_blocks,
                        .w2_chunk_rows = chunk_blocks * split.block_tiles / kGroupTiles - shape.bias_rows(),
                        .ctl_bytes = ctl_bytes};
                    auto plan = make_plan(split, buffering, expert_groups, routing);
                    if (plan.has_value()) {
                        return plan;
                    }
                }
            }
        }
        return std::nullopt;
    }
};

}  // namespace

PreparedLayout prepared_layout(uint32_t hidden_tiles, uint32_t intermediate_tiles, bool has_bias, uint32_t num_shards) {
    const uint32_t tiles_per_txn =
        moe_ring::tiles_per_txn_for_shape(hidden_tiles, intermediate_tiles, has_bias, num_shards);
    const uint32_t block = moe_ring::W0_W1_TXNS_PER_BLOCK * tiles_per_txn;
    const uint32_t bias_rows = has_bias ? 1 : 0;
    const uint32_t w0_w1_k_rows = hidden_tiles + bias_rows, w2_k_rows = intermediate_tiles + bias_rows;
    PreparedLayout layout;
    layout.w0_w1_piece_tiles = moe_ring::w0_w1_bank_blocks_per_expert(
                                   w0_w1_k_rows, intermediate_tiles, num_shards, num_shards, tiles_per_txn) *
                               block;
    layout.w0_w1_group_tiles = tt::div_up(w0_w1_k_rows, moe_ring::block_tiles_h(tiles_per_txn)) * block;
    layout.w2_slice_tiles =
        moe_ring::w2_core_blocks_per_expert(hidden_tiles, w2_k_rows, num_shards, tiles_per_txn) * block;
    layout.w2_group_tiles = tt::div_up(w2_k_rows, moe_ring::block_tiles_h(tiles_per_txn)) * block;
    layout.w2_groups = {
        moe_ring::w2_num_a2a_iters(hidden_tiles, num_shards),
        moe_ring::w2_last_a2a_iter_half(hidden_tiles, num_shards, tiles_per_txn)};
    return layout;
}

std::vector<ShardLayout> shard_layouts(
    uint32_t hidden_tiles, uint32_t intermediate_tiles, bool has_bias, uint32_t num_shards) {
    const uint32_t tiles_per_txn =
        moe_ring::tiles_per_txn_for_shape(hidden_tiles, intermediate_tiles, has_bias, num_shards);
    const uint32_t w0_w1_k_rows = hidden_tiles + (has_bias ? 1 : 0);
    const uint32_t column_blocks = tt::div_up(w0_w1_k_rows, moe_ring::block_tiles_h(tiles_per_txn));
    const uint32_t half_column_blocks = tt::div_up(w0_w1_k_rows, moe_ring::half_block_tiles_h(tiles_per_txn));
    const bool compact = moe_ring::w0_w1_compact_for_shape(intermediate_tiles, num_shards);
    std::vector<ShardLayout> layouts(num_shards);
    uint32_t column = 0, output_tile = 0;
    for (uint32_t position = 0; position < num_shards; ++position) {
        layouts[position].columns = moe_ring::shard_tiles(intermediate_tiles, position, num_shards);
        layouts[position].first_column = column;
        column += layouts[position].columns;
        layouts[position].output_tiles =
            moe_ring::w2_shard_tiles(hidden_tiles, position, intermediate_tiles, num_shards);
        layouts[position].first_output_tile = output_tile;
        output_tile += layouts[position].output_tiles;
        layouts[position].w0_w1_first_tile =
            moe_ring::w0_w1_core_block_offset(
                intermediate_tiles, position, num_shards, column_blocks, half_column_blocks) *
            moe_ring::W0_W1_TXNS_PER_BLOCK * tiles_per_txn;
        layouts[position].half_column = compact && layouts[position].columns % 2 == 1;
    }
    return layouts;
}

PlanOrRefusal plan_expert_rows(
    ttnn::MeshDevice* mesh_device,
    const MoEExpertRowsShape& shape,
    const std::vector<uint32_t>& shard_banks,
    uint32_t cb_budget,
    uint32_t row_tiles,
    tt::DataFormat partials) {
    const uint32_t hidden_tiles = shape.hidden_tiles();
    const uint32_t intermediate_tiles = shape.intermediate_tiles();
    const std::vector<CoreCoord> optimal =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0);
    const uint32_t num_shards = shard_banks.size();
    if (num_shards != optimal.size()) {
        return "the prepared weights have " + std::to_string(num_shards) + " shards for " +
               std::to_string(optimal.size()) + " DRAM banks";
    }
    if (row_tiles == 0) {
        return std::string("zero row tiles per job");
    }
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    // a read block is kBlockBytes of whole K rows of a column group in as many NoC packets as that takes; one packet
    // per block when the larger block does not fit
    const uint32_t weight_tile_bytes = tt::tile_size(tt::DataFormat::Bfp4_b);
    const uint32_t burst = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    const uint32_t packet_tiles = burst / weight_tile_bytes;
    std::vector<uint32_t> packet_counts = {std::max(1u, kBlockBytes / burst)};
    if (packet_counts[0] > 1) {
        packet_counts.push_back(1);
    }
    if (packet_tiles < kGroupTiles) {
        return fmt::format(
            "a NoC packet of {} B holds {} weight tiles of {} B, less than one K row of a column group ({} tiles)",
            burst,
            packet_tiles,
            weight_tile_bytes,
            kGroupTiles);
    }
    const auto layouts = shard_layouts(hidden_tiles, intermediate_tiles, shape.has_bias, num_shards);
    const PreparedLayout prepared = prepared_layout(hidden_tiles, intermediate_tiles, shape.has_bias, num_shards);
    uint32_t most_groups = 0;
    for (uint32_t position = 0; position < num_shards; ++position) {
        const ShardLayout& shard = layouts[position];
        if (shard.columns == 0) {
            return fmt::format(
                "DRAM bank {} (ring position {}) holds no intermediate column: {} intermediate tiles over {} banks",
                shard_banks[position],
                position,
                intermediate_tiles,
                num_shards);
        }
        if (shard.output_tiles == 0) {
            return fmt::format(
                "DRAM bank {} (ring position {}) holds no W2 output tile: {} hidden tiles over {} banks",
                shard_banks[position],
                position,
                hidden_tiles,
                num_shards);
        }
        most_groups = std::max(most_groups, shard.column_groups().count);
    }
    // reader cores per bank: 1 while the grid has fewer than kTwoReadersCoresPerBank worker cores per DRAM bank, else
    // 2; a larger split (up to the busiest bank's column groups) only when fewer block slots do not fit. Jobs of
    // several row tiles keep the first split: larger groups halve the groups that stream at once and double the
    // all-to-all.
    std::vector<uint32_t> readers_options;
    for (const uint32_t readers : {grid_cores / num_shards < kTwoReadersCoresPerBank ? 1u : 2u, 2u, most_groups}) {
        if (readers <= most_groups &&
            std::find(readers_options.begin(), readers_options.end(), readers) == readers_options.end()) {
            readers_options.push_back(readers);
        }
    }
    if (row_tiles > 1) {
        readers_options.resize(1);
    }
    const Planner planner{
        .shape = shape,
        .shard_banks = shard_banks,
        .optimal = optimal,
        .grid = grid,
        .cb_budget = cb_budget,
        .partial_tile_bytes = tt::tile_size(partials),
        .layouts = layouts,
        .prepared = prepared};
    for (const uint32_t block_packets : packet_counts) {
        for (const uint32_t readers : readers_options) {
            const uint32_t block_tiles =
                tt::round_down(std::min(block_packets * packet_tiles, kBlockBytes / weight_tile_bytes), kGroupTiles);
            const CoreSplit split = planner.split_cores(readers, block_tiles, block_packets);
            if (split.members.size() > grid_cores) {
                continue;
            }
            auto plan = row_tiles == 1 ? planner.plan_one_row_tile(split) : planner.plan_row_tiles(split, row_tiles);
            if (plan.has_value()) {
                return std::move(*plan);
            }
        }
    }
    return "no core split with " + std::to_string(row_tiles) +
           " row tiles per job fits the circular-buffer budget of " + std::to_string(cb_budget) + " B per core";
}

std::vector<uint32_t> shard_banks_of(const ttnn::Tensor& weights) {
    const auto& mapping = weights.buffer()->get_buffer_page_mapping();
    std::vector<uint32_t> banks;
    banks.reserve(mapping->all_cores.size());
    for (const auto& core : mapping->all_cores) {
        banks.push_back(static_cast<uint32_t>(core.x));
    }
    return banks;
}

GridCoords grid_coords(ttnn::MeshDevice* mesh_device, CoreCoord grid) {
    GridCoords coords;
    for (uint32_t x = 0; x < grid.x; ++x) {
        coords.xs.push_back(mesh_device->worker_core_from_logical_core(CoreCoord(x, 0)).x);
    }
    for (uint32_t y = 0; y < grid.y; ++y) {
        coords.ys.push_back(mesh_device->worker_core_from_logical_core(CoreCoord(0, y)).y);
    }
    for (uint32_t x = 0; x < grid.x; ++x) {
        for (uint32_t y = 0; y < grid.y; ++y) {
            const CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(CoreCoord(x, y));
            TT_FATAL(
                virtual_core.x == coords.xs[x] && virtual_core.y == coords.ys[y],
                "moe_compute expert rows: worker grid is not a product of rows and columns: logical ({}, {}) is "
                "virtual ({}, {}), its column and row give ({}, {})",
                x,
                y,
                virtual_core.x,
                virtual_core.y,
                coords.xs[x],
                coords.ys[y]);
        }
    }
    return coords;
}

std::vector<uint32_t> core_mask(const std::vector<CoreCoord>& cores, CoreCoord grid) {
    std::vector<uint32_t> words(tt::div_up(grid.x * grid.y, 32u), 0);
    for (const auto& core : cores) {
        const uint32_t linear_index = core.x * grid.y + core.y;
        words[linear_index / 32] |= 1u << (linear_index % 32);
    }
    return words;
}

PlaceLayout place_layout(uint32_t tokens, uint32_t top_k, uint32_t local_experts) {
    const uint32_t l1_align = tt::tt_metal::hal::get_l1_alignment();
    PlaceLayout layout;
    layout.table = expert_rows::table_layout(tokens, top_k, local_experts);
    layout.count_words = tt::align(local_experts * kWordBytes, l1_align) / kWordBytes;
    layout.e_t_entry_words = tt::align(2 * kWordBytes, l1_align) / kWordBytes;  // token, k (the ring's e_t entry)
    layout.e_t_page_bytes = (tokens + 1) * layout.e_t_entry_words * kWordBytes;
    layout.activation_row_bytes = tt::align((2 * local_experts + 1) * kWordBytes, l1_align);
    LayoutCursor cursor;
    layout.scratch_e_t = cursor.take(layout.e_t_page_bytes);
    layout.scratch_activation = cursor.take(expert_rows::activation_buffers * layout.activation_row_bytes);
    layout.scratch_count_row = cursor.take(layout.count_words * kWordBytes);
    layout.scratch_table = cursor.take(layout.table.bytes);
    layout.scratch_bytes = cursor.end;
    return layout;
}

uint32_t double_buffer_half_rows(const tt::tt_metal::TensorSpec& dense, uint32_t hidden_size, uint32_t width_shards) {
    const auto& shard = dense.memory_config().shard_spec()->shape;
    const uint32_t segment_width = hidden_size / width_shards;
    return detail::compute_fused_source_buffer_layout(
               shard[0], shard[1], segment_width, shard[0] * shard[1] * kBf16Bytes, segment_width * kBf16Bytes, 2)
        .rows_per_buffer;
}

CombineLayout combine_layout(uint32_t tokens, uint32_t top_k, uint32_t hidden_size, uint32_t senders) {
    constexpr uint32_t kStageBytes = 16 * 1024;  // per staging buffer
    constexpr uint32_t kMaxBatchRows = 8;
    CombineLayout layout;
    layout.sender_rows_max = tt::div_up(tokens * top_k, senders);
    layout.row_stride = align_to_part(hidden_size * kBf16Bytes);
    layout.batch_rows = std::clamp(kStageBytes / layout.row_stride, 1u, kMaxBatchRows);
    const uint32_t slice_bytes =
        align_to_part(layout.sender_rows_max * expert_rows::u16_bytes) + expert_rows::part_alignment;
    LayoutCursor cursor;
    layout.scratch_head = cursor.take(expert_rows::part_alignment);
    layout.scratch_tokens = cursor.take(slice_bytes);
    layout.scratch_k = cursor.take(slice_bytes);
    layout.scratch_order = cursor.take(layout.sender_rows_max * expert_rows::u16_bytes);
    layout.scratch_stage = cursor.take(2 * layout.batch_rows * layout.row_stride);
    layout.bytes = cursor.end;
    return layout;
}

void create_cb(
    tt::tt_metal::ProgramDescriptor& desc,
    const CoreRangeSet& cores,
    uint32_t index,
    uint32_t bytes,
    tt::DataFormat format,
    uint32_t page_bytes) {
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = bytes,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = format,
            .page_size = page_bytes,
        }}},
    });
}

}  // namespace ttnn::experimental::prim::expert_rows_host
