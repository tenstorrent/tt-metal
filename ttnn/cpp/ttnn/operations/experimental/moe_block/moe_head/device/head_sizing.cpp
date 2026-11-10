// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "head_sizing.hpp"
#include "ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/operations/experimental/moe_block/common/kernels/combine_plan_words.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <set>
#include <tuple>

using moe_block_tiles::kBf16TileBytes;
using moe_block_tiles::kFp32TileBytes;

namespace ttnn::operations::experimental::moe_block::moe_head {

namespace head_sizing_detail {

using head_math::align_to_16;
using head_math::align_to_64;
using head_math::ceil_div;
int64_t floor_div(int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    return (numerator % denominator != 0 && ((numerator < 0) != (denominator < 0))) ? quotient - 1 : quotient;
}

[[noreturn]] void fail(const std::string& why) { throw SizingError(why); }

// `total` tiles into `parts` widths, multiples of unit, as even as possible (each >= unit).
std::vector<uint32_t> split_units(uint32_t total, uint32_t parts, uint32_t unit) {
    if (!(total % unit == 0 && total / unit >= parts)) {
        fail("(" + std::to_string(total) + ", " + std::to_string(parts) + ", " + std::to_string(unit) + ")");
    }
    uint32_t base = total / unit / parts, remainder = total / unit % parts;
    std::vector<uint32_t> out(parts);
    for (uint32_t i = 0; i < parts; ++i) {
        out[i] = (base + (i < remainder ? 1 : 0)) * unit;
    }
    return out;
}

std::vector<uint32_t> split_units_padded(uint32_t total, uint32_t parts, uint32_t unit) {
    return total / unit >= parts ? split_units(total, parts, unit) : std::vector<uint32_t>(parts, unit);
}

// `total` into `parts` counts, as even as possible (zeros allowed).
std::vector<uint32_t> split_even(uint32_t total, uint32_t parts) {
    uint32_t base = total / parts, remainder = total % parts;
    std::vector<uint32_t> out(parts);
    for (uint32_t i = 0; i < parts; ++i) {
        out[i] = base + (i < remainder ? 1 : 0);
    }
    return out;
}

std::vector<uint32_t> prefix_sums(const std::vector<uint32_t>& values) {
    std::vector<uint32_t> out(values.size());
    uint32_t sum = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = sum;
        sum += values[i];
    }
    return out;
}

uint32_t max_of(const std::vector<uint32_t>& values) {
    return values.empty() ? 0 : *std::max_element(values.begin(), values.end());
}
uint32_t min_of(const std::vector<uint32_t>& values) {
    return values.empty() ? 0 : *std::min_element(values.begin(), values.end());
}
uint32_t sum_of(const std::vector<uint32_t>& values) { return std::accumulate(values.begin(), values.end(), 0u); }

// Two owner-subset sizes dividing family_steps, covering as many of engine_columns columns as possible (on a tie, the
// first in (first_size, second_size) order).
std::pair<uint32_t, uint32_t> subsets(uint32_t family_steps, uint32_t engine_columns) {
    std::vector<uint32_t> divisors;
    for (uint32_t divisor = 1; divisor <= engine_columns; ++divisor) {
        if (family_steps % divisor == 0) {
            divisors.push_back(divisor);
        }
    }
    auto score = [&](uint32_t first_size, uint32_t second_size) {
        return std::make_pair(
            std::min(first_size + second_size, engine_columns), -std::abs(int(first_size) - int(second_size)));
    };
    std::pair<uint32_t, uint32_t> best{divisors[0], divisors[0]};
    for (uint32_t first_size : divisors) {
        for (uint32_t second_size : divisors) {
            if (score(first_size, second_size) > score(best.first, best.second)) {
                best = {first_size, second_size};
            }
        }
    }
    return best;
}

std::string nested_list_string(const std::vector<std::vector<uint32_t>>& lists) {
    std::string text = "[";
    for (size_t i = 0; i < lists.size(); ++i) {
        text += (i ? ", " : "") + list_string(lists[i]);
    }
    return text + "]";
}

int64_t region(const Regions& regions, const std::string& name) {
    for (const auto& [entry_name, entry_bytes] : regions) {
        if (entry_name == name) {
            return entry_bytes;
        }
    }
    throw SizingError("'" + name + "'");
}

void set_region(Regions& regions, const std::string& name, int64_t bytes) {
    for (auto& [entry_name, entry_bytes] : regions) {
        if (entry_name == name) {
            entry_bytes = bytes;
            return;
        }
    }
    regions.emplace_back(name, bytes);
}

}  // namespace head_sizing_detail

using namespace head_sizing_detail;

std::string format_fixed(double value, int precision) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
    return buffer;
}

std::string list_string(const std::vector<uint32_t>& values) {
    std::string text = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        text += (i ? ", " : "") + std::to_string(values[i]);
    }
    return text + "]";
}

int64_t port_cb_bytes(uint32_t k_step_tiles, uint32_t payload, uint32_t group) {
    int64_t tiles_per_packet = std::max<int64_t>(1, std::min<int64_t>(k_step_tiles, payload / kBf16TileBytes));
    return 2 * int64_t(group) * tiles_per_packet * kBf16TileBytes;
}

int64_t arena_cap_of(int64_t l1_free, int64_t cb_floor) {
    // (the x-gather's port CB is an arena region now, not a static CB under the arena: only other programs' static CBs,
    // cb_floor, stay below it)
    // (the args region past it: kHeadArgsBytes, 64 B aligned)
    return std::min(kArenaMax, floor_div(l1_free - cb_floor - 1024 - kHeadArgsBytes, 64) * 64);
}

std::vector<uint32_t> column_split(uint32_t total, uint32_t parts, uint32_t unit) {
    if (total % unit) {
        fail("(" + std::to_string(total) + ", " + std::to_string(unit) + ")");
    }
    uint32_t base = total / unit / parts, remainder = total / unit % parts;
    std::vector<uint32_t> out(parts);
    for (uint32_t i = 0; i < parts; ++i) {
        out[i] = (base + (i < remainder ? 1 : 0)) * unit;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// plan_transport

ReduceScatterKind reduce_scatter_kind(uint32_t tp_size, bool ring) {
    if (!ring || tp_size == 2) {
        return ReduceScatterKind::Line;
    }
    return tp_size % 2 == 0 ? ReduceScatterKind::RingBalanced : ReduceScatterKind::RingUnbalanced;
}

namespace head_sizing_detail {

using head_math::positive_modulo;

ReduceScatterRole reduce_scatter_role(
    uint32_t position, uint32_t block, uint32_t half_index, uint32_t tp_size, ReduceScatterKind kind) {
    if (block == position) {
        return {ReduceScatterDirection::Final, false, 0};
    }
    if (kind == ReduceScatterKind::Line) {
        if (block > position) {
            return {ReduceScatterDirection::Forward, position == 0, block - position};
        }
        return {ReduceScatterDirection::Backward, position == tp_size - 1, position - block};
    }
    uint32_t forward_distance = positive_modulo(int64_t(block) - position, tp_size),
             backward_distance = tp_size - forward_distance;
    if (kind == ReduceScatterKind::RingUnbalanced) {
        uint32_t forward_blocks = tp_size / 2, backward_blocks = tp_size - 1 - forward_blocks;
        if (forward_distance <= forward_blocks) {
            return {ReduceScatterDirection::Forward, forward_distance == forward_blocks, forward_distance};
        }
        return {ReduceScatterDirection::Backward, backward_distance == backward_blocks, backward_distance};
    }
    uint32_t half_size = tp_size / 2;
    if (half_index == 0) {
        if (forward_distance <= half_size) {
            return {ReduceScatterDirection::Forward, forward_distance == half_size, forward_distance};
        }
        return {ReduceScatterDirection::Backward, backward_distance == half_size - 1, backward_distance};
    }
    if (forward_distance <= half_size - 1) {
        return {ReduceScatterDirection::Forward, forward_distance == half_size - 1, forward_distance};
    }
    return {ReduceScatterDirection::Backward, backward_distance == half_size, backward_distance};
}

}  // namespace head_sizing_detail

ReduceScatterTransport plan_transport(uint32_t tp_size, ReduceScatterKind kind) {
    const bool ring = kind != ReduceScatterKind::Line;
    ReduceScatterTransport out;
    out.chips.resize(tp_size);
    for (uint32_t position = 0; position < tp_size; ++position) {
        auto& chip = out.chips[position];
        for (uint32_t block = 0; block < tp_size; ++block) {
            for (uint32_t half_index = 0; half_index < 2; ++half_index) {
                chip.roles[{block, half_index}] = reduce_scatter_role(position, block, half_index, tp_size, kind);
            }
        }
        // the schedule (no mirror): the other blocks by descending distance (ties forward first, then lower block), own
        // last
        std::vector<uint32_t> others;
        for (uint32_t block = 0; block < tp_size; ++block) {
            if (block != position) {
                others.push_back(block);
            }
        }
        auto order_key = [&](uint32_t block) {
            uint32_t distance = std::max(chip.roles[{block, 0}].distance, chip.roles[{block, 1}].distance);
            return std::make_tuple(
                -int64_t(distance), chip.roles[{block, 0}].direction == ReduceScatterDirection::Forward ? 0 : 1, block);
        };
        std::stable_sort(others.begin(), others.end(), [&](uint32_t first, uint32_t second) {
            return order_key(first) < order_key(second);
        });
        others.push_back(position);
        chip.entries[0] = others;
        chip.entries[1] = others;
    }
    auto upstream = [&](uint32_t position, ReduceScatterDirection direction) -> std::optional<uint32_t> {
        int64_t neighbor = direction == ReduceScatterDirection::Forward ? int64_t(position) - 1 : int64_t(position) + 1;
        if (ring) {
            return uint32_t(positive_modulo(neighbor, tp_size));
        }
        if (neighbor >= 0 && neighbor < tp_size) {
            return uint32_t(neighbor);
        }
        return std::nullopt;
    };
    for (uint32_t position = 0; position < tp_size; ++position) {
        auto& chip = out.chips[position];
        for (uint32_t half_index = 0; half_index < 2; ++half_index) {
            for (ReduceScatterDirection direction :
                 {ReduceScatterDirection::Forward, ReduceScatterDirection::Backward}) {
                auto upstream_position = upstream(position, direction);
                std::vector<uint32_t> arrival_blocks;
                for (uint32_t block : chip.entries[half_index]) {
                    const auto& block_role = chip.roles.at({block, half_index});
                    if (block_role.direction == direction && !block_role.start) {
                        arrival_blocks.push_back(block);
                    } else if (
                        block_role.direction == ReduceScatterDirection::Final && upstream_position &&
                        out.chips[*upstream_position].roles.at({block, half_index}).direction == direction) {
                        arrival_blocks.push_back(block);
                    }
                }
                chip.arrivals[{half_index, direction}] = arrival_blocks;
            }
        }
    }
    for (ReduceScatterDirection direction : {ReduceScatterDirection::Forward, ReduceScatterDirection::Backward}) {
        size_t most_arrivals = 0;
        for (const auto& chip : out.chips) {
            for (uint32_t half_index = 0; half_index < 2; ++half_index) {
                most_arrivals = std::max(most_arrivals, chip.arrivals.at({half_index, direction}).size());
            }
        }
        out.slots[static_cast<int>(direction)] = std::max<uint32_t>(1, most_arrivals);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// ReduceScatterPlan

ReduceScatterPlan::ReduceScatterPlan(
    uint32_t tp_size_,
    bool ring,
    uint32_t engine_columns_,
    uint32_t engine_rows_,
    uint32_t row_tiles_per_core_,
    uint32_t hidden_tiles_,
    int scatter_,
    uint32_t links_,
    uint32_t max_pass_width_,
    bool compact_reduce_scatter_zones_) :
    tp_size(tp_size_),
    engine_columns(engine_columns_),
    engine_rows(engine_rows_),
    row_tiles_per_core(row_tiles_per_core_),
    hidden_tiles(hidden_tiles_),
    links(links_),
    max_pass_width(max_pass_width_),
    chunk_row_tiles(engine_rows_ * row_tiles_per_core_),
    num_cores(engine_columns_ * engine_rows_),
    scatter(scatter_),
    compact_reduce_scatter_zones(compact_reduce_scatter_zones_),
    kind(reduce_scatter_kind(tp_size_, ring)),
    transport(plan_transport(tp_size_, kind)) {
    if (scatter == -1) {
        if (hidden_tiles % tp_size) {
            fail("");
        }
        block_columns = hidden_tiles / tp_size;
        block_column_widths = split_even(block_columns, engine_columns);
        if (min_of(block_column_widths) < 1) {
            fail("");
        }
        block_column_starts = prefix_sums(block_column_widths);
        for (uint32_t width : block_column_widths) {
            passes_per_block = std::max<uint32_t>(passes_per_block, ceil_div(width, max_pass_width));
        }
        core_block_tiles = row_tiles_per_core * max_of(block_column_widths);
    } else {
        if (chunk_row_tiles % tp_size) {
            fail("");
        }
        block_row_tiles = chunk_row_tiles / tp_size;
        column_widths = split_even(hidden_tiles, engine_columns);
        column_starts = prefix_sums(column_widths);
        for (uint32_t width : column_widths) {
            num_passes = std::max<uint32_t>(num_passes, ceil_div(width, max_pass_width));
        }
        core_block_tiles = row_tiles_per_core * max_of(column_widths);
    }
    pass_major = scatter == -2;
    if (scatter == -2 && kind == ReduceScatterKind::RingBalanced && engine_columns > 1) {
        uint32_t best = 0;
        int64_t best_imbalance = -1;
        for (uint32_t i = 0; i + 1 < engine_columns; ++i) {
            int64_t prefix_tiles = 0;
            for (uint32_t column = 0; column <= i; ++column) {
                prefix_tiles += column_widths[column];
            }
            int64_t imbalance = std::abs(2 * prefix_tiles - int64_t(hidden_tiles));
            if (best_imbalance < 0 || imbalance < best_imbalance) {
                best_imbalance = imbalance;
                best = i;
            }
        }
        half_split_column = 1 + best;
    }
}

uint32_t ReduceScatterPlan::half(uint32_t core) const {
    if (half_split_column) {
        return core % engine_columns < *half_split_column ? 0 : 1;
    }
    return core < ceil_div(num_cores, 2) ? 0 : 1;
}

std::pair<uint32_t, uint32_t> ReduceScatterPlan::rows_in(uint32_t core_y, uint32_t block) const {
    uint32_t first_row = 0, row_count = 0;
    for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
        if ((core_y * row_tiles_per_core + row) / block_row_tiles == block) {
            if (row_count == 0) {
                first_row = row;
            }
            ++row_count;
        }
    }
    return {row_count ? first_row : 0, row_count};
}

uint32_t ReduceScatterPlan::count(uint32_t core_x, uint32_t core_y, uint32_t block) const {
    if (scatter == -1) {
        return row_tiles_per_core * block_column_widths[core_x];
    }
    return rows_in(core_y, block).second * column_widths[core_x];
}

const ReduceScatterRole& ReduceScatterPlan::role(uint32_t position, uint32_t block, uint32_t half_index) const {
    return transport.chips[position].roles.at({block, half_index});
}

std::vector<ReduceScatterPlan::Pass> ReduceScatterPlan::column_passes(uint32_t position, uint32_t core_x) const {
    std::vector<Pass> out;
    if (scatter == -1) {
        const auto& chip_entries = entries(position);
        for (uint32_t entry = 0; entry < chip_entries.size(); ++entry) {
            uint32_t first_column = chip_entries[entry] * block_columns + block_column_starts[core_x];
            for (uint32_t width : split_even(block_column_widths[core_x], passes_per_block)) {
                out.push_back({first_column, width, entry});
                first_column += width;
            }
        }
    } else {
        uint32_t first_column = column_starts[core_x];
        for (uint32_t width : split_even(column_widths[core_x], num_passes)) {
            out.push_back({first_column, width, std::nullopt});
            first_column += width;
        }
    }
    return out;
}

uint32_t ReduceScatterPlan::final_block_tiles(uint32_t core_x, uint32_t core_y) const {
    uint32_t tiles = 0;
    for (uint32_t block = 0; block < tp_size; ++block) {
        tiles += count(core_x, core_y, block);
    }
    return tiles;
}

bool ReduceScatterPlan::needs_hold(uint32_t position, uint32_t core_x, uint32_t core_y, uint32_t entry) const {
    uint32_t half_index = half(core_index(core_x, core_y));
    const auto& entry_role = role(position, entries(position, half_index)[entry], half_index);
    return !(entry_role.direction != ReduceScatterDirection::Final && entry_role.start);
}

namespace head_sizing_detail {

uint32_t index_of(const std::vector<uint32_t>& values, uint32_t value) {
    auto found = std::find(values.begin(), values.end(), value);
    if (found == values.end()) {
        fail(std::to_string(value) + " is not in list");
    }
    return found - values.begin();
}

}  // namespace head_sizing_detail

std::pair<std::map<uint32_t, uint32_t>, uint32_t> ReduceScatterPlan::hold_offsets(
    uint32_t position, uint32_t core_x, uint32_t core_y) const {
    const auto core_operations = operations(position, core_x, core_y);
    uint32_t half_index = half(core_index(core_x, core_y));
    const auto& chip_entries = entries(position, half_index);
    const auto passes = column_passes(position, core_x);
    std::vector<std::pair<uint32_t, uint32_t>> first_operation;  // (entry, first op), insertion order
    auto has_first_operation = [&](uint32_t entry) {
        return std::any_of(first_operation.begin(), first_operation.end(), [&](const auto& candidate) {
            return candidate.first == entry;
        });
    };
    for (uint32_t operation_index = 0; operation_index < core_operations.size(); ++operation_index) {
        if (core_operations[operation_index].first != Operation::Pass) {
            continue;
        }
        const auto& pass = passes[core_operations[operation_index].second];
        std::vector<uint32_t> pass_entries;
        if (pass.entry) {
            pass_entries.push_back(*pass.entry);
        } else {
            for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
                pass_entries.push_back(index_of(chip_entries, (core_y * row_tiles_per_core + row) / block_row_tiles));
            }
        }
        for (uint32_t pass_entry : pass_entries) {
            if (!has_first_operation(pass_entry)) {
                first_operation.emplace_back(pass_entry, operation_index);
            }
        }
    }
    std::map<uint32_t, uint32_t> final_operation;
    for (uint32_t operation_index = 0; operation_index < core_operations.size(); ++operation_index) {
        if (core_operations[operation_index].first == Operation::Final) {
            final_operation[core_operations[operation_index].second] = operation_index;
        }
    }
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> live;  // (offset, tiles, entry)
    std::map<uint32_t, uint32_t> offsets;
    uint32_t tiles_used = 0;
    for (uint32_t operation_index = 0; operation_index < core_operations.size(); ++operation_index) {
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> keep;
        for (const auto& live_hold : live) {
            auto final_at = final_operation.find(std::get<2>(live_hold));
            if (final_at == final_operation.end()) {
                fail(std::to_string(std::get<2>(live_hold)));
            }
            if (final_at->second >= operation_index) {
                keep.push_back(live_hold);
            }
        }
        live = keep;
        for (const auto& [entry, first_index] : first_operation) {
            if (first_index == operation_index && needs_hold(position, core_x, core_y, entry) &&
                count(core_x, core_y, chip_entries[entry]) > 0) {
                uint32_t tiles = count(core_x, core_y, chip_entries[entry]);
                uint32_t offset = 0;
                auto sorted_live = live;
                std::sort(sorted_live.begin(), sorted_live.end());
                for (const auto& [live_offset, live_tiles, _] : sorted_live) {
                    if (offset + tiles <= live_offset) {
                        break;
                    }
                    offset = std::max(offset, live_offset + live_tiles);
                }
                live.emplace_back(offset, tiles, entry);
                offsets[entry] = offset;
                tiles_used = std::max(tiles_used, offset + tiles);
            }
        }
    }
    return {offsets, tiles_used};
}

std::optional<uint32_t> ReduceScatterPlan::down(uint32_t position, ReduceScatterDirection direction) const {
    int64_t neighbor = direction == ReduceScatterDirection::Forward ? int64_t(position) + 1 : int64_t(position) - 1;
    if (kind != ReduceScatterKind::Line) {
        return uint32_t(positive_modulo(neighbor, tp_size));
    }
    if (neighbor >= 0 && neighbor < tp_size) {
        return uint32_t(neighbor);
    }
    return std::nullopt;
}

std::optional<uint32_t> ReduceScatterPlan::piece_entry(
    uint32_t position, uint32_t half_index, ReduceScatterDirection direction) const {
    if (!pass_major || !down(position, direction)) {
        return std::nullopt;
    }
    // port_units(position, direction)'s first unit: (entry, half mask, pieces)
    const auto& chip_entries = entries(position, 0);
    for (uint32_t entry = 0; entry < chip_entries.size(); ++entry) {
        uint32_t block = chip_entries[entry], mask = 0;
        for (uint32_t each_half = 0; each_half < 2; ++each_half) {
            if (role(position, block, each_half).direction == direction) {
                mask |= 1u << each_half;
            }
        }
        if (!mask) {
            continue;
        }
        bool pieces = true;
        for (uint32_t each_half = 0; each_half < 2; ++each_half) {
            if ((mask >> each_half) & 1) {
                pieces = pieces && role(position, block, each_half).start;
            }
        }
        if (!pieces || !((mask >> half_index) & 1)) {
            return std::nullopt;
        }
        return entry;
    }
    return std::nullopt;
}

std::vector<std::pair<ReduceScatterPlan::Operation, uint32_t>> ReduceScatterPlan::operations(
    uint32_t position, uint32_t core_x, uint32_t core_y) const {
    uint32_t half_index = half(core_index(core_x, core_y));
    const auto& chip_entries = entries(position, half_index);
    const auto passes = column_passes(position, core_x);
    std::map<uint32_t, uint32_t> last_pass;
    for (uint32_t i = 0; i < passes.size(); ++i) {
        if (passes[i].entry) {
            last_pass[*passes[i].entry] = i;
        } else {
            for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
                last_pass[index_of(chip_entries, (core_y * row_tiles_per_core + row) / block_row_tiles)] = i;
            }
        }
    }
    std::vector<uint32_t> piece_entries;
    for (ReduceScatterDirection direction : {ReduceScatterDirection::Forward, ReduceScatterDirection::Backward}) {
        auto piece = piece_entry(position, half_index, direction);
        if (piece && count(core_x, core_y, chip_entries[*piece])) {
            piece_entries.push_back(*piece);
        }
    }
    std::vector<std::pair<Operation, uint32_t>> out;
    std::vector<uint32_t> pending;
    for (uint32_t i = 0; i < passes.size(); ++i) {
        out.emplace_back(Operation::Pass, i);
        for (uint32_t entry : piece_entries) {
            out.emplace_back(Operation::Handoff, entry);
        }
        std::vector<uint32_t> finished;
        for (const auto& [entry, last_index] : last_pass) {
            if (last_index == i) {
                finished.push_back(entry);  // (std::map: sorted)
            }
        }
        if (!pending.empty() && !finished.empty()) {
            for (uint32_t entry : pending) {
                out.emplace_back(Operation::Final, entry);
            }
            pending.clear();
        }
        for (uint32_t entry : finished) {
            uint32_t block = chip_entries[entry];
            if (count(core_x, core_y, block) == 0 ||
                std::find(piece_entries.begin(), piece_entries.end(), entry) != piece_entries.end()) {
                continue;
            }
            const auto& entry_role = role(position, block, half_index);
            if (entry_role.direction != ReduceScatterDirection::Final && entry_role.start) {
                out.emplace_back(Operation::Handoff, entry);
            } else {
                pending.push_back(entry);
            }
        }
    }
    for (uint32_t entry : pending) {
        out.emplace_back(Operation::Final, entry);
    }
    return out;
}

std::vector<std::pair<uint32_t, ReduceScatterDirection>> ReduceScatterPlan::handoffs(
    uint32_t position, uint32_t core_x, uint32_t core_y) const {
    uint32_t half_index = half(core_index(core_x, core_y));
    const auto& chip_entries = entries(position, half_index);
    std::vector<std::pair<uint32_t, ReduceScatterDirection>> out;
    for (const auto& [operation, entry] : operations(position, core_x, core_y)) {
        if (operation == Operation::Pass) {
            continue;
        }
        ReduceScatterDirection direction = role(position, chip_entries[entry], half_index).direction;
        if (operation == Operation::Handoff ||
            (operation == Operation::Final && direction != ReduceScatterDirection::Final)) {
            out.emplace_back(entry, direction);
        }
    }
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> ReduceScatterPlan::zone_slots(
    uint32_t position, uint32_t core_x, uint32_t core_y, ReduceScatterDirection direction) const {
    const auto& chip_arrivals = transport.chips[position].arrivals;
    auto found = chip_arrivals.find({half(core_index(core_x, core_y)), direction});
    std::vector<std::pair<uint32_t, uint32_t>> out;
    if (found == chip_arrivals.end()) {
        return out;
    }
    const auto& arrival_blocks = found->second;
    if (!compact_reduce_scatter_zones) {
        for (uint32_t slot = 0; slot < arrival_blocks.size(); ++slot) {
            out.emplace_back(slot * core_block_tiles, core_block_tiles);
        }
        return out;
    }
    uint32_t offset = 0;
    for (uint32_t block : arrival_blocks) {
        uint32_t tiles = count(core_x, core_y, block);
        out.emplace_back(offset, tiles);
        offset += tiles;
    }
    return out;
}

uint32_t ReduceScatterPlan::zone_tiles(ReduceScatterDirection direction) const {
    if (!compact_reduce_scatter_zones) {
        return transport.slots[static_cast<int>(direction)] * core_block_tiles;
    }
    uint32_t most_tiles = 0;
    for (uint32_t position = 0; position < tp_size; ++position) {
        for (uint32_t core_y = 0; core_y < engine_rows; ++core_y) {
            for (uint32_t core_x = 0; core_x < engine_columns; ++core_x) {
                uint32_t tiles = 0;
                for (const auto& [_, slot_tiles] : zone_slots(position, core_x, core_y, direction)) {
                    tiles += slot_tiles;
                }
                most_tiles = std::max(most_tiles, tiles);
            }
        }
    }
    return most_tiles;
}

std::optional<uint32_t> ReduceScatterPlan::up(uint32_t position, ReduceScatterDirection direction) const {
    int64_t neighbor = direction == ReduceScatterDirection::Forward ? int64_t(position) - 1 : int64_t(position) + 1;
    if (kind != ReduceScatterKind::Line) {
        return uint32_t(positive_modulo(neighbor, tp_size));
    }
    if (neighbor >= 0 && neighbor < tp_size) {
        return uint32_t(neighbor);
    }
    return std::nullopt;
}

std::array<std::vector<ReduceScatterPlan::Arrival>, 2> ReduceScatterPlan::arrivals(
    uint32_t position, uint32_t core_x, uint32_t core_y) const {
    const uint32_t half_index = half(core_index(core_x, core_y));
    const auto& chip = transport.chips[position];
    const auto& chip_entries = chip.entries[half_index];
    std::array<std::vector<Arrival>, 2> out;
    for (ReduceScatterDirection direction : {ReduceScatterDirection::Forward, ReduceScatterDirection::Backward}) {
        const auto& arrival_blocks = chip.arrivals.at({half_index, direction});
        if (arrival_blocks.empty()) {
            continue;
        }
        const uint32_t upstream_position = *up(position, direction);
        std::vector<uint32_t> upstream_blocks;
        for (uint32_t block : transport.chips[upstream_position].entries[half_index]) {
            if (transport.chips[upstream_position].roles.at({block, half_index}).direction == direction &&
                count(core_x, core_y, block) > 0) {
                upstream_blocks.push_back(block);
            }
        }
        for (uint32_t slot = 0; slot < arrival_blocks.size(); ++slot) {
            const uint32_t block = arrival_blocks[slot];
            out[static_cast<int>(direction)].push_back(
                {index_of(chip_entries, block),
                 slot,
                 count(core_x, core_y, block) > 0 ? index_of(upstream_blocks, block) + 1 : 0});
        }
    }
    return out;
}

std::vector<ReduceScatterPlan::PortUnit> ReduceScatterPlan::port_units(
    uint32_t position, ReduceScatterDirection direction) const {
    const auto& chip = transport.chips[position];
    const auto downstream_position = down(position, direction);
    std::vector<PortUnit> out;
    const auto& chip_entries = chip.entries[0];
    for (uint32_t entry = 0; entry < chip_entries.size(); ++entry) {
        const uint32_t block = chip_entries[entry];
        uint32_t mask = 0;
        for (uint32_t half_index = 0; half_index < 2; ++half_index) {
            if (role(position, block, half_index).direction == direction) {
                mask |= 1u << half_index;
            }
        }
        if (!mask) {
            continue;
        }
        uint32_t slots[2] = {0, 0};
        for (uint32_t half_index = 0; half_index < 2; ++half_index) {
            if ((mask >> half_index) & 1) {
                slots[half_index] =
                    index_of(transport.chips[*downstream_position].arrivals.at({half_index, direction}), block);
            }
        }
        bool pieces = pass_major && out.empty();
        for (uint32_t half_index = 0; half_index < 2; ++half_index) {
            if ((mask >> half_index) & 1) {
                pieces = pieces && role(position, block, half_index).start;
            }
        }
        out.push_back({entry, block, mask, slots[0], slots[1], pieces});
    }
    return out;
}

std::vector<ReduceScatterDirection> ReduceScatterPlan::final_block_readers(
    uint32_t position, uint32_t core_x, uint32_t core_y) const {
    const uint32_t half_index = half(core_index(core_x, core_y));
    std::vector<ReduceScatterDirection> out;
    for (ReduceScatterDirection direction : {ReduceScatterDirection::Forward, ReduceScatterDirection::Backward}) {
        if (!down(position, direction)) {
            continue;
        }
        for (const auto& unit : port_units(position, direction)) {
            if (((unit.mask >> half_index) & 1) && count(core_x, core_y, unit.block) > 0) {
                out.push_back(direction);
                break;
            }
        }
    }
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> ReduceScatterPlan::port_cores(uint32_t link_index) const {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (uint32_t core_y = 0; core_y < engine_rows; ++core_y) {
        for (uint32_t core_x = 0; core_x < engine_columns; ++core_x) {
            if (link(core_index(core_x, core_y)) == link_index) {
                out.emplace_back(core_x, core_y);
            }
        }
    }
    return out;
}

std::tuple<uint32_t, uint32_t, uint32_t> ReduceScatterPlan::final_map(
    uint32_t position, uint32_t core_x, uint32_t core_y) const {
    if (scatter == -1) {
        return {
            core_y * row_tiles_per_core * block_columns + block_column_starts[core_x],
            block_columns,
            block_column_widths[core_x]};
    }
    const auto [first_row, row_count] = rows_in(core_y, position);
    if (row_count == 0) {
        return {0, hidden_tiles, 1};
    }
    return {
        (core_y * row_tiles_per_core + first_row - position * block_row_tiles) * hidden_tiles + column_starts[core_x],
        hidden_tiles,
        column_widths[core_x]};
}

uint32_t ReduceScatterPlan::prefix_mask(
    uint32_t position, uint32_t half_index, ReduceScatterDirection direction, uint32_t slot) const {
    const auto& chip_arrivals = transport.chips[position].arrivals;
    auto found = chip_arrivals.find({half_index, direction});
    uint32_t mask = 0;
    if (found != chip_arrivals.end()) {
        for (uint32_t i = 0; i < slot && i < found->second.size(); ++i) {
            mask |= 1u << found->second[i];
        }
    }
    return mask;
}

uint32_t ReduceScatterPlan::counts_word(uint32_t core_x, uint32_t core_y) const {
    if (core_block_tiles >= 256) {
        fail(
            "moe_head reduce-scatter: a core's sub-block of " + std::to_string(core_block_tiles) +
            " tiles > 255 (the counts word's bytes)");
    }
    if (tp_size <= 4) {
        uint32_t word = 0;
        for (uint32_t block = 0; block < tp_size; ++block) {
            word += count(core_x, core_y, block) << (8 * block);
        }
        return word;
    }
    std::vector<std::pair<uint32_t, uint32_t>> nonzero_blocks;
    for (uint32_t block = 0; block < tp_size; ++block) {
        if (count(core_x, core_y, block)) {
            nonzero_blocks.emplace_back(block, count(core_x, core_y, block));
        }
    }
    if (!(scatter == -2 && nonzero_blocks.size() <= 2)) {
        fail("moe_head reduce-scatter: TP > 4 needs the rows scatter with a core's rows in at most two blocks");
    }
    while (nonzero_blocks.size() < 2) {
        nonzero_blocks.emplace_back(0xFF, 0);
    }
    return nonzero_blocks[0].first | nonzero_blocks[0].second << 8 | nonzero_blocks[1].first << 16 |
           nonzero_blocks[1].second << 24;
}

uint32_t ReduceScatterPlan::final_block_offset(
    uint32_t position, uint32_t core_x, uint32_t core_y, uint32_t entry) const {
    const auto& chip_entries = entries(position, half(core_index(core_x, core_y)));
    uint32_t offset = 0;
    for (uint32_t i = 0; i < entry; ++i) {
        offset += count(core_x, core_y, chip_entries[i]);
    }
    return offset;
}

std::vector<uint32_t> ReduceScatterPlan::pass_widths(uint32_t core_x) const {
    std::vector<uint32_t> out;
    for (const auto& pass : column_passes(0, core_x)) {
        out.push_back(pass.width);
    }
    return out;
}

std::vector<std::pair<uint32_t, std::vector<uint32_t>>> ReduceScatterPlan::pass_bases(
    uint32_t position, uint32_t core_x, uint32_t core_y) const {
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> out;
    const auto& chip_entries = entries(position, half(core_index(core_x, core_y)));
    const auto hold = hold_offsets(position, core_x, core_y).first;
    auto base_of = [&](uint32_t entry, uint32_t within) -> uint32_t {
        auto found = hold.find(entry);
        if (found != hold.end()) {
            return (found->second + within) | (1u << 31);
        }
        return final_block_offset(position, core_x, core_y, entry) + within;
    };
    for (const auto& pass : column_passes(position, core_x)) {
        std::vector<uint32_t> bases;
        if (scatter == -1) {
            const uint32_t entry = *pass.entry, block = chip_entries[entry];
            const uint32_t offset_in_block = pass.first_column - block * block_columns - block_column_starts[core_x];
            for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
                bases.push_back(base_of(entry, row * block_column_widths[core_x] + offset_in_block));
            }
        } else {
            for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
                const uint32_t block = (core_y * row_tiles_per_core + row) / block_row_tiles;
                const auto [first_row, row_count] = rows_in(core_y, block);
                if (pass_major) {
                    bases.push_back(base_of(
                        index_of(chip_entries, block),
                        row_count * (pass.first_column - column_starts[core_x]) + (row - first_row) * pass.width));
                } else {
                    bases.push_back(base_of(
                        index_of(chip_entries, block),
                        (row - first_row) * column_widths[core_x] + pass.first_column - column_starts[core_x]));
                }
            }
        }
        out.emplace_back(pass.width, bases);
    }
    return out;
}

std::vector<uint32_t> ReduceScatterPlan::compute_args(uint32_t position, uint32_t core_x, uint32_t core_y) const {
    const auto core_operations = operations(position, core_x, core_y);
    std::vector<uint32_t> out = {uint32_t(core_operations.size())};
    for (const auto& [operation, index] : core_operations) {
        out.push_back(static_cast<uint32_t>(operation) | (index << 8));
    }
    for (const auto& [width, bases] : pass_bases(position, core_x, core_y)) {
        out.push_back(width);
        out.insert(out.end(), bases.begin(), bases.end());
    }
    const uint32_t half_index = half(core_index(core_x, core_y));
    const auto& chip_entries = entries(position, half_index);
    const auto hold = hold_offsets(position, core_x, core_y).first;
    const auto core_arrivals = arrivals(position, core_x, core_y);
    const auto forward_zones = zone_slots(position, core_x, core_y, ReduceScatterDirection::Forward),
               backward_zones = zone_slots(position, core_x, core_y, ReduceScatterDirection::Backward);
    for (uint32_t entry = 0; entry < chip_entries.size(); ++entry) {
        const uint32_t block = chip_entries[entry];
        const ReduceScatterDirection direction = role(position, block, half_index).direction;
        std::optional<uint32_t> forward_slot, backward_slot;
        for (const auto& arrival : core_arrivals[0]) {
            if (arrival.entry == entry && !forward_slot) {
                forward_slot = arrival.slot;
            }
        }
        for (const auto& arrival : core_arrivals[1]) {
            if (arrival.entry == entry && !backward_slot) {
                backward_slot = arrival.slot;
            }
        }
        const uint32_t mode =
            (forward_slot ? 1 : 0) | (backward_slot ? 2 : 0) | (direction == ReduceScatterDirection::Final ? 4 : 0);
        auto found = hold.find(entry);
        out.push_back(count(core_x, core_y, block));
        out.push_back(mode);
        out.push_back(forward_slot ? forward_zones[*forward_slot].first : 0);
        out.push_back(backward_slot ? backward_zones[*backward_slot].first : 0);
        out.push_back(found != hold.end() ? found->second : 0);
        out.push_back(final_block_offset(position, core_x, core_y, entry));
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// SharedExpertSizing

SharedExpertSizing::SharedExpertSizing(const SharedExpertArgs& sizing_args) : args(sizing_args) {
    engine_columns = sizing_args.engine_columns;
    engine_rows = sizing_args.engine_rows;
    row_tiles_per_core = sizing_args.row_tiles_per_core;
    chunk_row_tiles = engine_rows * row_tiles_per_core;
    rows = sizing_args.rows;
    hidden_dim = sizing_args.hidden_dim;
    shared_expert_dim = sizing_args.shared_expert_dim;
    shards = sizing_args.shards;
    if (sizing_args.rows % (32 * chunk_row_tiles) != 0) {
        fail(
            "moe_head shared expert: " + std::to_string(sizing_args.rows) +
            " rows are not whole chunks of chunk_row_tiles = engine_rows " + std::to_string(engine_rows) +
            " x row_tiles_per_core " + std::to_string(row_tiles_per_core) + " = " + std::to_string(chunk_row_tiles) +
            " row tiles");
    }
    row_chunks = sizing_args.rows / 32 / chunk_row_tiles;
    hidden_tiles = sizing_args.hidden_dim / 32;
    shared_expert_tiles = sizing_args.shared_expert_dim / 32;
    k_step_tiles = sizing_args.k_step_tiles;
    num_k_steps = hidden_tiles / k_step_tiles;
    if (!(hidden_tiles % k_step_tiles == 0 && num_k_steps % 2 == 0)) {
        fail(
            "moe_head shared expert: K step " + std::to_string(k_step_tiles) + " must divide hidden_dim's " +
            std::to_string(hidden_tiles) + " tiles into an even count of steps");
    }
    const int64_t arena_max = sizing_args.arena_max;
    if (!(num_k_steps % sizing_args.shards == 0 && (num_k_steps / sizing_args.shards) % 2 == 0)) {
        fail("even K steps a shard");
    }
    k_steps_per_shard = num_k_steps / sizing_args.shards;
    const uint32_t k_step_pairs = num_k_steps / 2;
    std::tie(family0_size, family1_size) = subsets(k_steps_per_shard / 2, engine_columns);
    for (uint32_t column = 0; column < family0_size; ++column) {
        family0_columns.push_back(column);
    }
    for (uint32_t column = engine_columns - family1_size; column < engine_columns; ++column) {
        family1_columns.push_back(column);
    }
    const auto pairs = column_split(shared_expert_tiles, engine_columns, 1);
    if (min_of(pairs) < 1) {
        fail(
            "moe_head shared expert: shared_expert_dim's " + std::to_string(shared_expert_tiles) +
            " tiles leave a column of the " + std::to_string(engine_columns) + " without a gate / up pair");
    }
    column_pairs = pairs;
    column_first_pair = prefix_sums(pairs);
    max_column_pairs = max_of(pairs);
    const uint32_t dest_tiles = kDestTiles;
    gate_up_pass_width = 8 / row_tiles_per_core;
    const uint32_t even_gate_up_pass_width = gate_up_pass_width / 2 * 2;
    gate_up_passes = 0;
    for (uint32_t pair_count : pairs) {
        gate_up_passes = std::max<uint32_t>(gate_up_passes, ceil_div(2 * pair_count, even_gate_up_pass_width));
    }
    for (uint32_t pair_count : pairs) {
        gate_up_pass_widths.push_back(split_units_padded(2 * pair_count, gate_up_passes, 2));
    }
    gate_up_passes_without_latent = gate_up_passes;
    latent_tiles = sizing_args.latent.value_or(0);
    if (sizing_args.latent) {
        latent_column_tiles = column_split(*sizing_args.latent, engine_columns, 1);
        latent_column_first_tile = prefix_sums(latent_column_tiles);
        if (min_of(latent_column_tiles) < 1) {
            fail(
                "moe_head shared expert: the latent's " + std::to_string(*sizing_args.latent) +
                " tiles leave a column of the " + std::to_string(engine_columns) + " without one");
        }
        max_latent_column_tiles = max_of(latent_column_tiles);
        const uint32_t latent_passes = ceil_div(max_latent_column_tiles, gate_up_pass_width);
        for (uint32_t column = 0; column < engine_columns; ++column) {
            auto extra = split_units(latent_column_tiles[column], latent_passes, 1);
            gate_up_pass_widths[column].insert(gate_up_pass_widths[column].end(), extra.begin(), extra.end());
        }
        gate_up_passes += latent_passes;
    }
    latent_pages = sizing_args.latent
                       ? int64_t(row_chunks > 1 ? sizing_args.shards : 1) * row_tiles_per_core * max_latent_column_tiles
                       : 0;
    const auto down_column_tiles = column_split(hidden_tiles, engine_columns, 1);
    max_down_pass_width = dest_tiles / row_tiles_per_core;
    down_passes = 0;
    for (uint32_t column_tiles : down_column_tiles) {
        down_passes = std::max<uint32_t>(down_passes, ceil_div(column_tiles, max_down_pass_width));
    }
    for (uint32_t column_tiles : down_column_tiles) {
        down_pass_widths.push_back(split_units(column_tiles, down_passes, 1));
    }
    const ReduceScatterPlan* reduce_scatter = sizing_args.reduce_scatter;
    if (reduce_scatter == nullptr) {
        fail("moe_head shared expert: no reduce-scatter plan");
    }
    {
        if (!(reduce_scatter->max_pass_width == max_down_pass_width &&
              reduce_scatter->row_tiles_per_core == row_tiles_per_core &&
              reduce_scatter->engine_columns == engine_columns && reduce_scatter->engine_rows == engine_rows)) {
            fail(
                "moe_head shared expert: the RS plan's tiling (engine_columns " +
                std::to_string(reduce_scatter->engine_columns) + " engine_rows " +
                std::to_string(reduce_scatter->engine_rows) + " row_tiles_per_core " +
                std::to_string(reduce_scatter->row_tiles_per_core) + " pwmax " +
                std::to_string(reduce_scatter->max_pass_width) + ") is not the matmul cores' (engine_columns " +
                std::to_string(engine_columns) + " engine_rows " + std::to_string(engine_rows) +
                " row_tiles_per_core " + std::to_string(row_tiles_per_core) + " pwmax " +
                std::to_string(max_down_pass_width) + ")");
        }
        down_pass_widths.clear();
        for (uint32_t column = 0; column < engine_columns; ++column) {
            std::vector<uint32_t> widths;
            for (const auto& pass : reduce_scatter->column_passes(0, column)) {
                widths.push_back(pass.width);
            }
            down_pass_widths.push_back(widths);
        }
        down_passes = down_pass_widths[0].size();
        for (const auto& widths : down_pass_widths) {
            if (widths.size() != down_passes) {
                fail("moe_head shared expert: uneven down passes " + nested_list_string(down_pass_widths));
            }
        }
    }
    down_k_step_tiles = 1;
    for (uint32_t divisor = 1; divisor <= 8; ++divisor) {
        if (shared_expert_tiles % divisor == 0) {
            down_k_step_tiles = divisor;
        }
    }
    down_k_steps = shared_expert_tiles / down_k_step_tiles;
    w_blocks_per_chunk = gate_up_passes * num_k_steps + down_passes * down_k_steps;
    if (w_blocks_per_chunk > 1024) {
        fail(
            "moe_head shared expert: " + std::to_string(w_blocks_per_chunk) +
            " W rounds a chunk > 1024 (gate_up_passes " + std::to_string(gate_up_passes) + " x num_k_steps " +
            std::to_string(num_k_steps) + " + down_passes " + std::to_string(down_passes) + " x down_k_steps " +
            std::to_string(down_k_steps) + "): a larger K step than " + std::to_string(k_step_tiles));
    }
    uint32_t max_gate_up_width = 0, max_down_width = 0;
    for (const auto& widths : gate_up_pass_widths) {
        max_gate_up_width = std::max(max_gate_up_width, max_of(widths));
    }
    for (const auto& widths : down_pass_widths) {
        max_down_width = std::max(max_down_width, max_of(widths));
    }
    w_slot_tiles = std::max(k_step_tiles * max_gate_up_width, down_k_step_tiles * max_down_width);
    switch (sizing_args.w_dtype) {
        case WeightDtype::Bf8: w_tile_bytes = tt::tile_size(tt::DataFormat::Bfp8_b); break;
        case WeightDtype::Bf4: w_tile_bytes = tt::tile_size(tt::DataFormat::Bfp4_b); break;
        case WeightDtype::Bf16: w_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b); break;
    }
    w_own_tiles = 0;
    for (uint32_t column = 0; column < engine_columns; ++column) {
        for (uint32_t core_y = 0; core_y < engine_rows; ++core_y) {
            int64_t sum = 0;
            for (uint32_t block = core_y; block < w_blocks_per_chunk; block += engine_rows) {
                sum += w_block_tiles(column, block);
            }
            w_own_tiles = std::max(w_own_tiles, sum);
        }
    }
    uint32_t owned_steps_per_chunk = 0;
    for (uint32_t column = 0; column < engine_columns; ++column) {
        owned_steps_per_chunk = std::max(
            owned_steps_per_chunk,
            (in_family0(column) ? k_step_pairs / family0_size : 0) +
                (in_family1(column) ? k_step_pairs / family1_size : 0));
    }
    x_own_tiles = int64_t(row_chunks) * owned_steps_per_chunk * row_tiles_per_core *
                  k_step_tiles;  // (every row chunk's x own resident)
    prep_full_column_tiles = sizing_args.latent ? max_latent_column_tiles : k_step_tiles;
    for (uint32_t divisor = 1; divisor <= kDestTiles; ++divisor) {
        if (prep_full_column_tiles % divisor == 0) {
            prep_block_column_tiles = divisor;
        }
    }

    // the reduce-scatter's regions: the same at every W ring / h slot count
    Regions reduce_scatter_regions;
    {
        uint32_t max_hold_tiles = 0, max_final_block_tiles = 0;
        size_t max_handoffs = 0;
        for (uint32_t position = 0; position < reduce_scatter->tp_size; ++position) {
            for (uint32_t core_x = 0; core_x < engine_columns; ++core_x) {
                for (uint32_t core_y = 0; core_y < engine_rows; ++core_y) {
                    max_hold_tiles =
                        std::max(max_hold_tiles, reduce_scatter->hold_offsets(position, core_x, core_y).second);
                    max_handoffs = std::max(max_handoffs, reduce_scatter->handoffs(position, core_x, core_y).size());
                }
            }
        }
        for (uint32_t core_x = 0; core_x < engine_columns; ++core_x) {
            for (uint32_t core_y = 0; core_y < engine_rows; ++core_y) {
                max_final_block_tiles =
                    std::max(max_final_block_tiles, reduce_scatter->final_block_tiles(core_x, core_y));
            }
        }
        reduce_scatter_regions = {
            {"hold", int64_t(std::max(1u, max_hold_tiles)) * kFp32TileBytes},
            {"finb", int64_t(std::max(1u, max_final_block_tiles)) * kBf16TileBytes},
            {"arrf", int64_t(reduce_scatter->zone_tiles(ReduceScatterDirection::Forward)) * kBf16TileBytes},
            {"arrb", int64_t(reduce_scatter->zone_tiles(ReduceScatterDirection::Backward)) * kBf16TileBytes},
            {"outf", 8 * kBf16TileBytes},
            {"finc", 16 * int64_t(std::max<size_t>(8, max_handoffs))},
            {"psync", 16 * int64_t(std::max<uint32_t>(8, down_passes))},
        };
        if (row_chunks > 1) {
            reduce_scatter_regions.emplace_back("ffree", 16 * int64_t(std::max<uint32_t>(2, row_chunks - 1)));
        }
    }
    reduce_scatter_sizes = reduce_scatter_regions;
    const std::vector<std::pair<uint32_t, uint32_t>> attempts =
        row_chunks == 1 ? std::vector<std::pair<uint32_t, uint32_t>>{{0, 1}, {5, 1}, {4, 1}, {3, 1}}
                        : std::vector<std::pair<uint32_t, uint32_t>>{{0, 2}, {5, 2}, {5, 1}, {4, 1}, {3, 1}};
    int64_t arena_end = 0;
    for (const auto& [ring_slots, h_slot_count] : attempts) {
        w_ring_slots = ring_slots;
        h_slots = h_slot_count;
        uint32_t busiest = 0;
        for (const auto& widths : gate_up_pass_widths) {
            busiest = std::max(busiest, sum_of(widths));
        }
        accumulator_pages = int64_t(row_tiles_per_core) * busiest;
        offsets.clear();
        arena_end = 0;
        Regions reduce_scatter_sequential = reduce_scatter_regions;
        Regions low_overlays, hold_overlay;
        if (!reduce_scatter_sequential.empty() && row_chunks == 1) {
            low_overlays.assign(reduce_scatter_sequential.begin(), reduce_scatter_sequential.begin() + 2);
            reduce_scatter_sequential.erase(reduce_scatter_sequential.begin(), reduce_scatter_sequential.begin() + 2);
        }
        if (!reduce_scatter_sequential.empty() && row_chunks > 1) {
            hold_overlay.assign(reduce_scatter_sequential.begin(), reduce_scatter_sequential.begin() + 1);
            reduce_scatter_sequential.erase(reduce_scatter_sequential.begin(), reduce_scatter_sequential.begin() + 1);
        }
        Regions sequential = {
            {"xown", x_own_tiles * kBf16TileBytes},
            {"wown", (w_ring_slots ? int64_t(w_ring_slots) * w_slot_tiles : w_own_tiles) * w_tile_bytes},
            {"win", int64_t(3) * w_slot_tiles * w_tile_bytes},  // (kWSlots W slots)
            {"hown", int64_t(row_tiles_per_core) * max_column_pairs * kBf16TileBytes},
            {"hall", int64_t(h_slots) * engine_columns * row_tiles_per_core * max_column_pairs * kBf16TileBytes},
            {"xin", int64_t(2) * row_tiles_per_core * k_step_tiles * kBf16TileBytes},
            {"acc", accumulator_pages * kFp32TileBytes},
            {"words", 256},
        };
        sequential.insert(sequential.end(), reduce_scatter_sequential.begin(), reduce_scatter_sequential.end());
        if (sizing_args.latent) {
            sequential.emplace_back("lat", int64_t(row_tiles_per_core) * max_latent_column_tiles * kBf16TileBytes);
            sequential.emplace_back(
                "latz",
                int64_t(sizing_args.shards - 1) * row_tiles_per_core * max_latent_column_tiles * kBf16TileBytes +
                    int64_t(row_chunks - 1) * latent_pages * kBf16TileBytes);
        }
        sequential.emplace_back("po", int64_t(2) * prep_full_column_tiles * kBf16TileBytes);
        if (sizing_args.latent) {
            sequential.emplace_back("lrdy", 16 * int64_t(std::max<uint32_t>(4, row_chunks)));
        }
        for (const auto& [name, bytes] : sequential) {
            set_region(offsets, name, arena_end);
            arena_end = align_to_64(arena_end + bytes);
        }
        {
            std::set<int64_t> region_ends{arena_end};
            for (const auto& [_, offset] : offsets) {
                region_ends.insert(offset);
            }
            sequential_ends.assign(region_ends.begin(), region_ends.end());
        }
        if (!low_overlays.empty()) {
            // (the room from xin up to the words, the region after acc)
            const int64_t room = region(offsets, "words") - region(offsets, "xin");
            const int64_t hold_bytes = region(low_overlays, "hold"), final_block_bytes = region(low_overlays, "finb");
            if (align_to_64(hold_bytes) + final_block_bytes <= room) {
                set_region(offsets, "hold", region(offsets, "xin"));
                set_region(offsets, "finb", region(offsets, "xin") + align_to_64(hold_bytes));
            } else {
                for (const auto& [name, bytes] : low_overlays) {
                    set_region(offsets, name, arena_end);
                    arena_end = align_to_64(arena_end + bytes);
                }
            }
        }
        if (!hold_overlay.empty()) {
            const int64_t hold_bytes = region(hold_overlay, "hold");
            if (hold_bytes <= accumulator_pages * kFp32TileBytes) {
                set_region(offsets, "hold", region(offsets, "acc"));
            } else {
                set_region(offsets, "hold", arena_end);
                arena_end = align_to_64(arena_end + hold_bytes);
            }
        }
        {  // the router logits on the engine
            const auto [expert_tiles, router_k_tiles] = sizing_args.router;
            router_expert_tiles_per_column = ceil_div(expert_tiles, engine_columns);
            if (sizing_args.router_iterations == 0 || row_chunks % sizing_args.router_iterations) {
                fail(
                    "router iterations " + std::to_string(sizing_args.router_iterations) +
                    " do not divide the row chunks");
            }
            const uint32_t router_row_tiles = row_tiles_per_core * row_chunks /
                                              sizing_args.router_iterations;  // (a core's rows of one router iteration)
            router_row_tiles_per_group =
                router_expert_tiles_per_column * router_row_tiles <= dest_tiles
                    ? router_row_tiles
                    : (router_expert_tiles_per_column * row_tiles_per_core <= dest_tiles ? row_tiles_per_core : 1);
            if (router_expert_tiles_per_column * router_row_tiles_per_group > dest_tiles) {
                fail("the router logits on the engine: router_expert_tiles_per_column x its row group in DEST");
            }
            int64_t zone_growth = 0;
            if (reduce_scatter->compact_reduce_scatter_zones) {
                for (ReduceScatterDirection direction :
                     {ReduceScatterDirection::Forward, ReduceScatterDirection::Backward}) {
                    zone_growth += (int64_t(reduce_scatter->transport.slots[static_cast<int>(direction)]) *
                                        reduce_scatter->core_block_tiles -
                                    reduce_scatter->zone_tiles(direction)) *
                                   kBf16TileBytes;
                }
            }
            int64_t router_start =
                reduce_scatter_sequential.empty() ? arena_end : region(offsets, reduce_scatter_sequential[0].first);
            std::vector<uint32_t> k_chunk_candidates;
            for (uint32_t divisor = router_k_tiles; divisor >= 1; --divisor) {
                if (router_k_tiles % divisor == 0 && router_k_tiles / divisor >= 2) {
                    k_chunk_candidates.push_back(divisor);
                }
            }
            if (k_chunk_candidates.empty()) {
                fail("moe_head shared expert: no router K chunk");
            }
            Regions router_regions;
            int64_t need = 0;
            const int64_t limit = std::min(arena_end + zone_growth, std::max(arena_end, arena_max));
            for (uint32_t k_chunk : k_chunk_candidates) {
                router_k_chunk_tiles = k_chunk;
                router_regions = {
                    {"rx", int64_t(2) * router_k_chunk_tiles * router_row_tiles * kBf16TileBytes},
                    {"rw", int64_t(2) * router_expert_tiles_per_column * router_k_chunk_tiles * kBf16TileBytes},
                    {"ro", int64_t(router_expert_tiles_per_column) * router_row_tiles * kFp32TileBytes},
                };
                need = 0;
                for (const auto& [_, bytes] : router_regions) {
                    need += align_to_64(bytes);
                }
                if (router_start + need <= limit || router_k_chunk_tiles == k_chunk_candidates.back() ||
                    (router_k_chunk_tiles == k_chunk_candidates.front() && router_k_tiles / router_k_chunk_tiles == 2 &&
                     arena_end + need <= arena_max)) {
                    break;
                }
            }
            if (arena_end - router_start < need && router_start + need <= limit) {
                arena_end = router_start + need;
            } else if (arena_end - router_start < need) {
                const int64_t hall_room = region(offsets, "xin") - region(offsets, "hall");
                router_start = hall_room >= need ? region(offsets, "hall") : arena_end;
                if (router_start == arena_end) {
                    arena_end = arena_end + need;
                }
            }
            for (const auto& [name, bytes] : router_regions) {
                set_region(offsets, name, router_start);
                router_start = align_to_64(router_start + bytes);
            }
        }
        arena_bytes = arena_end;
        if (arena_end <= arena_max) {
            break;
        }
    }
    std::set<int64_t> ends{arena_end};
    for (const auto& [_, offset] : offsets) {
        ends.insert(offset);
    }
    sizes.clear();
    for (const auto& [name, offset] : offsets) {
        sizes.emplace_back(name, offset < arena_end ? *ends.upper_bound(offset) - offset : 0);
    }
    {  // (the router's buffers over a region another chip writes: those writes wait for "router done")
        const int64_t router_low = at("rx"),
                      router_high = at("ro") + int64_t(router_expert_tiles_per_column) * row_tiles_per_core *
                                                   row_chunks / sizing_args.router_iterations * kFp32TileBytes;
        auto overlaps = [&](const std::string& name, int64_t bytes) {
            return bytes > 0 && at(name) < router_high && router_low < at(name) + bytes;
        };
        router_overlays_zones = overlaps("arrf", region(reduce_scatter_sizes, "arrf")) ||
                                overlaps("arrb", region(reduce_scatter_sizes, "arrb"));
        if (sizing_args.latent) {
            router_overlays_latent = overlaps(
                "latz",
                int64_t(shards - 1) * row_tiles_per_core * max_latent_column_tiles * kBf16TileBytes +
                    int64_t(row_chunks - 1) * latent_pages * kBf16TileBytes);
        }
    }
    router_overlay = router_overlays_zones || router_overlays_latent;
    if (arena_end > arena_max) {
        throw ArenaTooLarge("moe_head shared expert: arena too large", arena_end, arena_max);
    }
    for (uint32_t column = 0; column < engine_columns; ++column) {
        auto widths_key = std::make_pair(gate_up_pass_widths[column], down_pass_widths[column]);
        auto found =
            std::find_if(groups.begin(), groups.end(), [&](const auto& group) { return group.first == widths_key; });
        if (found == groups.end()) {
            groups.push_back({widths_key, {column}});
        } else {
            found->second.push_back(column);
        }
    }
}

int64_t SharedExpertSizing::at(const std::string& region_name) const { return region(offsets, region_name); }
bool SharedExpertSizing::in_family0(uint32_t column) const {
    return std::find(family0_columns.begin(), family0_columns.end(), column) != family0_columns.end();
}
bool SharedExpertSizing::in_family1(uint32_t column) const {
    return std::find(family1_columns.begin(), family1_columns.end(), column) != family1_columns.end();
}

std::tuple<uint32_t, uint32_t, uint32_t> SharedExpertSizing::owner(uint32_t step) const {
    const uint32_t family = step & 1;
    const auto& family_columns = family ? family1_columns : family0_columns;
    const uint32_t family_size = family ? family1_size : family0_size;
    return {family_columns[(step / 2) % family_size], family, (step / 2) % family_size};
}

uint32_t SharedExpertSizing::global_row_tile(uint32_t chunk, uint32_t local_row_tile) const {
    if (row_chunks == 1 || shards == 1) {
        return chunk * chunk_row_tiles + local_row_tile;
    }
    const uint32_t shard_row_tiles = chunk_row_tiles / shards;
    return (local_row_tile / shard_row_tiles) * (rows / 32 / shards) + chunk * shard_row_tiles +
           local_row_tile % shard_row_tiles;
}

std::pair<uint32_t, uint32_t> SharedExpertSizing::latent_rows(uint32_t core_y, uint32_t block) const {
    const uint32_t shard_row_tiles = chunk_row_tiles / shards;
    uint32_t first_row = 0, row_count = 0;
    for (uint32_t row = 0; row < row_tiles_per_core; ++row) {
        if ((core_y * row_tiles_per_core + row) / shard_row_tiles == block) {
            first_row = row_count ? first_row : row;
            ++row_count;
        }
    }
    return {row_count ? first_row : 0, row_count};
}

uint32_t SharedExpertSizing::w_block_tiles(uint32_t column, uint32_t block) const {
    return block < gate_up_passes * num_k_steps
               ? k_step_tiles * gate_up_pass_widths[column][block / num_k_steps]
               : down_k_step_tiles * down_pass_widths[column][(block - gate_up_passes * num_k_steps) / down_k_steps];
}

bool SharedExpertSizing::final_rows() const {
    const ReduceScatterPlan* reduce_scatter = args.reduce_scatter;
    return reduce_scatter->scatter == -1 && row_chunks > 1 && shards > 1 &&
           (chunk_row_tiles / shards) % row_tiles_per_core != 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// the router cores / the routing exchange

std::pair<Regions, int64_t> router_layout(
    uint32_t tp_size, uint32_t expert_tiles, uint32_t top_k, uint32_t full_expert_tiles, bool arrivals_first) {
    full_expert_tiles = full_expert_tiles ? full_expert_tiles : expert_tiles;
    Regions layout = {
        {"hold", int64_t(tp_size) * expert_tiles * kFp32TileBytes},
        {"sc", int64_t(expert_tiles) * kFp32TileBytes},
        {"idx", int64_t(expert_tiles) * kBf16TileBytes},
        {"best", kFp32TileBytes},
        {"bias", int64_t(full_expert_tiles) * kFp32TileBytes},
        {"arr", int64_t(tp_size - 1) * expert_tiles * kFp32TileBytes},
        {"finc", 16 * 8},
        {"psync", 16 * 2},
        {"arrb", 16 * int64_t(std::max<uint32_t>(1, tp_size - 1))},
        {"rdy", 16 * 2},
        {"out", 32 * 4 * int64_t(top_k)},
        {"words", 160},
        {"route", 32 * 4 * 2},
        {"msync", 16},
    };
    if (full_expert_tiles != expert_tiles) {
        layout.emplace_back("part", 2 * kFp32TileBytes);
    }
    if (arrivals_first) {
        std::swap(layout[4], layout[5]);
    }
    Regions offsets;
    int64_t end = 0;
    for (const auto& [name, bytes] : layout) {
        offsets.emplace_back(name, end);
        end = align_to_64(end + bytes);
    }
    return {offsets, end};
}

int64_t router_min_bytes(uint32_t tp_size, uint32_t experts, uint32_t top_k) {
    const uint32_t expert_tiles = experts / 32;
    return router_layout(tp_size, expert_tiles, top_k, expert_tiles).second;
}

int64_t routing_header_bytes(uint32_t experts_per_chip, uint32_t instances) {
    return align_to_64(
        64 + align_to_16(2 * int64_t(experts_per_chip)) +
        int64_t(instances) * align_to_16(2 * (int64_t(experts_per_chip) + 1)));
}

int64_t routing_slot_bytes(uint32_t tokens_per_chip, uint32_t top_k, uint32_t experts_per_chip, uint32_t instances) {
    return align_to_64(
        routing_header_bytes(experts_per_chip, instances) +
        4 * int64_t(tokens_per_chip) * std::min(top_k, experts_per_chip));
}

int64_t routing_worst_rows(uint32_t num_chips, uint32_t tokens_per_chip, uint32_t top_k, uint32_t experts_per_chip) {
    return (int64_t(num_chips) * tokens_per_chip * std::min(top_k, experts_per_chip) + 31) / 32 * 32 +
           32 * int64_t(experts_per_chip);
}

RoutingSizes routing_exchange_sizes(
    uint32_t num_chips,
    uint32_t tokens_per_chip,
    uint32_t top_k,
    uint32_t experts_per_chip,
    uint32_t accumulators,
    int64_t rows_capacity,
    uint32_t instances,
    uint32_t tokens_per_page) {
    instances = instances ? instances : std::min({8u, experts_per_chip, accumulators});
    const int64_t global_experts = int64_t(num_chips) * experts_per_chip;
    const int64_t routing_header_padded_bytes = routing_header_bytes(experts_per_chip, instances);
    const int64_t routing_slot_capacity_bytes = routing_slot_bytes(tokens_per_chip, top_k, experts_per_chip, instances);
    const int64_t owned_accumulators_max = (accumulators + instances - 1) / instances;
    const int64_t home_tokens_max = owned_accumulators_max * ((tokens_per_chip + accumulators - 1) / accumulators);
    const int64_t instance_block_bytes = align_to_16(2 * (int64_t(experts_per_chip) + 1));
    const int64_t max_picks_per_chip = std::min(top_k, experts_per_chip);
    const int64_t uncapped_entries = (ceil_div(int64_t(num_chips) * tokens_per_chip * max_picks_per_chip, instances) +
                                      int64_t(num_chips) * tokens_per_chip + 31) /
                                     32 * 32;
    const int64_t entries_capacity = std::min(uncapped_entries, rows_capacity);
    const int64_t instance_rows = std::min(entries_capacity + 32 * int64_t(experts_per_chip), rows_capacity);
    const int64_t chips_per_instance = (int64_t(num_chips) + instances - 1) / instances;
    const int64_t planner_l1_bytes =
        align_to_64((tokens_per_chip / tokens_per_page) * align_to_64(int64_t(tokens_per_page) * 4 * top_k)) +
        align_to_64(4 * int64_t(instances) * num_chips) + align_to_64(4 * num_chips) +
        num_chips * instance_block_bytes + align_to_64(4 * home_tokens_max * top_k + 16 * int64_t(num_chips)) +
        num_chips * routing_header_padded_bytes +
        align_to_64(4 * entries_capacity + 128 * int64_t(num_chips) * instances) +
        align_to_64(4 * int64_t(num_chips) * instances) + align_to_64(2 * int64_t(experts_per_chip) * tokens_per_chip) +
        align_to_64(4 * instance_rows) + align_to_64(8 * instance_rows) + align_to_64(128 * owned_accumulators_max) +
        instances * 64 + moe_block_plan_words::kPlanHeadBytes + 2 * align_to_64(4 * (int64_t(experts_per_chip) + 32)) +
        64 + num_chips * align_to_16(2 * int64_t(experts_per_chip)) +
        chips_per_instance * instances * align_to_16(2 * int64_t(experts_per_chip)) +
        align_to_64(chips_per_instance * align_to_16(2 * int64_t(experts_per_chip))) + 128;
    const int64_t port_l1_bytes = int64_t(num_chips) * routing_slot_capacity_bytes + 64;
    std::vector<std::string> reasons;
    if (num_chips > moe_block_plan_words::kMaxChips || accumulators > moe_block_plan_words::kAccMax || top_k % 2 ||
        tokens_per_chip % tokens_per_page ||
        !(1 <= instances && instances <= std::min({experts_per_chip, accumulators, 16u}))) {
        reasons.push_back(
            "shape: N " + std::to_string(num_chips) + " <= 32, A " + std::to_string(accumulators) +
            " <= " + std::to_string(moe_block_plan_words::kAccMax) + ", K " + std::to_string(top_k) + " even, S " +
            std::to_string(tokens_per_chip) + " % " + std::to_string(tokens_per_page) + ", M " +
            std::to_string(instances) + " <= min(EPC, A, 16)");
    }
    if (global_experts > 1024) {
        reasons.push_back("local counters: N EPC " + std::to_string(global_experts) + " > 1024");
    }
    if (tokens_per_chip > 2048 || experts_per_chip > 64) {
        reasons.push_back(
            "an entry: S " + std::to_string(tokens_per_chip) + " <= 2048 and EPC " + std::to_string(experts_per_chip) +
            " <= 64");
    }
    const int64_t l1_limit = 1'400'000;
    if (planner_l1_bytes > l1_limit || port_l1_bytes > l1_limit) {
        reasons.push_back(
            "L1: planner " + std::to_string(planner_l1_bytes >> 10) + " KB, port " +
            std::to_string(port_l1_bytes >> 10) + " KB");
    }
    std::string reason;
    for (size_t i = 0; i < reasons.size(); ++i) {
        reason += (i ? "; " : "") + reasons[i];
    }
    return RoutingSizes{instances, entries_capacity, planner_l1_bytes, port_l1_bytes, reasons.empty(), reason};
}

PlannerPlace place_planner(
    uint32_t num_chips,
    uint32_t tokens_per_chip,
    uint32_t top_k,
    uint32_t experts_per_chip,
    uint32_t accumulators,
    uint32_t num_router_cores,
    int64_t overlay,
    int64_t router_bytes,
    int64_t arena,
    uint32_t fixed_instances,
    uint32_t num_planner_only_cores) {
    PlannerPlace place;
    place.num_router_cores = num_router_cores;
    place.num_planner_only_cores = num_planner_only_cores;
    const int64_t rows_capacity = routing_worst_rows(num_chips, tokens_per_chip, top_k, experts_per_chip);
    uint32_t instances = fixed_instances ? fixed_instances
                                         : std::min(
                                               {num_router_cores + num_planner_only_cores,
                                                experts_per_chip,
                                                accumulators,
                                                14u});  // (one instance a core: planner_slot)
    for (; instances > 0; --instances) {
        place.sizes = routing_exchange_sizes(
            num_chips, tokens_per_chip, top_k, experts_per_chip, accumulators, rows_capacity, instances);
        place.step = align_to_64(place.sizes.planner_l1_bytes);
        // (the regions a planner-only core and a router core hold at most)
        const int64_t planner_only_core_regions = num_planner_only_cores && instances ? 1 : 0;
        const int64_t router_core_instances =
            std::max<int64_t>(0, int64_t(instances) - int64_t(num_planner_only_cores));
        const int64_t router_core_regions = ceil_div(router_core_instances, int64_t(num_router_cores));
        place.router_region_start = router_core_regions * place.step <= overlay ? 0 : align_to_64(router_bytes);
        if (place.sizes.fits && place.router_region_start + router_core_regions * place.step <= arena &&
            planner_only_core_regions * place.step <= arena) {
            place.ok = true;
            place.planner_instances = instances;
            return place;
        }
        if (fixed_instances) {
            break;
        }
    }
    place.planner_instances = instances;
    return place;
}

// ---------------------------------------------------------------------------------------------------------------------
// routing_paths: the routing exchange's senders

namespace head_sizing_detail {

enum : int { kForward = 1, kBackward = 0 };  // (x sender keys sort "bwd" < "fwd")

// A sender key, ordered ("x", "bwd" | "fwd", link) < ("y", 2) < ("y", 3).
using SenderKey = std::tuple<int, int, int>;  // (0 x | 1 y, x: direction / y: 2 | 3, link)

struct HostChip {
    std::optional<std::pair<int, int>> next, previous;
    bool sends[2] = {
        false, false};  // [kBackward], [kForward]: the gather's schedule toward previous / next is non-empty
    bool links[2] = {false, false};  // a direction carries links
};

std::string coord_string(int row, int column) {
    return "(" + std::to_string(row) + ", " + std::to_string(column) + ")";
}

}  // namespace head_sizing_detail

std::variant<RoutingPaths, std::string> routing_paths(
    uint32_t mesh_rows, uint32_t mesh_columns, bool ring, uint32_t links, const std::vector<uint8_t>& first_hops) {
    const int port_links = 2;  // (ROUTING_L: the routing exchange's port tables)
    const int num_chips = int(mesh_rows * mesh_columns);
    std::map<std::pair<int, int>, HostChip> chips;
    for (int row = 0; row < int(mesh_rows); ++row) {
        for (int column = 0; column < int(mesh_columns); ++column) {
            HostChip chip;
            if (ring || column < int(mesh_columns) - 1) {
                chip.next = std::make_pair(row, int(positive_modulo(column + 1, mesh_columns)));
            }
            if (ring || column > 0) {
                chip.previous = std::make_pair(row, int(positive_modulo(column - 1, mesh_columns)));
            }
            // the gather's schedule: which directions send shards
            int gather_size = int(mesh_columns);
            if (ring && gather_size % 2 == 0 && gather_size >= 4) {
                chip.sends[kForward] = gather_size / 2 > 0;
                chip.sends[kBackward] = gather_size / 2 > 0;
            } else if (ring) {
                chip.sends[kForward] = gather_size / 2 > 0;
                chip.sends[kBackward] = gather_size - 1 - gather_size / 2 > 0;
            } else {
                chip.sends[kForward] = column < gather_size - 1;
                chip.sends[kBackward] = column > 0;
            }
            chips[{row, column}] = chip;
        }
    }
    for (auto& [coord, chip] : chips) {
        for (int direction : {kForward, kBackward}) {
            const auto& peer = direction == kForward ? chip.next : chip.previous;
            // (needs_ready, as in head_model.cpp: the peer's opposite-direction port sends to this chip)
            if (peer && (chip.sends[direction] || chips.at(*peer).sends[1 - direction])) {
                chip.links[direction] = links > 0;
            }
        }
    }
    if (first_hops.size() != size_t(num_chips) * size_t(num_chips)) {
        return "the fabric's first hops: " + std::to_string(first_hops.size()) + " entries for " +
               std::to_string(num_chips) + " chips";
    }
    auto x_directions = [&](int row, int column) {
        std::map<int, int> directions;  // 0 (X+) | 1 (X-) -> the gather's direction toward it
        const auto& host_chip = chips.at({row, column});
        for (int direction : {kForward, kBackward}) {
            const auto& peer = direction == kForward ? host_chip.next : host_chip.previous;
            if (peer && host_chip.links[direction]) {
                bool forward = ring ? positive_modulo(peer->second - column, mesh_columns) == 1 : peer->second > column;
                directions[forward ? 0 : 1] = direction;
            }
        }
        return directions;
    };
    // the sender of chip source's packet to chip destination: the one the fabric's first hop from source toward
    // destination leaves by (HeadShape.first_hops: asked of the fabric, never derived) - a Y sender, or the gather's X
    // port that way (link destination % port_links); one fabric send, the fabric routing every hop to destination
    // (head_model.cpp HeadModel::sender: the same)
    auto sender = [&](int source, int destination) {
        const int first_hop = first_hops[size_t(source) * size_t(num_chips) + size_t(destination)];
        if (first_hop >= 2) {
            return SenderKey{1, first_hop, 0};
        }
        const int row = source / int(mesh_columns), column = source % int(mesh_columns);
        auto directions = x_directions(row, column);
        if (!directions.count(first_hop)) {
            throw SizingError(
                "chip " + coord_string(row, column) + ": no X port toward " + std::to_string(destination));
        }
        const int link = (destination % port_links) < int(links) ? destination % port_links : 0;
        return SenderKey{0, directions[first_hop], link};
    };
    try {
        uint32_t max_x_packets = 0, max_y_packets = 0, max_senders = 0;
        for (const auto& [coord, _] : chips) {
            int my_chip = coord.first * int(mesh_columns) + coord.second;
            std::map<SenderKey, uint32_t> packets;
            for (int destination = 0; destination < num_chips; ++destination) {
                if (destination != my_chip) {
                    ++packets[sender(my_chip, destination)];
                }
            }
            std::vector<SenderKey> y_keys;
            for (const auto& [key, count] : packets) {
                if (std::get<0>(key) == 1) {
                    y_keys.push_back(key);
                }
            }
            ++packets[y_keys.empty() ? packets.begin()->first : y_keys[0]];  // (my own chip)
            if (y_keys.size() > 2) {
                throw SizingError(
                    "chip " + coord_string(coord.first, coord.second) + ": " + std::to_string(y_keys.size()) +
                    " Y directions");
            }
            for (const auto& [key, count] : packets) {
                (std::get<0>(key) == 0 ? max_x_packets : max_y_packets) =
                    std::max(std::get<0>(key) == 0 ? max_x_packets : max_y_packets, count);
            }
            max_senders = std::max<uint32_t>(max_senders, packets.size());
        }
        return RoutingPaths{max_x_packets, max_y_packets, max_senders};
    } catch (const SizingError& error) {
        return std::string(error.what());
    }
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head
