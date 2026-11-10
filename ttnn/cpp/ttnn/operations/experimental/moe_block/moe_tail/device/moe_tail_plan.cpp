// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// moe_tail's host plans (moe_tail_plan.hpp has the overview): the pure sizing and scheduling functions first, then
// TailSchedule (the all-to-all's routing and fence counts), the all-to-all plan, and the latent engine's geometry,
// watchers and plan.

#include "moe_tail_plan.hpp"

#include <algorithm>
#include <numeric>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"
#include "ttnn/operations/experimental/moe_block/common/port_placement.hpp"

namespace ttnn::operations::experimental::moe_block::moe_tail {

namespace moe_tail_plan_detail {

using moe_block_tiles::kBf16TileBytes;
using moe_block_tiles::kFp32TileBytes;
using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;

uint32_t ceil_div(uint32_t numerator, uint32_t denominator) { return (numerator + denominator - 1) / denominator; }

// len(range(start, stop, step))
uint32_t range_length(uint32_t start, uint32_t stop, uint32_t step) {
    return start < stop ? ceil_div(stop - start, step) : 0;
}

}  // namespace moe_tail_plan_detail
using namespace moe_tail_plan_detail;

std::optional<uint32_t> row_neighbor(uint32_t position, uint32_t tp_size, bool ring, TailDirection direction) {
    if (direction == TailDirection::Forward) {
        return (ring || position + 1 < tp_size) ? std::optional<uint32_t>((position + 1) % tp_size) : std::nullopt;
    }
    return (ring || position > 0) ? std::optional<uint32_t>((position + tp_size - 1) % tp_size) : std::nullopt;
}

std::pair<std::vector<Unit>, std::vector<Unit>> gather_schedule(uint32_t position, uint32_t tp_size, bool ring) {
    std::vector<Unit> forward, backward;
    if (ring && tp_size % 2 == 0 && tp_size >= 4) {
        const uint32_t half = tp_size / 2;
        for (uint32_t i = 0; i < half; ++i) {
            forward.push_back({(position + tp_size - i) % tp_size, i == half - 1 ? 1u : 0u});
            backward.push_back({(position + i) % tp_size, i == half - 1 ? 2u : 0u});
        }
        return {forward, backward};
    }
    if (ring) {
        const uint32_t num_forward = tp_size / 2, num_backward = tp_size - 1 - num_forward;
        for (uint32_t i = 0; i < num_forward; ++i) {
            forward.push_back({(position + tp_size - i) % tp_size, 0});
        }
        for (uint32_t i = 0; i < num_backward; ++i) {
            backward.push_back({(position + i) % tp_size, 0});
        }
        return {forward, backward};
    }
    if (position + 1 < tp_size) {
        for (uint32_t i = 0; i <= position; ++i) {
            forward.push_back({position - i, 0});
        }
    }
    if (position > 0) {
        for (uint32_t i = 0; i < tp_size - position; ++i) {
            backward.push_back({position + i, 0});
        }
    }
    return {forward, backward};
}

std::vector<uint32_t> chunk_order(uint32_t position, uint32_t tp_size, bool ring) {
    auto distance = [&](uint32_t other) {
        return ring ? std::min((other + tp_size - position) % tp_size, (position + tp_size - other) % tp_size)
                    : (other > position ? other - position : position - other);
    };
    std::vector<uint32_t> out(tp_size);
    std::iota(out.begin(), out.end(), 0u);
    std::stable_sort(out.begin(), out.end(), [&](uint32_t left, uint32_t right) {
        const auto key = [&](uint32_t other) {
            return std::make_tuple(distance(other), (other + tp_size - position) % tp_size != tp_size - 1, other);
        };
        return key(left) < key(right);
    });
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> even_split(uint32_t count, uint32_t parts) {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    const uint32_t base = count / parts, remainder = count % parts;
    uint32_t first = 0;
    for (uint32_t i = 0; i < parts; ++i) {
        const uint32_t width = base + (i < remainder ? 1 : 0);
        out.emplace_back(first, first + width);
        first += width;
    }
    return out;
}

std::vector<uint32_t> column_split(uint32_t num_tiles, uint32_t columns, uint32_t unit) {
    TT_FATAL(num_tiles % unit == 0, "moe_tail: {} tiles in units of {}", num_tiles, unit);
    const uint32_t base = num_tiles / unit / columns, remainder = num_tiles / unit % columns;
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < columns; ++i) {
        out.push_back((base + (i < remainder ? 1 : 0)) * unit);
    }
    return out;
}

uint32_t ring_pages(uint32_t pages, const std::vector<uint32_t>& pushes) {
    const std::set<uint32_t> sizes(pushes.begin(), pushes.end());
    if (sizes.size() == 1 && pages % *sizes.begin() == 0) {
        return pages;
    }
    return pages + *sizes.rbegin() - 1;
}

// ---- TailSchedule ----

TailSchedule::TailSchedule(
    uint32_t tp_size,
    bool ring,
    uint32_t num_links,
    uint32_t block_tiles,
    uint32_t owned_row_tiles,
    uint32_t num_dram_banks,
    uint32_t pages_per_run,
    std::map<std::pair<uint32_t, uint32_t>, TailDirection> ties) :
    tp_size(tp_size),
    num_links(num_links),
    block_tiles(block_tiles),
    owned_row_tiles(owned_row_tiles),
    num_dram_banks(num_dram_banks),
    pages_per_run(pages_per_run),
    ring(ring),
    ties(std::move(ties)) {
    for (uint32_t source = 0; source < tp_size; ++source) {
        for (TailDirection toward : {TailDirection::Forward, TailDirection::Backward}) {
            for (uint32_t link = 0; link < num_links; ++link) {
                for (const auto& [destination, part] : destinations(source, toward)) {
                    // (my arrivals: the destination's port facing the side my packets come in by; my ready: its port
                    // facing the fabric's way back to me)
                    expected_arrivals_[{destination, opposite(toward), link}] +=
                        ceil_div(owned_row_tiles * runs(part, link), kArrivalIncrementEvery);
                    readies_[{destination, direction(destination, source), link}].insert({source, toward, link});
                }
            }
        }
    }
}

TailDirection TailSchedule::direction(uint32_t source, uint32_t destination) const {
    if (!ring) {
        return destination > source ? TailDirection::Forward : TailDirection::Backward;
    }
    const uint32_t forward_distance = (destination + tp_size - source) % tp_size;
    if (2 * forward_distance == tp_size) {
        auto tie = ties.find({source, destination});
        return tie == ties.end() ? TailDirection::Forward : tie->second;
    }
    return 2 * forward_distance < tp_size ? TailDirection::Forward : TailDirection::Backward;
}

std::vector<Unit> TailSchedule::destinations(uint32_t source, TailDirection toward) const {
    std::vector<Unit> out;
    if (!ring) {
        if (toward == TailDirection::Forward) {
            for (uint32_t destination = source + 1; destination < tp_size; ++destination) {
                out.push_back({destination, 0});
            }
        } else {
            for (uint32_t destination = source; destination-- > 0;) {
                out.push_back({destination, 0});
            }
        }
        return out;
    }
    for (uint32_t distance = 1; distance < tp_size; ++distance) {
        const uint32_t destination =
            toward == TailDirection::Forward ? (source + distance) % tp_size : (source + tp_size - distance) % tp_size;
        if (direction(source, destination) == toward) {
            out.push_back({destination, 0});
        }
    }
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> TailSchedule::run_list(uint32_t part, uint32_t link) const {
    const uint32_t banks_used = std::min(num_dram_banks, block_tiles);
    const uint32_t my_offsets = range_length(link, banks_used, num_links);
    const uint32_t first_offset = part == 2 ? my_offsets / 2 : 0, end_offset = part == 1 ? my_offsets / 2 : my_offsets;
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (uint32_t i = first_offset; i < end_offset; ++i) {
        const uint32_t bank_offset = link + i * num_links;
        for (uint32_t run_first_page = bank_offset; run_first_page < block_tiles;
             run_first_page += pages_per_run * num_dram_banks) {
            out.emplace_back(
                run_first_page, std::min(pages_per_run, ceil_div(block_tiles - run_first_page, num_dram_banks)));
        }
    }
    return out;
}

uint32_t TailSchedule::hops(uint32_t source, uint32_t destination, TailDirection toward) const {
    return toward == TailDirection::Forward ? (destination + tp_size - source) % tp_size
                                            : (source + tp_size - destination) % tp_size;
}

uint32_t TailSchedule::expected_arrivals(const Port& port) const {
    auto found = expected_arrivals_.find(port);
    return found == expected_arrivals_.end() ? 0 : found->second;
}

std::vector<TailSchedule::Port> TailSchedule::readies(const Port& port) const {
    auto found = readies_.find(port);
    if (found == readies_.end()) {
        return {};
    }
    std::vector<Port> out(found->second.begin(), found->second.end());
    // (sorted by (position, direction, link), Backward first)
    std::sort(out.begin(), out.end(), [](const Port& left, const Port& right) {
        return std::make_tuple(std::get<0>(left), std::get<1>(left) == TailDirection::Forward, std::get<2>(left)) <
               std::make_tuple(std::get<0>(right), std::get<1>(right) == TailDirection::Forward, std::get<2>(right));
    });
    return out;
}

TailAllToAllGeometry TailAllToAllGeometry::make(
    uint32_t tp_size, uint32_t rows, uint32_t hidden_dim, uint32_t num_dram_banks, size_t max_payload_bytes) {
    TailAllToAllGeometry geometry;
    geometry.tp_size = tp_size;
    geometry.rows = rows;
    geometry.hidden_dim = hidden_dim;
    geometry.owned_rows = rows / tp_size;
    TT_FATAL(
        rows % tp_size == 0 && geometry.owned_rows % TILE_HEIGHT == 0 && hidden_dim % (TILE_WIDTH * tp_size) == 0,
        "moe_tail (all-to-all): {} rows, H {} over TP {}: whole tiles",
        rows,
        hidden_dim,
        tp_size);
    geometry.owned_row_tiles = geometry.owned_rows / TILE_HEIGHT;
    geometry.hidden_tiles = hidden_dim / TILE_WIDTH;
    geometry.block_tiles = geometry.hidden_tiles / tp_size;
    for (uint32_t divisor = 1; divisor <= 8;
         ++divisor) {  // (a worker's tile columns: up to 8, the add's DEST tiles at bfloat16)
        if (geometry.block_tiles % divisor == 0) {
            geometry.worker_columns = divisor;
        }
    }
    geometry.num_workers = geometry.hidden_tiles / geometry.worker_columns;
    geometry.num_dram_banks = num_dram_banks;
    geometry.pages_per_run = std::max<uint32_t>(1, static_cast<uint32_t>(max_payload_bytes / kBf16TileBytes));
    // (a port's staging CB: two groups in flight within 96 KB, at most 8 runs a group)
    geometry.runs_per_group =
        std::max<uint32_t>(1, std::min<uint32_t>(8, (96 * 1024) / (2 * geometry.pages_per_run * kBf16TileBytes)));
    return geometry;
}

std::shared_ptr<const TailAllToAllPlan> make_tail_all_to_all_plan(
    MoEBlockResources& resources, uint32_t rows, uint32_t hidden_dim) {
    auto& mesh = resources.mesh_device();
    const auto& topology = resources.topology();
    const uint32_t tp_size = topology.C;
    TT_FATAL(tp_size >= 2, "moe_tail (all-to-all): a TP group (mesh row) of {} chips", tp_size);
    TT_FATAL(
        tp_size - 1 <= kMaxPortDestinations,
        "moe_tail (all-to-all): TP {}: a port sends to at most {} chips",
        tp_size,
        kMaxPortDestinations);
    const auto dram_grid = mesh.dram_grid_size();
    auto plan = std::make_shared<TailAllToAllPlan>();
    plan->geometry = TailAllToAllGeometry::make(
        tp_size, rows, hidden_dim, dram_grid.x * dram_grid.y, resources.fabric_max_payload_bytes());
    plan->topology = topology;
    plan->ring = topology.wrap_x;
    plan->num_links = resources.config().num_links;
    const auto& geometry = plan->geometry;

    auto node = [&](ChipCoord chip) { return mesh.get_fabric_node_id(mesh_coord(chip, topology)); };
    // The ring's far side (distance tp_size / 2): the way the fabric routes it from `position`, the same on every mesh
    // row.
    std::map<std::pair<uint32_t, uint32_t>, TailDirection> ties;
    if (plan->ring && tp_size % 2 == 0) {
        for (uint32_t mesh_row = 0; mesh_row < topology.R; ++mesh_row) {
            for (uint32_t position = 0; position < tp_size; ++position) {
                const uint32_t far_position = (position + tp_size / 2) % tp_size;
                const auto forward = tt::tt_fabric::get_eth_forwarding_direction(
                    node({mesh_row, position}), node({mesh_row, (position + 1) % tp_size}));
                const auto toward_far_position = tt::tt_fabric::get_eth_forwarding_direction(
                    node({mesh_row, position}), node({mesh_row, far_position}));
                const TailDirection direction =
                    toward_far_position == forward ? TailDirection::Forward : TailDirection::Backward;
                const auto [existing, inserted] = ties.emplace(std::make_pair(position, far_position), direction);
                TT_FATAL(
                    inserted || existing->second == direction,
                    "moe_tail (all-to-all): ring ties differ between mesh rows");
            }
        }
    }
    plan->links = row_links(resources.mesh_plan());
    plan->schedule = TailSchedule(
        tp_size,
        plan->ring,
        plan->num_links,
        geometry.block_tiles,
        geometry.owned_row_tiles,
        geometry.num_dram_banks,
        geometry.pages_per_run,
        std::move(ties));

    // Ports: one per (direction, link) with a neighbour on the row, next to its link's Ethernet core (NoC1).
    MeshPlan row_plan = resources.mesh_plan();
    for (auto& [chip, directions] : row_plan.chips) {
        directions[static_cast<size_t>(Dir::YPlus)].reset();
        directions[static_cast<size_t>(Dir::YMinus)].reset();
    }
    const auto cores = place_ports(mesh, row_plan);
    std::set<std::pair<size_t, size_t>> taken;
    std::vector<CoreRange> ranges;
    for (const auto& [key, core] : cores) {
        const TailDirection direction = key.dir == Dir::XPlus ? TailDirection::Forward : TailDirection::Backward;
        plan->ports[{key.chip, direction, key.link_index}] = core;
        if (taken.insert({core.y, core.x}).second) {
            ranges.emplace_back(core, core);
        }
    }
    plan->port_union = CoreRangeSet(ranges);
    const auto grid = mesh.compute_with_storage_grid_size();
    for (uint32_t y = 0; y < grid.y && plan->workers.size() < geometry.num_workers; ++y) {
        for (uint32_t x = 0; x < grid.x && plan->workers.size() < geometry.num_workers; ++x) {
            if (!taken.contains({y, x})) {
                plan->workers.emplace_back(x, y);
            }
        }
    }
    TT_FATAL(
        plan->workers.size() == geometry.num_workers,
        "moe_tail (all-to-all): {} workers, {} free cores",
        geometry.num_workers,
        plan->workers.size());
    return plan;
}

// ---- latent engine ----

TailEngineGeometry TailEngineGeometry::make(
    uint32_t rows,
    uint32_t latent_dim,
    uint32_t output_dim,
    uint32_t num_chunks,
    uint32_t engine_columns,
    uint32_t engine_rows,
    uint32_t w_tile_bytes,
    uint32_t capacity_bytes) {
    TailEngineGeometry geometry;
    TT_FATAL(engine_rows % 2 == 0, "moe_tail (engine): {} engine rows is odd", engine_rows);
    TT_FATAL(
        latent_dim % (2 * TILE_WIDTH) == 0 && output_dim % TILE_WIDTH == 0,
        "moe_tail (engine): latent_dim {} / output_dim {}: whole tiles of each K half",
        latent_dim,
        output_dim);
    TT_FATAL(
        rows % (TILE_HEIGHT * num_chunks) == 0,
        "moe_tail (engine): {} rows: whole row tiles a chunk ({} chunks)",
        rows,
        num_chunks);
    geometry.rows = rows;
    geometry.latent_dim = latent_dim;
    geometry.output_dim = output_dim;
    geometry.num_chunks = num_chunks;
    geometry.engine_columns = engine_columns;
    geometry.engine_rows = engine_rows;
    geometry.w_tile_bytes = w_tile_bytes;
    geometry.latent_tiles = latent_dim / TILE_WIDTH;
    geometry.output_tiles = output_dim / TILE_WIDTH;
    geometry.row_tiles_per_chunk = rows / TILE_HEIGHT / num_chunks;
    geometry.num_pairs = engine_rows / 2;
    const uint32_t row_tiles = num_chunks * geometry.row_tiles_per_chunk;
    for (uint32_t pair = 0; pair < geometry.num_pairs; ++pair) {
        geometry.steps.push_back(range_length(pair, row_tiles, geometry.num_pairs));
        geometry.own_steps.push_back(range_length(pair, geometry.row_tiles_per_chunk, geometry.num_pairs));
    }
    geometry.max_steps = *std::max_element(geometry.steps.begin(), geometry.steps.end());
    geometry.max_own_steps = *std::max_element(geometry.own_steps.begin(), geometry.own_steps.end());
    geometry.k_halves = even_split(geometry.latent_tiles, 2);
    geometry.k_half_tiles = geometry.k_halves[0].second;
    TT_FATAL(
        geometry.k_halves[1].second - geometry.k_halves[1].first == geometry.k_half_tiles,
        "moe_tail (engine): equal K halves");
    geometry.column_tiles = column_split(geometry.output_tiles, engine_columns, 1);
    uint32_t first_tile = 0;
    for (uint32_t column = 0; column < engine_columns; ++column) {
        geometry.column_first_tile.push_back(first_tile);
        first_tile += geometry.column_tiles[column];
    }
    geometry.max_column_tiles = *std::max_element(geometry.column_tiles.begin(), geometry.column_tiles.end());
    // (a step's column tiles in DEST at once: 8 fp32 tiles at full sync, one kept for the scale)
    TT_FATAL(
        geometry.max_column_tiles <= 7, "moe_tail (engine): a step's {} fp32 tiles in DEST", geometry.max_column_tiles);
    // (the columns a core finishes: half 0 the first ceil(tiles / 2), half 1 the rest - at most kExchangeTiles - 1
    // either way)
    std::set<uint32_t> counts;
    for (uint32_t tiles : geometry.column_tiles) {
        const uint32_t first_half = ceil_div(tiles, 2);
        geometry.finished_columns.push_back(
            {std::make_pair(0u, first_half), std::make_pair(first_half, tiles - first_half)});
        for (const auto& [offset, count] : geometry.finished_columns.back()) {
            TT_FATAL(
                count > 0 && count <= kExchangeTiles - 1 && tiles - count <= kExchangeTiles - 1,
                "moe_tail (engine): {} columns a core",
                tiles);
            counts.insert(count);
        }
    }
    geometry.finished_counts.assign(counts.begin(), counts.end());
    uint32_t least_common_multiple = 1;
    for (uint32_t count : geometry.finished_counts) {
        least_common_multiple = std::lcm(least_common_multiple, count);
    }
    geometry.finished_ring_pages =
        least_common_multiple * ceil_div(2 * geometry.finished_counts.back(), least_common_multiple);
    geometry.w_k_ranges = even_split(geometry.k_half_tiles, geometry.num_pairs);
    geometry.share_k_ranges = even_split(geometry.k_half_tiles, engine_columns);
    for (const auto& [k_first, k_end] : geometry.share_k_ranges) {
        geometry.max_share_tiles = std::max(geometry.max_share_tiles, k_end - k_first);
    }
    // the z ring and the squares slots: a slot a step where L1 has room (no slot reused: no free waits), else fewer
    const uint32_t lowest = std::min<uint32_t>(2, geometry.max_steps);
    uint32_t slots = geometry.max_steps, total = 0;
    for (;; --slots) {
        total = geometry.layout(slots).second;
        if (total <= capacity_bytes || slots == lowest) {
            break;
        }
    }
    TT_FATAL(
        total <= capacity_bytes,
        "moe_tail (engine): its L1 {} KB > {} KB at {} ring slots",
        total >> 10,
        capacity_bytes >> 10,
        lowest);
    geometry.ring_slots = slots;
    geometry.arena_bytes = total;
    return geometry;
}

std::pair<std::vector<uint32_t>, uint32_t> TailEngineGeometry::layout(uint32_t slots) const {
    auto align = [](uint32_t bytes) { return (bytes + 63) / 64 * 64; };
    const std::vector<uint32_t> sizes = {
        k_half_tiles * max_column_tiles * w_tile_bytes,    // up weights
        slots * k_half_tiles * kBf16TileBytes,             // ring
        max_steps * max_share_tiles * kBf16TileBytes,      // landing
        slots * engine_columns * kFp32TileBytes,           // row squares
        kFp32TileBytes,                                    // my squares
        kBf16TileBytes,                                    // ones
        finished_ring_pages * kFp32TileBytes,              // finished
        2 * kExchangeTiles * kFp32TileBytes,               // exchange send
        2 * kFp32TileBytes,                                // sum of squares
        max_steps * kExchangeTiles * kFp32TileBytes,       // exchange receive
        finished_ring_pages * kBf16TileBytes,              // shared expert
        finished_ring_pages * kBf16TileBytes,              // output
        max_own_steps * max_share_tiles * kBf16TileBytes,  // own rows
    };
    std::vector<uint32_t> offsets;
    uint32_t offset = 0;
    for (uint32_t region_bytes : sizes) {
        offsets.push_back(offset);
        offset = align(offset + region_bytes);
    }
    return {offsets, offset};
}

uint32_t TailEngineGeometry::region_tiles() const {
    uint32_t max_range_tiles = 0;
    for (const auto& [k_first, k_end] : w_k_ranges) {
        max_range_tiles = std::max(max_range_tiles, k_end - k_first);
    }
    return max_range_tiles * max_column_tiles;
}

std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> engine_watchers(
    uint32_t position, uint32_t tp_size, bool ring, const TailEngineGeometry& geometry, uint32_t group_columns) {
    std::set<Unit> units;  // (every block landed here by a port: from either neighbour)
    for (uint32_t sender = 0; sender < tp_size; ++sender) {
        const auto [forward, backward] = gather_schedule(sender, tp_size, ring);
        if (row_neighbor(sender, tp_size, ring, TailDirection::Forward) == position) {
            units.insert(forward.begin(), forward.end());
        }
        if (row_neighbor(sender, tp_size, ring, TailDirection::Backward) == position) {
            units.insert(backward.begin(), backward.end());
        }
    }
    const auto order = chunk_order(position, geometry.num_chunks, ring);
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> out;
    for (const auto& [source_position, part] : units) {
        const uint32_t first_column = part == 2 ? geometry.engine_columns / 2 : 0;
        const uint32_t end_column = part == 1 ? geometry.engine_columns / 2 : geometry.engine_columns;
        const uint32_t chunk_position =
            static_cast<uint32_t>(std::find(order.begin(), order.end(), source_position) - order.begin());
        for (uint32_t chunk_row_tile = 0; chunk_row_tile < geometry.row_tiles_per_chunk; ++chunk_row_tile) {
            const uint32_t row_tile = chunk_position * geometry.row_tiles_per_chunk + chunk_row_tile,
                           pair = row_tile % geometry.num_pairs, step = row_tile / geometry.num_pairs;
            for (uint32_t group_first_column = first_column; group_first_column < end_column;
                 group_first_column += group_columns) {
                const uint32_t group_end_column = std::min(group_first_column + group_columns, end_column);
                for (uint32_t half = 0; half < 2; ++half) {
                    if (group_end_column - group_first_column > 1) {
                        out[{group_end_column - 1, pair + geometry.num_pairs * half}].push_back(
                            step | group_first_column << 8 | group_end_column << 16);
                    }
                }
            }
        }
    }
    for (auto& [watcher_core, watched_steps] : out) {
        TT_FATAL(
            watched_steps.size() <= std::min<uint32_t>(geometry.max_steps, 32),
            "moe_tail (engine): {} watched steps",
            watched_steps.size());
        std::sort(watched_steps.begin(), watched_steps.end());
    }
    return out;
}

std::shared_ptr<const TailEnginePlan> make_tail_engine_plan(
    MoEBlockResources& resources,
    uint32_t rows,
    uint32_t latent_dim,
    uint32_t output_dim,
    uint32_t w_tile_bytes,
    uint32_t capacity_bytes) {
    auto& mesh = resources.mesh_device();
    const auto& topology = resources.topology();
    const auto grid = mesh.compute_with_storage_grid_size();
    // (the engine: every row of the grid, an even count of them, and every column but the last)
    const uint32_t engine_rows = grid.y / 2 * 2;
    TT_FATAL(
        grid.x >= 2 && engine_rows >= 4,
        "moe_tail (engine): a {}x{} grid (at least 2 columns, 4 rows)",
        grid.x,
        grid.y);
    TT_FATAL(
        resources.config().num_links == 2,
        "moe_tail (engine): {} links (a link a K half)",
        resources.config().num_links);
    const uint32_t engine_columns = grid.x - 1;  // (the last column: the gather's ports)
    auto plan = std::make_shared<TailEnginePlan>();
    plan->geometry = TailEngineGeometry::make(
        rows, latent_dim, output_dim, topology.C, engine_columns, engine_rows, w_tile_bytes, capacity_bytes);
    plan->topology = topology;
    plan->ring = topology.wrap_x;
    plan->num_links = 2;
    plan->num_dram_banks = mesh.dram_grid_size().x;
    plan->tiles_per_packet =
        std::max<uint32_t>(1, static_cast<uint32_t>(resources.fabric_max_payload_bytes() / kBf16TileBytes));
    plan->norm_eps = resources.dims().norm_eps;
    plan->links = row_links(resources.mesh_plan());
    // (the ports' column has no row with room for a placement search: placed down it - forward above, backward below,
    // a link apart)
    const std::array<uint32_t, 4> port_rows = {
        engine_rows / 10, 3 * engine_rows / 10, 6 * engine_rows / 10, 8 * engine_rows / 10};
    std::vector<CoreRange> ranges;
    for (uint32_t direction_index = 0; direction_index < 2; ++direction_index) {
        for (uint32_t link = 0; link < 2; ++link) {
            const CoreCoord core(engine_columns, port_rows[2 * direction_index + link]);
            plan->ports[{direction_index == 0 ? TailDirection::Forward : TailDirection::Backward, link}] = core;
            ranges.emplace_back(core, core);
        }
    }
    plan->port_cores = CoreRangeSet(ranges);
    plan->engine_cores = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(engine_columns - 1, engine_rows - 1)));
    const auto& geometry = plan->geometry;
    // (a port core holds the landing CB (its base names the engine's landing slots), its staging (three groups: one
    // sending, one landing, one being issued), the group counts and the poll words)
    const uint32_t port_bytes = geometry.max_steps * geometry.max_share_tiles * kBf16TileBytes +
                                3 * kPacketsPerGroup * plan->tiles_per_packet * kBf16TileBytes + 64 + 16 +
                                geometry.row_tiles_per_chunk * geometry.engine_columns * 16;
    TT_FATAL(
        port_bytes <= capacity_bytes,
        "moe_tail (engine): a port core's L1 {} KB > {} KB",
        port_bytes >> 10,
        capacity_bytes >> 10);
    return plan;
}

std::map<std::pair<ChipCoord, TailDirection>, std::vector<uint32_t>> row_links(const MeshPlan& plan) {
    std::map<std::pair<ChipCoord, TailDirection>, std::vector<uint32_t>> out;
    for (const auto& [chip, ports] : plan.chips) {
        for (TailDirection direction : {TailDirection::Forward, TailDirection::Backward}) {
            if (const auto& port = ports[static_cast<size_t>(mesh_direction(direction))]) {
                out[{chip, direction}] = port->links;
            }
        }
    }
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::moe_tail
