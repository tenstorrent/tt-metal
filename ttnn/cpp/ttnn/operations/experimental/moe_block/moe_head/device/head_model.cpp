// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE head's model (head_model.hpp): HeadModel's members, make_head_model, the gather schedule, the persistent
// words (head_words).
#include "head_model.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

namespace head_program_detail {

// The gather's schedule: (shards toward position + 1, toward position - 1), own first: (position, part)
std::pair<std::vector<Coord>, std::vector<Coord>> gather_schedule(
    uint32_t position, uint32_t tp_size, bool ring, bool balance) {
    std::vector<Coord> forward, backward;
    if (ring && balance && tp_size % 2 == 0 && tp_size >= 4) {
        const uint32_t half = tp_size / 2;
        for (uint32_t i = 0; i < half; ++i) {
            forward.emplace_back(positive_modulo(int64_t(position) - i, tp_size), i == half - 1 ? 1 : 0);
            backward.emplace_back(positive_modulo(int64_t(position) + i, tp_size), i == half - 1 ? 2 : 0);
        }
        return {forward, backward};
    }
    if (ring) {
        const uint32_t forward_count = tp_size / 2, backward_count = tp_size - 1 - forward_count;
        for (uint32_t i = 0; i < forward_count; ++i) {
            forward.emplace_back(positive_modulo(int64_t(position) - i, tp_size), 0);
        }
        for (uint32_t i = 0; i < backward_count; ++i) {
            backward.emplace_back(positive_modulo(int64_t(position) + i, tp_size), 0);
        }
        return {forward, backward};
    }
    if (position < tp_size - 1) {
        for (uint32_t i = 0; i <= position; ++i) {
            forward.emplace_back(position - i, 0);
        }
    }
    if (position > 0) {
        for (uint32_t i = 0; i < tp_size - position; ++i) {
            backward.emplace_back(position + i, 0);
        }
    }
    return {forward, backward};
}

}  // namespace head_program_detail

uint32_t HeadModel::chunk_packets(uint32_t link, uint32_t part) const {
    uint32_t rows = 0;
    for (uint32_t local_row_tile = 0; local_row_tile < sizing->chunk_row_tiles; ++local_row_tile) {
        if ((local_row_tile / 2) % gather_links == link && (part == 0 || (local_row_tile & 1) == part - 1)) {
            ++rows;
        }
    }
    return rows * k_steps_per_shard * ceil_div(sizing->k_step_tiles, tiles_per_packet);
}

HeadModel::PortPlan HeadModel::port_plan(const Coord& coord, XDirection direction, uint32_t link) const {
    auto increments = [](uint32_t packets) {
        return packets / kArrivalIncrementEvery + (packets % kArrivalIncrementEvery ? 1 : 0);
    };
    const auto& gather_ring = chips.at(coord).gather_ring;
    PortPlan plan;
    plan.peer = gather_ring.peer(direction);
    plan.upstream = direction == XDirection::Forward ? gather_ring.previous : gather_ring.next;
    std::vector<Coord> mine = plan.peer ? gather_ring.schedule[static_cast<int>(direction)] : std::vector<Coord>{};
    std::vector<Coord> upstream_schedule =
        plan.upstream ? chips.at(*plan.upstream).gather_ring.schedule[static_cast<int>(direction)]
                      : std::vector<Coord>{};
    std::vector<uint32_t> upstream_base;
    uint32_t total_increments = 0;
    for (const auto& [_, part] : upstream_schedule) {
        upstream_base.push_back(total_increments);
        total_increments += increments(chunk_packets(link, part));
    }
    for (size_t k = 0; k < mine.size(); ++k) {
        const auto [shard, part] = mine[k];
        plan.mine.emplace_back(shard, part, k > 0 ? upstream_base[k - 1] : 0, chunk_packets(link, part));
    }
    plan.upstream_increments = total_increments;
    plan.expected_increments = upstream_schedule.empty() ? 0 : sizing->row_chunks * total_increments;
    plan.has_upstream = !upstream_schedule.empty();
    return plan;
}

std::pair<std::vector<uint32_t>, std::map<uint32_t, std::vector<std::tuple<XDirection, uint32_t, uint32_t, uint32_t>>>>
HeadModel::shard_order(const Coord& coord) const {
    const auto& gather_ring = chips.at(coord).gather_ring;
    const uint32_t own =
        (!gather_ring.schedule[0].empty() ? gather_ring.schedule[0] : gather_ring.schedule[1])[0].first;
    std::map<uint32_t, std::vector<std::tuple<XDirection, uint32_t, uint32_t, uint32_t>>> carry;
    std::vector<uint32_t> order_seen;
    for (XDirection direction : {XDirection::Forward, XDirection::Backward}) {
        const auto& upstream = direction == XDirection::Forward ? gather_ring.previous : gather_ring.next;
        if (!upstream) {
            continue;
        }
        const auto& upstream_schedule = chips.at(*upstream).gather_ring.schedule[static_cast<int>(direction)];
        for (uint32_t j = 0; j < upstream_schedule.size(); ++j) {
            const auto [shard, part] = upstream_schedule[j];
            for (uint32_t link = 0; link < gather_links; ++link) {
                if (chunk_packets(link, part)) {
                    carry[shard].emplace_back(direction, link, j, u32(upstream_schedule.size()));
                }
            }
        }
    }
    std::vector<uint32_t> rest;
    for (const auto& [shard, _] : carry) {
        rest.push_back(shard);
    }
    auto key = [&](uint32_t shard) {
        uint32_t last_index = 0;
        for (const auto& [direction, link, j, count] : carry.at(shard)) {
            last_index = std::max(last_index, j);
        }
        return std::make_pair(last_index, shard);
    };
    std::stable_sort(rest.begin(), rest.end(), [&](uint32_t left, uint32_t right) { return key(left) < key(right); });
    std::vector<uint32_t> out = {own};
    out.insert(out.end(), rest.begin(), rest.end());
    return {out, carry};
}

std::vector<uint32_t> HeadModel::ranks(const Coord& coord) const {
    const auto shards = shard_order(coord).first;
    std::vector<uint32_t> shard_ranks(tp_size, 0);
    for (uint32_t k = 0; k < shards.size(); ++k) {
        shard_ranks[shards[k]] = k;
    }
    return shard_ranks;
}

std::vector<uint32_t> HeadModel::tables(const Coord& coord) const {
    const auto& engine_sizing = *sizing;
    const uint32_t half_k_steps = engine_sizing.num_k_steps / 2;
    std::vector<uint32_t> out = ranks(coord);
    out.insert(out.end(), engine_sizing.family0_columns.begin(), engine_sizing.family0_columns.end());
    out.insert(out.end(), engine_sizing.family1_columns.begin(), engine_sizing.family1_columns.end());
    for (uint32_t column = 0; column < engine_sizing.engine_columns; ++column) {
        const uint32_t family0_steps = engine_sizing.in_family0(column) ? half_k_steps / engine_sizing.family0_size : 0,
                       family1_steps = engine_sizing.in_family1(column) ? half_k_steps / engine_sizing.family1_size : 0;
        out.push_back(family0_steps + family1_steps);
        out.push_back(family0_steps);
    }
    for (uint32_t column = 0; column < engine_sizing.engine_columns; ++column) {
        out.push_back(u32(engine_virtual_core(column, 0).x));
    }
    for (uint32_t core_y = 0; core_y < engine_sizing.engine_rows; ++core_y) {
        out.push_back(u32(engine_virtual_core(0, core_y).y));
    }
    return out;
}

std::vector<uint32_t> HeadModel::port_layout(uint32_t link) const {
    std::vector<uint32_t> out;
    for (uint32_t j = 0; j < router_cores.size(); ++j) {
        if (j % gather_links == link) {
            out.push_back(j);
        }
    }
    return out;
}

std::vector<uint32_t> HeadModel::destinations(uint32_t from, XDirection direction) const {
    std::vector<uint32_t> targets;
    for (uint32_t to = 0; to < tp_size; ++to) {
        if (to != from && direction_toward(from, to) == direction) {
            targets.push_back(to);
        }
    }
    std::stable_sort(targets.begin(), targets.end(), [&](uint32_t left, uint32_t right) {
        return hops(from, left) > hops(from, right);
    });
    return targets;
}

std::vector<uint32_t> HeadModel::arrival_order(uint32_t receiver) const {
    std::vector<uint32_t> sources;
    for (uint32_t source = 0; source < tp_size; ++source) {
        if (source != receiver) {
            sources.push_back(source);
        }
    }
    auto key = [&](uint32_t source) {
        const auto source_destinations = destinations(source, direction_toward(source, receiver));
        const uint32_t index = u32(
            std::find(source_destinations.begin(), source_destinations.end(), receiver) - source_destinations.begin());
        return std::make_tuple(index, hops(source, receiver), source);
    };
    std::stable_sort(
        sources.begin(), sources.end(), [&](uint32_t left, uint32_t right) { return key(left) < key(right); });
    std::vector<uint32_t> out;
    for (uint32_t source : sources) {
        out.push_back(source < receiver ? source : source - 1);
    }
    return out;
}

std::map<uint32_t, XDirection> HeadModel::x_directions(const Coord& chip_coord) const {
    std::map<uint32_t, XDirection> out;
    const auto& gather_ring = chips.at(chip_coord).gather_ring;
    for (XDirection direction : {XDirection::Forward, XDirection::Backward}) {
        const auto& peer = gather_ring.peer(direction);
        if (peer && !gather_ring.links[static_cast<int>(direction)].empty()) {
            const bool forward = ring ? positive_modulo(int64_t(peer->second) - chip_coord.second, mesh_columns) == 1
                                      : peer->second > chip_coord.second;
            out[forward ? 0 : 1] = direction;
        }
    }
    return out;
}

// The sender of chip source_chip's packet to chip destination_chip: the one whose fabric connection the fabric's first
// hop from source_chip toward destination_chip leaves by (HeadShape.first_hops, asked of the fabric) - a Y sender, or
// the gather's X port that way (link destination_chip % gather_links). The packet goes in one fabric send, the fabric
// routing every hop to destination_chip (head_sizing.cpp routing_paths: the same).
SendKey HeadModel::sender(uint32_t source_chip, uint32_t destination_chip) const {
    const auto& first_hops = config.shape.first_hops;
    TT_FATAL(
        first_hops.size() == size_t(num_chips) * num_chips,
        "moe_head: the fabric's first hops: {} entries for {} chips",
        first_hops.size(),
        num_chips);
    const uint32_t first_hop = first_hops[size_t(source_chip) * num_chips + destination_chip];
    if (first_hop >= 2) {
        return SendKey{true, first_hop, 0};
    }
    const Coord source_coord{source_chip / mesh_columns, source_chip % mesh_columns};
    const auto port_directions = x_directions(source_coord);
    TT_FATAL(
        port_directions.count(first_hop),
        "moe_head: chip ({}, {}): no X port toward {}",
        source_coord.first,
        source_coord.second,
        destination_chip);
    return SendKey::x(
        port_directions.at(first_hop),
        (destination_chip % routing_links) < gather_links ? destination_chip % routing_links : 0);
}

std::map<SendKey, std::vector<uint32_t>> HeadModel::serve(const Coord& chip_coord) const {
    const uint32_t my_index = chip_coord.first * mesh_columns + chip_coord.second;
    std::map<SendKey, std::vector<uint32_t>> out;
    for (uint32_t destination = 0; destination < num_chips; ++destination) {
        if (destination != my_index) {
            out[sender(my_index, destination)].push_back(destination);
        }
    }
    std::vector<SendKey> y_keys;
    for (const auto& [key, _] : out) {
        if (key.y) {
            y_keys.push_back(key);
        }
    }
    out[y_keys.empty() ? out.begin()->first : y_keys[0]].push_back(my_index);
    TT_FATAL(
        y_keys.size() <= 2,
        "moe_head: chip ({}, {}): {} Y directions",
        chip_coord.first,
        chip_coord.second,
        y_keys.size());
    return out;
}

int64_t HeadModel::buffer_offset(const Coord& chip_coord, const SendKey& key) const {
    if (!key.y) {
        return port_routing_offset;
    }
    const auto& chip_serves = serves.at(chip_coord.first * mesh_columns + chip_coord.second);
    uint32_t y_index = 0;
    for (const auto& [other_key, _] : chip_serves) {
        if (other_key.y) {
            if (other_key == key) {
                return int64_t(y_index) * max_y_packets * routing_slot_capacity_bytes;
            }
            ++y_index;
        }
    }
    TT_THROW("moe_head: no Y sender");
}

std::shared_ptr<const HeadModel> make_head_model(
    MeshDevice& mesh, const HeadProgramConfig& config, const PortLuts& luts) {
    auto model_pointer = std::make_shared<HeadModel>();
    HeadModel& model = *model_pointer;
    model.config = config;
    model.mesh = &mesh;
    const HeadShape& shape = config.shape;
    const HeadLayout& layout = config.layout;
    model.mesh_rows =
        shape.transposed ? mesh.shape()[1] : mesh.shape()[0];  // (SP rows, TP columns: the block's coordinates)
    model.mesh_columns = shape.transposed ? mesh.shape()[0] : mesh.shape()[1];
    model.num_chips = model.mesh_rows * model.mesh_columns;
    model.tp_size = shape.tp_size;
    TT_FATAL(
        model.mesh_columns == shape.tp_size && model.mesh_rows == shape.sp_rows,
        "moe_head: the plan's mesh {}x{} is not the device's {}x{}",
        shape.sp_rows,
        shape.tp_size,
        model.mesh_rows,
        model.mesh_columns);
    model.grid = mesh.compute_with_storage_grid_size();
    model.mover_column = std::min<uint32_t>(11, model.grid.x - 1);
    TT_FATAL(
        model.grid.x - model.mover_column == 1,
        "moe_head: the movers are one column (grid {}x{})",
        model.grid.x,
        model.grid.y);
    model.ring = shape.ring && shape.tp_size > 2;
    model.wrap_y = shape.sp_ring && shape.sp_rows > 2;
    model.gather_links = shape.links;
    model.payload = u32(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    model.luts = luts;
    const uint32_t tp_size = model.tp_size, engine_columns = layout.engine_columns, engine_rows = layout.engine_rows,
                   row_tiles_per_core = layout.row_tiles_per_core;

    // the engine: the reduce-scatter plan and the shared expert's sizing at the planned layout
    model.reduce_scatter.emplace(
        tp_size,
        model.ring,
        engine_columns,
        engine_rows,
        row_tiles_per_core,
        shape.hidden_dim / 32,
        shape.reduce_scatter_dim(),
        shape.links,
        kDestTiles / row_tiles_per_core,
        layout.compact_reduce_scatter_zones);
    SharedExpertArgs sizing_args;
    sizing_args.rows = shape.rows;
    sizing_args.hidden_dim = shape.hidden_dim;
    sizing_args.shared_expert_dim = shape.shared_expert_dim;
    sizing_args.engine_columns = engine_columns;
    sizing_args.engine_rows = engine_rows;
    sizing_args.row_tiles_per_core = row_tiles_per_core;
    sizing_args.k_step_tiles = layout.k_step_tiles;
    sizing_args.shards = tp_size;
    sizing_args.reduce_scatter = &*model.reduce_scatter;
    if (shape.latent) {
        sizing_args.latent = *shape.latent / 32 / tp_size;
    }
    sizing_args.router = std::make_pair(
        shape.router_candidates / 32, shape.hidden_dim / tp_size / 32);  // (the router logits on the matmul cores)
    sizing_args.router_iterations = layout.router_iterations;
    sizing_args.arena_max = arena_cap_of(config.l1_free, config.cb_floor);
    sizing_args.w_dtype = shape.w_dtype;
    model.sizing.emplace(sizing_args);
    const SharedExpertSizing& sizing = *model.sizing;

    // the movers: ports, the watcher, the router cores - a router core a row tile of one router iteration
    model.router_iterations = layout.router_iterations;
    const uint32_t router_cores_needed = shape.rows / 32 / tp_size / layout.router_iterations;
    if (router_cores_needed + 2 * model.gather_links + 1 <= model.grid.y) {
        for (uint32_t core_y = model.grid.y - router_cores_needed; core_y < model.grid.y; ++core_y) {
            model.router_cores.emplace_back(model.mover_column, core_y);
        }
        model.watcher = CoreCoord(model.mover_column, model.grid.y - router_cores_needed - 1);
    } else {
        std::vector<CoreCoord> free_cells;
        for (uint32_t y = engine_rows; y < model.grid.y; ++y) {
            for (uint32_t column = 0; column < engine_columns; ++column) {
                free_cells.emplace_back(column, y);
            }
        }
        TT_FATAL(
            router_cores_needed <= free_cells.size(), "moe_head: no cells for {} router cores", router_cores_needed);
        model.router_cores.assign(free_cells.begin(), free_cells.begin() + router_cores_needed);
        model.watcher = CoreCoord(model.mover_column, model.grid.y - 1);
    }

    // the gather: the TP rings, simple placement in the mover column
    model.k_shard_tiles = sizing.hidden_tiles / tp_size;
    model.k_steps_per_shard = model.k_shard_tiles / sizing.k_step_tiles;
    model.tiles_per_packet =
        std::max<uint32_t>(1, std::min<uint32_t>(sizing.k_step_tiles, model.payload / kBf16TileBytes));
    for (uint32_t row = 0; row < model.mesh_rows; ++row) {
        for (uint32_t ring_position = 0; ring_position < tp_size; ++ring_position) {
            GatherChip gather_chip;
            gather_chip.gather_ring.position = ring_position;
            if (model.ring || ring_position < tp_size - 1) {
                gather_chip.gather_ring.next = Coord{row, u32(positive_modulo(ring_position + 1, tp_size))};
            }
            if (model.ring || ring_position > 0) {
                gather_chip.gather_ring.previous =
                    Coord{row, u32(positive_modulo(int64_t(ring_position) - 1, tp_size))};
            }
            auto [forward, backward] = gather_schedule(ring_position, tp_size, model.ring, model.ring);
            gather_chip.gather_ring.schedule = {forward, backward};
            model.chips[{row, ring_position}] = gather_chip;
        }
    }
    for (auto& [coord, gather_chip] : model.chips) {
        for (XDirection direction : {XDirection::Forward, XDirection::Backward}) {
            const auto& peer = gather_chip.gather_ring.peer(direction);
            if (!peer) {
                continue;
            }
            const bool needs_ready =
                !model.chips.at(*peer).gather_ring.schedule[static_cast<int>(opposite(direction))].empty();
            if (gather_chip.gather_ring.schedule[static_cast<int>(direction)].empty() && !needs_ready) {
                continue;
            }
            auto links = tt::tt_fabric::get_forwarding_link_indices(mesh, model.node(coord), model.node(*peer));
            TT_FATAL(
                links.size() >= model.gather_links,
                "moe_head: ({}, {}) -> ({}, {}) has {} usable link(s), {} requested",
                coord.first,
                coord.second,
                peer->first,
                peer->second,
                links.size(),
                model.gather_links);
            links.resize(model.gather_links);
            gather_chip.gather_ring.links[static_cast<int>(direction)] = links;
        }
    }
    {
        std::set<std::pair<size_t, size_t>> allowed;
        for (uint32_t y = 0; y < model.grid.y; ++y) {
            allowed.insert({model.mover_column, y});
        }
        for (auto& [coord, gather_chip] : model.chips) {
            std::set<std::pair<size_t, size_t>> taken;
            auto below = [&]() {  // (at NoC column 0: the free cell of the lowest virtual y)
                std::optional<CoreCoord> best;
                for (const auto& [x, y] : allowed) {
                    if (taken.count({x, y})) {
                        continue;
                    }
                    const CoreCoord cell(x, y);
                    if (!best || model.virtual_core(cell).y < model.virtual_core(*best).y) {
                        best = cell;
                    }
                }
                TT_FATAL(best.has_value(), "moe_head: the mover column has too few cores for the ports");
                taken.insert({best->x, best->y});
                return *best;
            };
            for (XDirection direction : {XDirection::Forward, XDirection::Backward}) {
                for (uint32_t link = 0; link < model.gather_links; ++link) {
                    gather_chip.ports.emplace_back(direction, link, below());
                }
            }
            std::vector<uint32_t> column_rows;
            for (uint32_t y = 0; y < model.grid.y; ++y) {
                column_rows.push_back(y);
            }
            const uint32_t middle = column_rows[column_rows.size() / 2];
            std::vector<uint32_t> walk;
            for (uint32_t y : column_rows) {
                if (y >= middle) {
                    walk.push_back(y);
                }
            }
            for (uint32_t y : column_rows) {
                if (y < middle) {
                    walk.push_back(y);
                }
            }
            for (uint32_t link = 0; link < model.gather_links; ++link) {
                for (uint32_t y : walk) {
                    if (!taken.count({model.mover_column, y})) {
                        gather_chip.copy.emplace_back(model.mover_column, y);
                        taken.insert({model.mover_column, y});
                        break;
                    }
                }
            }
            TT_FATAL(
                gather_chip.copy.size() == model.gather_links,
                "moe_head: the mover column has too few cores for the copy cores");
        }
    }
    // (the ring's far side: the way the fabric's routing sends a packet: the ties ring_direction reads)
    if (model.ring && tp_size % 2 == 0) {
        for (const auto& [coord, gather_chip] : model.chips) {
            const auto next_direction = tt::tt_fabric::get_eth_forwarding_direction(
                model.node(coord), model.node(*gather_chip.gather_ring.next));
            for (const auto& [other_coord, other_chip] : model.chips) {
                if (other_coord.first != coord.first ||
                    positive_modulo(
                        int64_t(other_chip.gather_ring.position) - gather_chip.gather_ring.position, tp_size) *
                            2 !=
                        tp_size) {
                    continue;
                }
                const XDirection direction = tt::tt_fabric::get_eth_forwarding_direction(
                                                 model.node(coord), model.node(other_coord)) == next_direction
                                                 ? XDirection::Forward
                                                 : XDirection::Backward;
                auto [tie, inserted] = model.ties.emplace(
                    std::make_pair(gather_chip.gather_ring.position, other_chip.gather_ring.position), direction);
                TT_FATAL(inserted || tie->second == direction, "moe_head: ring ties differ between mesh rows");
            }
        }
    }

    // the arena: the engine's, grown to what the router cores need
    model.arena_bytes = sizing.arena_bytes;
    const int64_t need = align_to_64(router_min_bytes(tp_size, shape.router_candidates, shape.top_k));
    if (model.arena_bytes < need && need <= sizing.args.arena_max) {
        model.arena_bytes = need;
    }

    // the mailbox words
    model.handoff_bytes = config.handoff_bytes;
    model.logits_words_offset = (2 * engine_columns * engine_rows * model.handoff_bytes / 4 + 4 * 8 + 3) / 4 * 4;
    model.port_words = std::max<uint32_t>(16, (tp_size - 1) * u32(ceil_div(router_cores_needed, model.gather_links)));
    model.zone_words_offset = model.logits_words_offset + 4 * model.port_words;
    model.mailbox_words = model.zone_words_offset + 16;

    // the router cores
    model.router_row_tiles = shape.rows / 32 / tp_size;
    TT_FATAL(
        model.router_row_tiles % model.router_iterations == 0,
        "moe_head: {} router iterations do not divide {} row tiles",
        model.router_iterations,
        model.router_row_tiles);
    model.router_row_tiles_per_iteration = model.router_row_tiles / model.router_iterations;
    model.router_k_tiles = shape.hidden_dim / tp_size / 32;
    model.expert_tiles = shape.router_candidates / 32;
    model.top_k = shape.top_k;
    TT_FATAL(
        model.router_cores.size() >= model.router_row_tiles_per_iteration,
        "moe_head: {} router cores < {}",
        model.router_cores.size(),
        model.router_row_tiles_per_iteration);
    model.router_cores.resize(model.router_row_tiles_per_iteration);
    {
        std::tie(model.router_offsets, model.router_bytes) =
            router_layout(tp_size, model.expert_tiles, model.top_k, model.expert_tiles, shape.row_blocks > 1);
        TT_FATAL(
            model.router_bytes <= model.arena_bytes,
            "moe_head: router cores need {} B of the arena's {}",
            model.router_bytes,
            model.arena_bytes);
        model.router_cores_per_link = u32(ceil_div(model.router_cores.size(), model.gather_links));
        model.logits_partials_bytes =
            int64_t(tp_size - 1) * model.router_cores_per_link * model.expert_tiles * kFp32TileBytes;
    }

    // the routing exchange
    model.tokens_per_chip = shape.rows / tp_size;
    model.tokens_per_chip_all_blocks =
        shape.tokens_per_chip_all_blocks();  // (the planner and the exchange: every row block's tokens)
    model.row_blocks = shape.row_blocks;
    model.splits = shape.splits;
    model.experts_per_chip = shape.experts_per_chip();
    model.accumulators = shape.accumulators;
    model.rows_capacity =
        u32(routing_worst_rows(model.num_chips, model.tokens_per_chip_all_blocks, model.top_k, model.experts_per_chip));
    {
        const uint32_t router_core_count = u32(model.router_cores.size());
        // (row blocks: planner-only cores in the cells no head piece takes)
        const auto free_cells = head_free_cells(
            model.grid.x,
            model.grid.y,
            sizing.engine_columns,
            sizing.engine_rows,
            router_core_count + 2 * model.gather_links + 1 > model.grid.y ? router_core_count : 0);
        const uint32_t num_planner_only = planner_only_cores(model.row_blocks, free_cells.size());
        model.planner_cores = model.router_cores;
        model.num_planner_only_cores = num_planner_only;
        for (uint32_t i = 0; i < num_planner_only; ++i) {
            model.planner_cores.emplace_back(free_cells[i].first, free_cells[i].second);
        }
        const int64_t bias =
            std::get<1>(*std::find_if(model.router_offsets.begin(), model.router_offsets.end(), [](const auto& region) {
                return region.first == "bias";
            }));
        const auto placement = place_planner(
            model.num_chips,
            model.tokens_per_chip_all_blocks,
            model.top_k,
            model.experts_per_chip,
            model.accumulators,
            router_core_count,
            bias,
            model.router_bytes,
            model.arena_bytes,
            0,
            num_planner_only);
        TT_FATAL(placement.ok, "moe_head: routing planner: no room ({})", placement.sizes.reason);
        model.planner_instances = placement.planner_instances;
        model.entries_capacity = u32(placement.sizes.entries_capacity);
        for (uint32_t instance = 0; instance < model.planner_instances; ++instance) {
            model.planner_bases.push_back(placement.planner_base(instance));
        }
        if (model.row_blocks >
            1) {  // (an earlier row block's planner: the same instances, no token, past the router's bytes)
            const int64_t arrivals_offset = std::get<1>(
                *std::find_if(model.router_offsets.begin(), model.router_offsets.end(), [](const auto& region) {
                    return region.first == "arr";
                }));
            const auto idle_placement = place_planner(
                model.num_chips,
                kPlannerIdleTokens,
                model.top_k,
                model.experts_per_chip,
                model.accumulators,
                router_core_count,
                arrivals_offset,
                model.router_bytes,
                model.arena_bytes,
                model.planner_instances,
                num_planner_only);
            TT_FATAL(
                idle_placement.ok,
                "moe_head: an earlier row block's routing planner: no room ({})",
                idle_placement.sizes.reason);
            model.idle_rows_capacity =
                u32(routing_worst_rows(model.num_chips, kPlannerIdleTokens, model.top_k, model.experts_per_chip));
            model.idle_entries_capacity = u32(idle_placement.sizes.entries_capacity);
            for (uint32_t instance = 0; instance < model.planner_instances; ++instance) {
                model.idle_planner_bases.push_back(idle_placement.planner_base(instance));
            }
        }
    }
    model.routing_header_padded_bytes = routing_header_bytes(model.experts_per_chip, model.planner_instances);
    model.expert_count_block_bytes = align_to_16(2 * int64_t(model.experts_per_chip));
    model.instance_block_bytes = align_to_16(2 * (int64_t(model.experts_per_chip) + 1));
    model.routing_slot_capacity_bytes = routing_slot_bytes(
        model.tokens_per_chip_all_blocks, model.top_k, model.experts_per_chip, model.planner_instances);
    model.max_payload_bytes = int64_t(model.payload) / 64 * 64;
    for (const auto& [coord, _] : model.chips) {
        model.serves[coord.first * model.mesh_columns + coord.second] = model.serve(coord);
    }
    for (const auto& [coord, _] :
         model.chips) {  // (routing_exchange_y_sender waits only when it sends: no hello may land elsewhere)
        TT_FATAL(
            model.y_hello_to(coord) || !model.y_hellos_in(coord),
            "moe_head: chip ({}, {}): a Y hello lands on a chip that sends none",
            coord.first,
            coord.second);
    }
    for (const auto& [_, chip_serves] : model.serves) {
        for (const auto& [key, targets] : chip_serves) {
            (key.y ? model.max_y_packets : model.max_x_packets) =
                std::max<uint32_t>(key.y ? model.max_y_packets : model.max_x_packets, u32(targets.size()));
        }
    }
    model.port_routing_offset = align_to_64(model.logits_partials_bytes);
    // (the arena grows to the ports' logits partials + routing buffers, and to the watcher's Y buffers, up to its cap -
    // head_plan's cores_check)
    for (const int64_t grown :
         {model.port_routing_offset + int64_t(model.max_x_packets) * model.routing_slot_capacity_bytes,
          2 * int64_t(model.max_y_packets) * model.routing_slot_capacity_bytes}) {
        if (model.arena_bytes < grown && grown <= sizing.args.arena_max) {
            model.arena_bytes = align_to_64(grown);
        }
    }
    TT_FATAL(
        model.port_routing_offset + int64_t(model.max_x_packets) * model.routing_slot_capacity_bytes <=
            model.arena_bytes,
        "moe_head: ports: no L1 for the routing buffers");
    // the x-gather's packet staging (kXGatherCb): an arena region on the port cores, after their routing buffers (the
    // ports hold nothing else in the arena: logits partials [0, logits_partials_bytes), routing [port_routing_offset, +
    // max_x_packets routing_slot_capacity_bytes))
    model.x_gather_cb_bytes = 2 * int64_t(model.group) * model.tiles_per_packet * kBf16TileBytes;
    model.x_gather_cb_offset =
        align_to_64(model.port_routing_offset + int64_t(model.max_x_packets) * model.routing_slot_capacity_bytes);
    if (model.arena_bytes < model.x_gather_cb_offset + model.x_gather_cb_bytes &&
        model.x_gather_cb_offset + model.x_gather_cb_bytes <= sizing.args.arena_max) {
        model.arena_bytes = align_to_64(model.x_gather_cb_offset + model.x_gather_cb_bytes);
    }
    TT_FATAL(
        model.x_gather_cb_offset + model.x_gather_cb_bytes <= model.arena_bytes,
        "moe_head: ports: no L1 for the x-gather staging ({} B at {} of the arena's {})",
        model.x_gather_cb_bytes,
        model.x_gather_cb_offset,
        model.arena_bytes);
    TT_FATAL(
        2 * int64_t(model.max_y_packets) * model.routing_slot_capacity_bytes <= model.arena_bytes,
        "moe_head: the watcher core: no L1 for the Y buffers");
    // (the kernels' args blocks past every region: arena_cap_of kept kHeadArgsBytes free above the cap)
    model.args_offset = align_to_64(model.arena_bytes);
    return model_pointer;
}

std::vector<HeadWordsSpec> head_words(const HeadModel& model) {
    const auto ports = cells(model.all_ports());
    auto union_of = [](const std::vector<CoreCoord>& first, const std::vector<CoreCoord>& second) {
        std::vector<CoreCoord> cores = first;
        for (const auto& core : second) {
            if (std::find(cores.begin(), cores.end(), core) == cores.end()) {
                cores.push_back(core);
            }
        }
        return cores;
    };
    const auto ports_and_watcher = union_of(model.all_ports(), {model.watcher});
    std::vector<HeadWordsSpec> out = {
        {"mbox", ports, model.mailbox_words},
        {"arr", model.engine_cores(), 4 * 4},
        {"xg", cells(ports_and_watcher), 4 * (3 + 2 * model.gather_links)},
        {"rt", cells(model.router_cores), 4 * 2},
        {"slot_words", cells(model.router_cores), 4 * std::max<uint32_t>(1, model.tp_size - 1)},
        {"sfree", cells(model.router_cores), 32},
        {"ra_c", cells({model.first_planner_core()}), 4 * 2},
        {"ra_p", cells(model.planner_cores), 4 * 9},
        {"ra_s", cells(ports_and_watcher), 4 * 4},
        {"yhello", cells({model.watcher}), 16},
    };
    out.push_back({"ewords", model.engine_cores(), 64});
    // a padded forward's metadata, read by the kernels that skip its row chunks past the last real one
    // (SKIP_PADDING_CHUNKS: padded_rows.hpp padded_live_chunks): 128 B, 64 B aligned, a core
    out.push_back({"valid", head_arena_cores(model), 48});
    // router iterations: [0] router core 0's count of the chip's router cores / ports done with an iteration, [16] a
    // port's count of the chip's router cores done
    if (model.router_iterations > 1) {
        out.push_back({"rit", cells(union_of(model.router_cores, model.all_ports())), 4 * 2});
    }
    // row blocks in one program: the block words of every core the program runs on (row_block.hpp: 5 done
    // words, the go word, the coordinator's count, 16 B apart)
    if (model.row_blocks > 1) {
        out.push_back({"row_blocks", head_arena_cores(model), kHeadRowBlockWordsBytes / 4});
    }
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head
