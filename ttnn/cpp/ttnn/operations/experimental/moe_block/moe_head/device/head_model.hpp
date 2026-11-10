// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The MoE head's model of one call on the mesh (HeadModel): the gather rings and their ports, the routing exchange's
// senders, the shared expert's sizing and reduce-scatter plan, the router and planner cores, the arena regions.
// Built once a configuration (make_head_model, head_model.cpp). Internal to moe_head.
#pragma once

#include "head_program_common.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

// ---------------------------------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------------------------------

// A gather ring a chip.
struct GatherRing {
    uint32_t position = 0;
    std::optional<Coord> next, previous;
    std::array<std::vector<Coord>, 2> schedule;  // [forward, backward]: (position, part)
    std::array<std::vector<uint32_t>, 2> links;  // fabric link indices a direction (empty: no port traffic)
    const std::optional<Coord>& peer(XDirection direction) const {
        return direction == XDirection::Forward ? next : previous;
    }
};

struct GatherChip {
    GatherRing gather_ring;
    std::vector<std::tuple<XDirection, uint32_t, CoreCoord>> ports;  // (direction, link, logical core), plan order
    std::vector<CoreCoord> copy;
    CoreCoord port(XDirection direction, uint32_t link) const {
        for (const auto& [port_direction, port_link, core] : ports) {
            if (port_direction == direction && port_link == link) {
                return core;
            }
        }
        TT_THROW("moe_head: no port ({}, {})", static_cast<int>(direction), link);
    }
};

// A routing exchange sender key: ("x", direction, link) or ("y", 2 | 3), ordered x (bwd < fwd, then link) < y 2 < y 3.
struct SendKey {
    bool y = false;
    uint32_t direction_code = 0;  // x: 0 bwd / 1 fwd (sort order); y: 2 | 3
    uint32_t link = 0;
    auto operator<=>(const SendKey&) const = default;
    XDirection direction() const { return direction_code == 1 ? XDirection::Forward : XDirection::Backward; }
    static SendKey x(XDirection direction, uint32_t link) {
        return {false, direction == XDirection::Forward ? 1u : 0u, link};
    }
};

class HeadModel {
public:
    HeadProgramConfig config;
    MeshDevice* mesh = nullptr;
    uint32_t mesh_rows = 0, mesh_columns = 0, num_chips = 0, tp_size = 0;
    CoreCoord grid;
    uint32_t mover_column = 0;  // the mover column
    bool ring = false, wrap_y = false;
    uint32_t gather_links = 0;  // links (the gather's)
    uint32_t payload = 0;
    std::optional<ReduceScatterPlan> reduce_scatter;
    std::optional<SharedExpertSizing> sizing;
    // gather
    uint32_t k_shard_tiles = 0, k_steps_per_shard = 0, tiles_per_packet = 0, group = 4;
    std::map<Coord, GatherChip> chips;
    CoreCoord watcher;
    std::map<std::pair<uint32_t, uint32_t>, XDirection> ties;  // (from, to) of a ring's far side -> the fabric's way
    // router
    std::vector<CoreCoord> router_cores;
    // the routing planner's cores: the router cores, then (row blocks) planner-only cores - an instance on
    // the core and RISC instance_slot(instance) gives
    std::vector<CoreCoord> planner_cores;
    uint32_t num_planner_only_cores = 0;  // planner-only cores (planner_cores past the router cores)
    // an instance's planner core (planner_cores index) and RISC (head_sizing planner_slot)
    std::pair<uint32_t, uint32_t> instance_slot(uint32_t instance) const {
        const auto slot = planner_slot(instance, u32(router_cores.size()), num_planner_only_cores);
        return {slot.router ? slot.core : u32(router_cores.size()) + slot.core, slot.risc};
    }
    // the instance on planner_cores[core]'s RISC risc, or -1
    int32_t instance_at(uint32_t core, uint32_t risc) const {
        for (uint32_t instance = 0; instance < planner_instances; ++instance) {
            if (instance_slot(instance) == std::make_pair(core, risc)) {
                return int32_t(instance);
            }
        }
        return -1;
    }
    // instance 0's core: the routing packets' arrival counters, the planners' parts
    CoreCoord first_planner_core() const { return planner_cores[instance_slot(0).first]; }
    uint32_t router_row_tiles = 0, router_k_tiles = 0, expert_tiles = 0, top_k = 0, router_cores_per_link = 1;
    // router iterations (HeadLayout.router_iterations): the router stage over router_row_tiles_per_iteration =
    // router_row_tiles / router_iterations of my owned row tiles at a time, one router core each
    uint32_t router_iterations = 1, router_row_tiles_per_iteration = 0;
    Regions router_offsets;
    int64_t router_bytes = 0, logits_partials_bytes = 0;
    // the mailbox words
    uint32_t handoff_bytes = 16, logits_words_offset = 0, port_words = 16, zone_words_offset = 0, mailbox_words = 0;
    // routing
    uint32_t tokens_per_chip = 0, tokens_per_chip_all_blocks = 0, row_blocks = 1, splits = 1, experts_per_chip = 0,
             accumulators = 0, routing_links = 2, planner_instances = 0, rows_capacity = 0, entries_capacity = 0;
    int64_t routing_header_padded_bytes = 0, expert_count_block_bytes = 0, instance_block_bytes = 0,
            routing_slot_capacity_bytes = 0, max_payload_bytes = 0;
    std::vector<int64_t> planner_bases;  // the planner instances' regions, arena offsets
    // row blocks: an earlier block's planner (no token): its instances' regions past the router's bytes, its caps
    std::vector<int64_t> idle_planner_bases;
    uint32_t idle_rows_capacity = 0, idle_entries_capacity = 0;
    int64_t port_routing_offset = 0;
    int64_t x_gather_cb_offset = 0,
            x_gather_cb_bytes = 0;  // the x-gather packet staging on the port cores (kXGatherCb), arena offset / bytes
    uint32_t max_x_packets = 0, max_y_packets = 0;
    PortLuts luts;
    std::map<uint32_t, std::map<SendKey, std::vector<uint32_t>>> serves;  // per chip index
    int64_t arena_bytes = 0;  // the arena's bytes a core (the engine's, grown to the router cores')
    int64_t args_offset = 0;  // the kernels' args region in the arena, past its regions (head_args_to_dram)

    CoreCoord virtual_core(const CoreCoord& logical) const { return mesh->worker_core_from_logical_core(logical); }
    CoreCoord engine_virtual_core(uint32_t x, uint32_t y) const { return virtual_core(CoreCoord(x, y)); }
    tt::tt_fabric::FabricNodeId node(const Coord& chip_coord) const {
        return mesh->get_fabric_node_id(mesh_coord(ChipCoord{chip_coord.first, chip_coord.second}, topology()));
    }
    // the block's coordinates (SP rows x TP columns; transposed: SP on mesh axis 1)
    MeshTopology topology() const {
        return MeshTopology{
            .R = mesh_rows, .C = mesh_columns, .wrap_x = ring, .wrap_y = wrap_y, .transposed = config.shape.transposed};
    }
    tt::tt_fabric::FabricNodeId node_of_index(uint32_t index) const {
        return node({index / mesh_columns, index % mesh_columns});
    }
    const GatherChip& chip(const Coord& chip_coord) const { return chips.at(chip_coord); }
    // head_words' order
    static uint32_t word_index(const std::string& name) {
        static const std::vector<std::string> names = {
            "mbox",
            "arr",
            "xg",
            "rt",
            "slot_words",
            "sfree",
            "ra_c",
            "ra_p",
            "ra_s",
            "yhello",
            "ewords",
            "valid",
            "rit"};
        const auto found = std::find(names.begin(), names.end(), name);
        TT_FATAL(found != names.end(), "moe_head: no persistent words '{}'", name);
        return u32(found - names.begin());
    }
    Coord coord_of_position(const Coord& row_chip, uint32_t ring_position) const {
        for (const auto& [candidate_coord, gather_chip] : chips) {
            if (candidate_coord.first == row_chip.first && gather_chip.gather_ring.position == ring_position) {
                return candidate_coord;
            }
        }
        TT_THROW("moe_head: no chip at position {}", ring_position);
    }
    uint32_t position(const Coord& chip_coord) const { return chips.at(chip_coord).gather_ring.position; }
    // the Y hello (routing_exchange_y_sender): a chip with a remote Y destination sends one to its first Y direction's
    // neighbour (the row the BRISC's fabric connection reaches) and waits for the ones that land on it. The row it
    // sends to, if any:
    std::optional<uint32_t> y_hello_to(const Coord& chip_coord) const {
        const uint32_t my_index = chip_coord.first * mesh_columns + chip_coord.second;
        for (const auto& [key, chip_destinations] : serves.at(my_index)) {
            if (key.y) {
                if (chip_destinations.size() == (chip_destinations.back() == my_index ? 1u : 0u)) {
                    return std::nullopt;  // (no remote destination: no connection, no hello)
                }
                return u32(positive_modulo(int64_t(chip_coord.first) + (key.direction_code == 2 ? 1 : -1), mesh_rows));
            }
        }
        return std::nullopt;
    }
    // the hellos landing on chip chip_coord a call: the rows of its column whose hello goes to its row. A ring (and R
    // 2): one each. A line (R > 2, no wrap): every row's first Y direction is Y+ except the far end's (Y- only) - the
    // near end gets none, the row before the far end two, the others one
    uint32_t y_hellos_in(const Coord& chip_coord) const {
        uint32_t count = 0;
        for (uint32_t row = 0; row < mesh_rows; ++row) {
            count += row != chip_coord.first && y_hello_to({row, chip_coord.second}) == chip_coord.first;
        }
        return count;
    }
    CoreRangeSet engine_cores() const {
        return CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(sizing->engine_columns - 1, sizing->engine_rows - 1)));
    }
    std::vector<CoreCoord> all_ports() const {
        std::set<std::pair<size_t, size_t>> port_cells;
        for (const auto& [_, gather_chip] : chips) {
            for (const auto& [direction, link, core] : gather_chip.ports) {
                port_cells.insert({core.x, core.y});
            }
        }
        std::vector<CoreCoord> out;
        for (const auto& [x, y] : port_cells) {
            out.emplace_back(x, y);
        }
        return out;
    }

    // ---- the x gather ----
    XDirection ring_direction(uint32_t from, uint32_t to) const {
        const uint32_t forward_distance = positive_modulo(int64_t(to) - from, tp_size);
        if (forward_distance != tp_size - forward_distance) {
            return forward_distance < tp_size - forward_distance ? XDirection::Forward : XDirection::Backward;
        }
        return ties.at({from, to});
    }
    struct PortPlan {
        std::optional<Coord> peer, upstream;
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> mine;  // (shard, part, upstream base, packets)
        uint32_t upstream_increments = 0, expected_increments = 0;
        bool has_upstream = false;
    };
    uint32_t chunk_packets(uint32_t link, uint32_t part) const;
    PortPlan port_plan(const Coord& coord, XDirection direction, uint32_t link) const;
    std::pair<
        std::vector<uint32_t>,
        std::map<uint32_t, std::vector<std::tuple<XDirection, uint32_t, uint32_t, uint32_t>>>>
    shard_order(const Coord& coord) const;
    std::vector<uint32_t> ranks(const Coord& coord) const;
    std::vector<uint32_t> tables(const Coord& coord) const;
    // ---- the router cores ----
    std::vector<uint32_t> port_layout(uint32_t link) const;
    // logits_first: the ports send the logits ahead of the x gather (meshes of up to 8 chips: x first only
    // with one router iteration - with several, every iteration's logits wait behind the whole gather's sends and the
    // matmul cores' router phase stalls; traced head, x first vs logits first: M3 2x4 5120 (4 iterations) 1179 vs
    // 1050 us, M3 2560 (2) 623 vs 527, GLM-5.3 5120 (4) 1075 vs 1047)
    bool logits_first() const {
        if (mesh_rows * mesh_columns <= 8) {
            return router_iterations > 1;
        }
        if (config.shape.latent && ring) {
            return false;
        }
        return true;
    }
    XDirection direction_toward(uint32_t from, uint32_t to) const {
        return ring ? ring_direction(from, to) : (to > from ? XDirection::Forward : XDirection::Backward);
    }
    uint32_t hops(uint32_t from, uint32_t to) const {
        return ring ? std::min(
                          positive_modulo(int64_t(to) - from, tp_size), positive_modulo(int64_t(from) - to, tp_size))
                    : (to > from ? to - from : from - to);
    }
    std::vector<uint32_t> destinations(uint32_t from, XDirection direction) const;
    std::vector<uint32_t> arrival_order(uint32_t receiver) const;
    // ---- the routing exchange ----
    std::map<uint32_t, XDirection> x_directions(const Coord& chip_coord) const;
    SendKey sender(uint32_t source_chip, uint32_t destination_chip) const;
    std::map<SendKey, std::vector<uint32_t>> serve(const Coord& chip_coord) const;
    CoreCoord core_of(const Coord& chip_coord, const SendKey& key) const {
        return key.y ? watcher : chips.at(chip_coord).port(key.direction(), key.link);
    }
    int64_t buffer_offset(const Coord& chip_coord, const SendKey& key) const;
};

}  // namespace ttnn::operations::experimental::moe_block::moe_head
