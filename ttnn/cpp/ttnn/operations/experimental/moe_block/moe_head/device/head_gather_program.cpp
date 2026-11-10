// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The x gather's port kernels (x, the reduce-scatter's and the latent's blocks, the watcher) and the routing
// exchange's Y senders: ChipProgram::gather, ChipProgram::y_kernels and their arg blocks.
#include "head_chip_program.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

// ---- the x gather (port cores) ----

Args ChipProgram::reduce_scatter_block(XDirection direction, uint32_t link) const {
    const ReduceScatterDirection scatter_direction = reduce_scatter_direction(direction);
    const auto downstream = reduce_scatter.down(position, scatter_direction);
    const auto units = downstream ? reduce_scatter.port_units(position, scatter_direction)
                                  : std::vector<ReduceScatterPlan::PortUnit>{};
    const auto cores = units.empty() ? std::vector<std::pair<uint32_t, uint32_t>>{} : reduce_scatter.port_cores(link);
    Args out = {u32(units.size())};
    for (const auto& unit : units) {
        const uint32_t prefix_mask0 = reduce_scatter.compact_reduce_scatter_zones
                                          ? reduce_scatter.prefix_mask(*downstream, 0, scatter_direction, unit.slot0)
                                          : 0;
        const uint32_t prefix_mask1 = reduce_scatter.compact_reduce_scatter_zones
                                          ? reduce_scatter.prefix_mask(*downstream, 1, scatter_direction, unit.slot1)
                                          : 0;
        out.push_back(
            unit.entry | unit.block << 8 | unit.mask << 16 | prefix_mask0 << 20 | prefix_mask1 << 24 |
            static_cast<uint32_t>(unit.pieces) << 28);
        out.push_back(unit.slot0 | unit.slot1 << 16);
    }
    out.push_back(u32(cores.size()));
    for (const auto& [core_x, core_y] : cores) {
        const auto engine_core = model.engine_virtual_core(core_x, core_y);
        const uint32_t core_index = reduce_scatter.core_index(core_x, core_y);
        uint32_t pass_width_sum = 0;
        if (reduce_scatter.pass_major) {
            for (uint32_t width : reduce_scatter.pass_widths(core_x)) {
                pass_width_sum += width;
            }
        }
        TT_FATAL(
            core_index < 128 && pass_width_sum < 32,
            "moe_head: a core word's k {} < 128, pn {} < 32",
            core_index,
            pass_width_sum);
        out.push_back(
            u32(engine_core.x | engine_core.y << 8 | reduce_scatter.half(core_index) << 16 | core_index << 20 |
                pass_width_sum << 27));
        out.push_back(reduce_scatter.counts_word(core_x, core_y));
    }
    const int direction_index = static_cast<int>(direction);
    auto lands = [&](uint32_t at_position, uint32_t core_x, uint32_t core_y, ReduceScatterDirection arrival_direction) {
        const auto arrivals = reduce_scatter.arrivals(at_position, core_x, core_y);
        for (const auto& scatter_arrival : arrivals[static_cast<int>(arrival_direction)]) {
            if (scatter_arrival.threshold) {
                return true;
            }
        }
        return false;
    };
    const uint32_t router_overlay_flag = sizing.router_overlays_zones ? 1 : 0;
    const ReduceScatterDirection opposite_direction = scatter_direction == ReduceScatterDirection::Forward
                                                          ? ReduceScatterDirection::Backward
                                                          : ReduceScatterDirection::Forward;
    uint32_t zones_needed = 0, zones_expected = 0;
    if (sizing.row_chunks > 1 || router_overlay_flag) {
        for (const auto& [core_x, core_y] : reduce_scatter.port_cores(link)) {
            zones_needed += lands(position, core_x, core_y, opposite_direction) ? 1 : 0;
        }
        if (downstream) {
            bool any = false;
            for (const auto& [core_x, core_y] : reduce_scatter.port_cores(link)) {
                any = any || lands(*downstream, core_x, core_y, scatter_direction);
            }
            zones_expected = any ? sizing.row_chunks - 1 + router_overlay_flag : 0;
        }
    }
    const auto& entries = reduce_scatter.entries(position);
    out.insert(out.end(), entries.begin(), entries.end());
    for (const Arg& value : std::initializer_list<Arg>{
             arena_arg("finb"),
             mailbox,
             direction_index == 0 ? arena_arg("arrf") : arena_arg("arrb"),
             engine_arrival_words[direction_index],
             reduce_scatter.compact_reduce_scatter_zones ? 0u : reduce_scatter.core_block_tiles,
             engine_arrival_words.size() > 3 ? engine_arrival_words[3] : Arg(0u),
             zone_free + 16,
             zone_free,
             zones_needed,
             zones_expected,
             sizing.router_overlay ? zone_free + 32 : 0u,
             sizing.router_overlay ? (sizing.engine_columns * sizing.engine_rows | router_overlay_flag << 16 |
                                      static_cast<uint32_t>(sizing.router_overlays_latent) << 17)
                                   : 0u}) {
        out.push_back(value);
    }
    return out;
}

Args ChipProgram::latent_block(XDirection direction, uint32_t link) const {
    if (!sizing.latent_tiles) {
        return Args{0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    }
    auto latent_direction = [&](uint32_t from, uint32_t to) {
        return reduce_scatter.kind == ReduceScatterKind::Line ? (to > from ? XDirection::Forward : XDirection::Backward)
                                                              : model.ring_direction(from, to);
    };
    std::vector<uint32_t> peer_positions;
    for (uint32_t peer_position = 0; peer_position < model.tp_size; ++peer_position) {
        if (peer_position != position && latent_direction(position, peer_position) == direction) {
            peer_positions.push_back(peer_position);
        }
    }
    TT_FATAL(
        peer_positions.size() <= 3 && model.group + 1 + peer_positions.size() <= 12,
        "moe_head: packet headers: 12 a RISC");
    Args out = {u32(peer_positions.size())};
    for (uint32_t peer_position : peer_positions) {
        const auto destination_node = model.node(model.coord_of_position(coord, peer_position));
        out.push_back(
            peer_position | (position < peer_position ? position : position - 1) << 8 |
            u32(destination_node.chip_id) << 16);
        out.push_back(u32(*destination_node.mesh_id));
    }
    std::vector<uint32_t> core_words;
    uint32_t num_cores = 0;
    for (uint32_t core_y = 0; core_y < sizing.engine_rows; ++core_y) {
        for (uint32_t core_x = 0; core_x < sizing.engine_columns; ++core_x) {
            if ((core_y * sizing.engine_columns + core_x) % model.gather_links != link) {
                continue;
            }
            const auto engine_core = model.engine_virtual_core(core_x, core_y);
            uint32_t rows = 0;
            for (uint32_t owner = 0; owner < model.tp_size; ++owner) {
                const auto [first_row, num_rows] = sizing.latent_rows(core_y, owner);
                rows |= (first_row | num_rows << 2) << (4 * owner);
            }
            core_words.push_back(
                u32(engine_core.x | engine_core.y << 8 | (core_y * sizing.engine_columns + core_x) << 16));
            core_words.push_back(rows);
            ++num_cores;
        }
    }
    out.push_back(num_cores);
    out.insert(out.end(), core_words.begin(), core_words.end());
    const Arg fence = mailbox + model.handoff_bytes * sizing.engine_columns * sizing.engine_rows;
    std::vector<uint32_t> sends;
    for (uint32_t peer_position = 0; peer_position < model.tp_size; ++peer_position) {
        if (peer_position == position || latent_direction(position, peer_position) != direction) {
            continue;
        }
        const Coord peer_coord = model.coord_of_position(coord, peer_position);
        const auto peer_port =
            model.virtual_core(model.chip(peer_coord).port(latent_direction(peer_position, position), link));
        const auto peer_node = model.node(peer_coord);
        sends.push_back(u32(peer_node.chip_id) | u32(*peer_node.mesh_id) << 16);
        sends.push_back(u32(peer_port.x | peer_port.y << 8));
    }
    for (const Arg& value : std::initializer_list<Arg>{
             arena_arg("lat"),
             fence + 128,
             arena_arg("latz"),
             engine_arrival_words[2],
             sizing.max_latent_column_tiles,
             fence,
             u32(peer_positions.size()),
             u32(sends.size() / 2)}) {
        out.push_back(value);
    }
    out.insert(out.end(), sends.begin(), sends.end());
    return out;
}

// The port sender's logits args.
Args ChipProgram::port_block(XDirection direction, uint32_t link) const {
    // (router iterations: my "rit" word - my count of the chip's router cores done at +16, router core 0's count at
    // +0 - router core 0's NoC, the router cores an iteration; every port counts on router core 0, a peer or not)
    Args iteration_args = {0u, 0u, 0u};
    if (model.router_iterations > 1) {
        const auto first_router_core = model.virtual_core(model.router_cores[0]);
        iteration_args = Args{
            word("rit"), u32(first_router_core.x | first_router_core.y << 16), model.router_row_tiles_per_iteration};
    }
    if (!model.chip(coord).gather_ring.peer(direction)) {
        Args out(11, 0u);
        add(out, iteration_args);
        return out;
    }
    const auto destination_positions = model.destinations(position, direction);
    const auto router_core_indices = model.port_layout(link);
    Args out = {u32(destination_positions.size())};
    for (uint32_t destination : destination_positions) {
        const auto destination_node = model.node(model.coord_of_position(coord, destination));
        out.push_back(u32(destination_node.chip_id) | (position < destination ? position : position - 1) << 16);
        out.push_back(u32(*destination_node.mesh_id));
    }
    out.push_back(u32(router_core_indices.size()));
    for (uint32_t router_core : router_core_indices) {
        const auto router_virtual_core = model.virtual_core(model.router_cores[router_core]);
        out.push_back(u32(router_virtual_core.x | router_virtual_core.y << 8));
    }
    const uint32_t tiles_per_payload = std::max<uint32_t>(1, model.payload / kFp32TileBytes);
    TT_FATAL(
        destination_positions.size() * router_core_indices.size() <= model.port_words,
        "moe_head: the port's sub-block words: {} a port",
        model.port_words);
    for (const Arg& value : std::initializer_list<Arg>{
             arena_base,
             mailbox + 4 * model.logits_words_offset,
             router_arena_arg("arr"),
             router_arrival0,
             model.expert_tiles,
             tiles_per_payload,
             model.expert_tiles,
             std::min<uint32_t>(model.router_cores_per_link, u32(router_core_indices.size())),
             slots_free + 64 * static_cast<uint32_t>(direction)}) {
        out.push_back(value);
    }
    add(out, iteration_args);
    return out;
}

Args ChipProgram::logits_block(XDirection direction, uint32_t link) const {
    const Arg fence = zone_free + 48;
    if (!model.chip(coord).gather_ring.peer(direction)) {
        return Args{0u, 0u, 0u};
    }
    const auto router_core_indices = model.port_layout(link);
    const uint32_t need = router_core_indices.empty() ? 0 : u32(model.destinations(position, direction).size());
    std::vector<uint32_t> sends;
    for (uint32_t peer_position = 0; peer_position < model.tp_size; ++peer_position) {
        if (peer_position == position || model.direction_toward(position, peer_position) != direction ||
            router_core_indices.empty()) {
            continue;
        }
        const Coord peer_coord = model.coord_of_position(coord, peer_position);
        const XDirection peer_direction = model.direction_toward(peer_position, position);
        if (!model.chip(peer_coord).gather_ring.peer(peer_direction)) {
            continue;
        }
        const auto peer_port = model.virtual_core(model.chip(peer_coord).port(peer_direction, link));
        const auto peer_node = model.node(peer_coord);
        sends.push_back(u32(peer_node.chip_id) | u32(*peer_node.mesh_id) << 16);
        sends.push_back(u32(peer_port.x | peer_port.y << 8));
    }
    Args out = {(need || !sends.empty()) ? fence : 0u, need, u32(sends.size() / 2)};
    out.insert(out.end(), sends.begin(), sends.end());
    return out;
}

std::vector<KernelDescriptor> ChipProgram::gather(std::vector<CBDescriptor>& cbs) {
    const auto& chip = model.chip(coord);
    const auto& gather_ring = chip.gather_ring;
    const Arg x_own_address = arena_arg("xown");
    const auto tables = model.tables(coord);
    std::vector<CoreCoord> port_cores;
    for (const auto& [direction, link, core] : chip.ports) {
        port_cores.push_back(core);
    }
    const auto port_core_set = cells(port_cores);
    // (in the arena, as every head CB: no static CB under the arena)
    cbs.push_back(arena_cb(
        kXGatherCb,
        model.x_gather_cb_offset,
        u32(model.x_gather_cb_bytes),
        tt::DataFormat::Float16_b,
        kBf16TileBytes,
        port_core_set));
    const std::vector<uint32_t> port_compile_time_args = {
        sizing.k_step_tiles,
        sizing.row_tiles_per_core,
        model.k_steps_per_shard,
        model.tiles_per_packet,
        sizing.chunk_row_tiles,
        sizing.row_chunks,
        model.gather_links,
        model.group,
        kArrivalIncrementEvery,
        sizing.engine_columns,
        sizing.engine_rows,
        sizing.family0_size,
        sizing.family1_size,
        model.tp_size};
    const uint32_t num_passes = reduce_scatter.pass_major ? reduce_scatter.num_passes : 1;
    const std::string latent_chunk_bytes =
        unsigned_literal(sizing.row_chunks > 1 ? sizing.latent_pages * kBf16TileBytes : 0);
    auto reader = kernel(
        "dataflow/fabric_port_stager.cpp",
        port_core_set,
        data_movement(DataMovementProcessor::RISCV_0, NOC::RISCV_1_default));
    reader.compile_time_args = port_compile_time_args;
    append_accessor(reader.compile_time_args, buffers[SOURCE_X]);
    define(reader, "X_GATHER_HANDOFF_BYTES", unsigned_literal(model.handoff_bytes));
    define(reader, "LATENT_CHUNK_BYTES", latent_chunk_bytes);
    define(reader, "REDUCE_SCATTER_PASSES", unsigned_literal(num_passes));
    auto sender = kernel(
        "dataflow/fabric_port_sender.cpp",
        port_core_set,
        data_movement(DataMovementProcessor::RISCV_1, NOC::RISCV_0_default));
    sender.compile_time_args = port_compile_time_args;
    define(sender, "X_GATHER_HANDOFF_BYTES", unsigned_literal(model.handoff_bytes));
    define(sender, "ROUTER_LOGITS_FIRST", integer_literal(model.logits_first()));
    define(sender, "LATENT_CHUNK_BYTES", latent_chunk_bytes);
    define(sender, "REDUCE_SCATTER_PASSES", unsigned_literal(num_passes));
    routing_defines(sender);
    if (model.router_iterations > 1) {
        define(sender, "ROUTER_LOGITS_ITERATIONS", integer_literal(model.router_iterations));
    }
    routing_accessors(sender);
    for (const auto& [direction, link, core] : chip.ports) {
        auto port_plan = model.port_plan(coord, direction, link);
        const bool fence = !port_plan.mine.empty();
        std::vector<uint32_t> reader_entries, sender_entries;
        for (const auto& [shard, part, upstream_base, packets] : port_plan.mine) {
            reader_entries.insert(reader_entries.end(), {shard, part, upstream_base, packets});
            sender_entries.insert(sender_entries.end(), {shard, part, packets});
        }
        const Args reduce_scatter_args = reduce_scatter_block(direction, link);
        Args reader_args = {source_arg(SOURCE_X)};
        add(reader_args, Args{x_own_address, link, arrival, u32(port_plan.mine.size()), port_plan.upstream_increments});
        add(reader_args, reader_entries);
        add(reader_args, tables);
        add(reader_args, Args{x_loaded, sizing.engine_columns * sizing.engine_rows});
        add(reader_args, reduce_scatter_args);
        add(reader_args, latent_block(direction, link));
        set_runtime_args(reader, core, reader_args);

        Args blocks = reduce_scatter_args;
        add(blocks, port_block(direction, link));
        add(blocks, latent_block(direction, link));
        add(blocks, logits_block(direction, link));
        add(blocks, send_block(SendKey::x(direction, link)));
        Args args = {x_own_address, link, arrival, port_plan.expected_increments, ready, u32(fence)};
        const auto& peer = gather_ring.peer(direction);
        if (peer && !gather_ring.links[static_cast<int>(direction)].empty()) {
            const auto& peer_chip = model.chip(*peer);
            const auto peer_opposite_port = model.virtual_core(peer_chip.port(opposite(direction), link)),
                       peer_same_port = model.virtual_core(peer_chip.port(direction, link)),
                       watcher_core = model.virtual_core(model.watcher);
            const auto peer_node = model.node(*peer);
            add(args,
                Args{
                    u32(!peer_chip.gather_ring.schedule[static_cast<int>(opposite(direction))].empty()),
                    u32(peer_opposite_port.x),
                    u32(peer_opposite_port.y),
                    u32(peer_same_port.x),
                    u32(peer_same_port.y),
                    u32(*peer_node.mesh_id),
                    u32(peer_node.chip_id),
                    u32(watcher_core.x),
                    u32(watcher_core.y),
                    watch.at({direction, link}),
                    u32(port_plan.mine.size())});
            add(args, sender_entries);
            add(args, model.tables(*peer));
            add(args, blocks);
            add(args, fabric_connection_args(*peer, gather_ring.links[static_cast<int>(direction)][link], core));
        } else {
            add(args, std::vector<uint32_t>(11, 0));
            add(args, tables);
            add(args, blocks);
        }
        set_runtime_args(sender, core, args);
    }
    // the watcher
    std::vector<CoreCoord> engine_virtual_cores;
    for (uint32_t core_x = 0; core_x < sizing.engine_columns; ++core_x) {
        for (uint32_t core_y = 0; core_y < sizing.engine_rows; ++core_y) {
            engine_virtual_cores.push_back(model.engine_virtual_core(core_x, core_y));
        }
    }
    const auto [shards, carry] = model.shard_order(coord);
    Args rank_stream_args;
    std::vector<std::pair<Arg, uint32_t>> finals;
    for (size_t k = 1; k < shards.size(); ++k) {
        const auto& carries = carry.at(shards[k]);
        rank_stream_args.push_back(u32(carries.size()));
        for (const auto& [carry_direction, carry_link, entry_index, num_entries] : carries) {
            add(rank_stream_args, Args{watch.at({carry_direction, carry_link}), entry_index, num_entries});
        }
    }
    for (XDirection direction : {XDirection::Forward, XDirection::Backward}) {
        const auto& upstream = direction == XDirection::Forward ? gather_ring.previous : gather_ring.next;
        if (upstream && !model.chip(*upstream).gather_ring.schedule[static_cast<int>(direction)].empty()) {
            for (uint32_t link = 0; link < model.gather_links; ++link) {
                finals.emplace_back(
                    watch.at({direction, link}),
                    sizing.row_chunks *
                        u32(model.chip(*upstream).gather_ring.schedule[static_cast<int>(direction)].size()));
            }
        }
    }
    Args watcher_args = {
        sizing.row_chunks,
        kXReadySemaphore,
        u32(shards.size() - 1),
        u32(engine_virtual_cores.size()),
        u32(finals.size())};
    for (const auto& [counter, count] : finals) {
        add(watcher_args, Args{counter, count});
    }
    for (const auto& engine_core : engine_virtual_cores) {
        add(watcher_args, Args{u32(engine_core.x), u32(engine_core.y)});
    }
    add(watcher_args, rank_stream_args);
    auto watcher = kernel(
        "dataflow/x_gather_watcher.cpp",
        cells({model.watcher}),
        data_movement(DataMovementProcessor::RISCV_1, NOC::RISCV_1_default));
    define(watcher, "X_GATHER_HANDOFF_BYTES", unsigned_literal(model.handoff_bytes));
    std::vector<SendKey> y_keys;
    for (const auto& [key, _] : senders) {
        if (key.y) {
            y_keys.push_back(key);
        }
    }
    // (the second Y direction's sender, after the watch). One program for the mesh: every
    // chip's watcher has it when any chip has a second Y direction - a chip without one sends nothing (an empty send
    // block, no connection); WATCHER_ROUTING_ARGS_OFFSET is a chip value (chip_values)
    bool second_y = false;
    for (const auto& [_, served] : model.serves) {
        second_y = second_y ||
                   std::count_if(served.begin(), served.end(), [](const auto& entry) { return entry.first.y; }) >= 2;
    }
    if (second_y) {
        routing_defines(watcher);
        define(watcher, "WATCHER_ROUTING_ARGS_OFFSET", integer_literal(watcher_args.size()));
        add(watcher_args, y_keys.size() >= 2 ? y_args(y_keys[1], model.watcher, false) : Args{0u, 0u});
    }
    set_runtime_args(watcher, model.watcher, watcher_args);
    if (skip_padding()) {
        skip_defines(watcher, 0);
        set_common(watcher, live_args());
    }
    routing_accessors(watcher);
    if (skip_padding()) {  // (the gather's ports: each reads the live chunks itself)
        for (auto* port_kernel : {&reader, &sender}) {
            skip_defines(*port_kernel, 0);
            set_common(*port_kernel, live_args());
        }
    }
    std::vector<KernelDescriptor> kernels;
    push(kernels, reader);
    push(kernels, sender);
    push(kernels, watcher);
    return kernels;
}

Args ChipProgram::y_args(const SendKey& key, const CoreCoord& core, bool with_hellos) {
    const auto& destinations = senders.at(key);
    const uint32_t num_remote = u32(destinations.size()) - (destinations.back() == my_chip ? 1 : 0);
    Args args = send_block(key);
    add(args, u32(num_remote > 0));
    if (num_remote) {
        const int64_t y_step = key.direction_code == 2 ? 1 : -1;
        const Coord peer{u32(positive_modulo(int64_t(coord.first) + y_step, model.mesh_rows)), coord.second};
        add(args, word("yhello"));
        add(args, Args{u32(model.node(peer).chip_id), u32(*model.node(peer).mesh_id)});
        const auto links = tt::tt_fabric::get_forwarding_link_indices(*model.mesh, model.node(coord), model.node(peer));
        TT_FATAL(!links.empty(), "moe_head: no link toward the Y neighbour");
        add(args, fabric_connection_args(peer, links[0], core));
        // (after the connection's args, for the Y sender: the hellos to wait for. The watcher's second Y
        // direction sends no hello and waits for none.)
        if (with_hellos) {
            add(args, model.y_hellos_in(coord));
        }
    }
    return args;
}

std::vector<KernelDescriptor> ChipProgram::y_kernels() {
    std::vector<KernelDescriptor> kernels;
    for (const auto& [key, _] : senders) {
        if (!key.y) {
            continue;
        }
        const CoreCoord core = model.watcher;
        auto y_sender = kernel(
            "dataflow/routing_exchange_y_sender.cpp",
            cells({core}),
            data_movement(DataMovementProcessor::RISCV_0, NOC::RISCV_0_default));
        routing_defines(y_sender);
        routing_accessors(y_sender);
        set_runtime_args(y_sender, core, y_args(key, core));
        push(kernels, y_sender);
        break;  // (the first Y direction; the second rides the watcher's NCRISC)
    }
    return kernels;
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head
