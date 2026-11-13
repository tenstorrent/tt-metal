// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The matmul cores' kernels (the shared expert, the router's logits, the reduce-scatter and the dispatch prep):
// ChipProgram::engine and its arg helpers.
#include "head_chip_program.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

std::vector<uint32_t> ChipProgram::shard_order_args() const {
    std::vector<uint32_t> shard_order = model.shard_order(coord).first;
    std::vector<uint32_t> out = {sizing.num_k_steps / u32(shard_order.size()), u32(shard_order.size())};
    out.insert(out.end(), shard_order.begin(), shard_order.end());
    return out;
}

// ---- the matmul cores' (shared expert's) kernels ----

Args ChipProgram::reduce_scatter_row_dataflow_args(uint32_t core_x, uint32_t core_y) const {
    const uint32_t core_index = reduce_scatter.core_index(core_x, core_y);
    const auto handoffs = reduce_scatter.handoffs(position, core_x, core_y);
    Args out = {u32(handoffs.size()), mailbox + model.handoff_bytes * core_index};
    for (const auto& [_, direction] : handoffs) {
        const auto& port_core = port_virtual_cores.at(
            {direction == ReduceScatterDirection::Forward ? XDirection::Forward : XDirection::Backward,
             reduce_scatter.link(core_index)});
        out.emplace_back(u32(port_core.x));
        out.emplace_back(u32(port_core.y));
    }
    return out;
}

Args ChipProgram::latent_row_dataflow_args(uint32_t core_x, uint32_t core_y) const {
    if (!sizing.latent_tiles) {
        return {};
    }
    const uint32_t core_index = core_y * sizing.engine_columns + core_x;
    std::vector<uint32_t> peer_positions;
    for (uint32_t peer_position = 0; peer_position < model.tp_size; ++peer_position) {
        if (peer_position != position && sizing.latent_rows(core_y, peer_position).second) {
            peer_positions.push_back(peer_position);
        }
    }
    const Arg latent_words = mailbox + model.handoff_bytes * sizing.engine_columns * sizing.engine_rows + 128;
    Args out = {u32(peer_positions.size()), latent_words + model.handoff_bytes * core_index};
    for (uint32_t peer_position : peer_positions) {
        const XDirection direction = reduce_scatter.kind == ReduceScatterKind::Line
                                         ? (peer_position > position ? XDirection::Forward : XDirection::Backward)
                                         : model.ring_direction(position, peer_position);
        const auto& port_core = port_virtual_cores.at({direction, core_index % reduce_scatter.links});
        out.emplace_back(u32(port_core.x));
        out.emplace_back(u32(port_core.y));
    }
    return out;
}

Args ChipProgram::latent_column_dataflow_args(uint32_t core_y) const {
    if (!sizing.latent_tiles) {
        return {};
    }
    const uint32_t num_latent_rows = sizing.latent_rows(core_y, position).second;
    return {num_latent_rows, num_latent_rows ? model.tp_size - 1 : 0u, engine_arrival_words[2]};
}

Args ChipProgram::reduce_scatter_column_dataflow_args(uint32_t core_x, uint32_t core_y) const {
    const auto arrivals = reduce_scatter.arrivals(position, core_x, core_y);
    const auto forward_zone_slots =
                   reduce_scatter.zone_slots(position, core_x, core_y, ReduceScatterDirection::Forward),
               backward_zone_slots =
                   reduce_scatter.zone_slots(position, core_x, core_y, ReduceScatterDirection::Backward);
    auto [first_output_tile, stride, width] = reduce_scatter.final_map(position, core_x, core_y);
    Args out = {u32(arrivals[0].size())};
    for (const auto& scatter_arrival : arrivals[0]) {
        out.push_back(scatter_arrival.threshold | forward_zone_slots[scatter_arrival.slot].second << 16);
    }
    out.push_back(u32(arrivals[1].size()));
    for (const auto& scatter_arrival : arrivals[1]) {
        out.push_back(scatter_arrival.threshold | backward_zone_slots[scatter_arrival.slot].second << 16);
    }
    for (const Arg& value : std::initializer_list<Arg>{
             reduce_scatter.count(core_x, core_y, position),
             first_output_tile,
             stride,
             width,
             engine_arrival_words[0],
             engine_arrival_words[1]}) {
        out.push_back(value);
    }
    if (sizing.row_chunks == 1) {
        return out;
    }
    std::vector<uint32_t> chunk_first_output_tiles;
    if (reduce_scatter.scatter == -1) {
        if (reduce_scatter.count(core_x, core_y, position)) {
            out[out.size() - 5] = first_output_tile + (sizing.global_row_tile(0, core_y * sizing.row_tiles_per_core) -
                                                       core_y * sizing.row_tiles_per_core) *
                                                          reduce_scatter.block_columns;
        }
        for (uint32_t chunk = 1; chunk < sizing.row_chunks; ++chunk) {
            chunk_first_output_tiles.push_back(
                first_output_tile + (sizing.global_row_tile(chunk, core_y * sizing.row_tiles_per_core) -
                                     core_y * sizing.row_tiles_per_core) *
                                        reduce_scatter.block_columns);
        }
    } else {
        for (uint32_t chunk = 1; chunk < sizing.row_chunks; ++chunk) {
            chunk_first_output_tiles.push_back(
                first_output_tile + chunk * reduce_scatter.block_row_tiles * reduce_scatter.hidden_tiles);
        }
    }
    const uint32_t link = reduce_scatter.link(reduce_scatter.core_index(core_x, core_y));
    auto facing = [&](int direction_index) -> uint32_t {
        if (arrivals[direction_index].empty()) {
            return 0;
        }
        const auto& port_core =
            port_virtual_cores.at({direction_index == 0 ? XDirection::Backward : XDirection::Forward, link});
        return u32(port_core.x | port_core.y << 16);
    };
    out.insert(out.end(), chunk_first_output_tiles.begin(), chunk_first_output_tiles.end());
    for (const Arg& value : std::initializer_list<Arg>{
             facing(0),
             facing(1),
             zone_free,
             engine_arrival_words.size() > 3 ? engine_arrival_words[3] : Arg(0u),
             u32(reduce_scatter.final_block_readers(position, core_x, core_y).size())}) {
        out.push_back(value);
    }
    if (sizing.final_rows()) {
        const uint32_t column = first_output_tile - core_y * sizing.row_tiles_per_core * reduce_scatter.block_columns;
        for (uint32_t chunk = 0; chunk < sizing.row_chunks; ++chunk) {
            for (uint32_t row_tile = 0; row_tile < sizing.row_tiles_per_core; ++row_tile) {
                out.push_back(
                    column + sizing.global_row_tile(chunk, core_y * sizing.row_tiles_per_core + row_tile) *
                                 reduce_scatter.block_columns);
            }
        }
    }
    return out;
}

std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> ChipProgram::prep_units(
    uint32_t core_x, uint32_t core_y) const {
    const uint32_t row_tiles_per_core = sizing.row_tiles_per_core, k_step_tiles = sizing.k_step_tiles,
                   num_shards = sizing.shards;
    // (the dispatch buffer's slots: chip my_chip's at my_chip x tokens_per_chip_all_blocks, this row block's at
    // block x tokens_per_chip in them)
    const uint32_t first_row = my_chip * model.tokens_per_chip_all_blocks + buffers.block * model.tokens_per_chip;
    const uint32_t owner_row_tiles = sizing.rows / 32 / num_shards;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> out;
    if (sizing.latent_tiles) {
        const auto [first_latent_row, num_latent_rows] = sizing.latent_rows(core_y, position);
        const uint32_t latent_zone_offset = u32(sizing.at("latz") - sizing.at("lat"));
        const uint32_t owner_chunk_row_tiles = sizing.chunk_row_tiles / num_shards;
        for (uint32_t chunk = 0; chunk < sizing.row_chunks; ++chunk) {
            const uint32_t chunk_offset = u32(chunk * sizing.latent_pages * kBf16TileBytes);
            const uint32_t owner_row_tile = core_y * row_tiles_per_core + first_latent_row -
                                            position * owner_chunk_row_tiles + chunk * owner_chunk_row_tiles;
            for (uint32_t i = 0; i < num_latent_rows; ++i) {
                const uint32_t row_offset = (first_latent_row + i) * sizing.max_latent_column_tiles * kBf16TileBytes,
                               column_tiles = sizing.latent_column_tiles[core_x];
                out.emplace_back(
                    chunk_offset + row_offset,
                    column_tiles,
                    0,
                    first_row + (owner_row_tile + i) * 32,
                    (position * sizing.latent_tiles + sizing.latent_column_first_tile[core_x]) * 64);
                for (uint32_t slot = 0; slot < num_shards - 1; ++slot) {
                    const uint32_t source_position = slot < position ? slot : slot + 1;
                    out.emplace_back(
                        chunk_offset + latent_zone_offset +
                            slot * row_tiles_per_core * sizing.max_latent_column_tiles * kBf16TileBytes + row_offset,
                        column_tiles,
                        1,
                        first_row + (owner_row_tile + i) * 32,
                        (source_position * sizing.latent_tiles + sizing.latent_column_first_tile[core_x]) * 64);
                }
            }
        }
        return out;
    }
    const std::vector<uint32_t> shard_order = model.shard_order(coord).first;
    const uint32_t k_steps_per_shard = sizing.num_k_steps / u32(shard_order.size());
    std::map<uint32_t, uint32_t> rank;
    for (uint32_t k = 0; k < shard_order.size(); ++k) {
        rank[shard_order[k]] = k;
    }
    const uint32_t half_k_steps = sizing.num_k_steps / 2;
    const uint32_t family0_slots = sizing.in_family0(core_x) ? half_k_steps / sizing.family0_size : 0;
    const uint32_t owned_slots = family0_slots + (sizing.in_family1(core_x) ? half_k_steps / sizing.family1_size : 0);
    const uint32_t owner_chunk_row_tiles = sizing.chunk_row_tiles / num_shards;
    for (uint32_t owner_row = 0; owner_row < owner_row_tiles; ++owner_row) {
        uint32_t chunk, local_row_tile;
        if (sizing.row_chunks > 1) {
            chunk = owner_row / owner_chunk_row_tiles;
            local_row_tile = position * owner_chunk_row_tiles + owner_row % owner_chunk_row_tiles;
        } else {
            chunk = (position * owner_row_tiles + owner_row) / sizing.chunk_row_tiles;
            local_row_tile = (position * owner_row_tiles + owner_row) % sizing.chunk_row_tiles;
        }
        const uint32_t owner_global_row = position * owner_row_tiles + owner_row;
        if (local_row_tile / row_tiles_per_core != core_y) {
            continue;
        }
        const uint32_t core_row_tile = local_row_tile % row_tiles_per_core;
        for (uint32_t j = 0; j < shard_order.size() * k_steps_per_shard; ++j) {
            const uint32_t k_step = rank.at(j / k_steps_per_shard) * k_steps_per_shard + j % k_steps_per_shard;
            const auto [column, family, _] = sizing.owner(k_step);
            if (column != core_x) {
                continue;
            }
            const uint32_t pair_index = k_step / 2;
            const uint32_t slot =
                family ? family0_slots + pair_index / sizing.family1_size : pair_index / sizing.family0_size;
            out.emplace_back(
                ((chunk * owned_slots + slot) * row_tiles_per_core + core_row_tile) * k_step_tiles * kBf16TileBytes,
                k_step_tiles,
                0,
                first_row + (owner_global_row - position * owner_row_tiles) * 32,
                j * k_step_tiles * 64);
        }
    }
    return out;
}

std::vector<uint32_t> ChipProgram::prep_compute(
    const std::vector<uint32_t>& compute_args, size_t num_prefix_args, uint32_t core_x, uint32_t core_y) const {
    const uint32_t num_operations = compute_args[0];
    std::vector<uint32_t> operations(compute_args.begin() + 1, compute_args.begin() + 1 + num_operations),
        rest(compute_args.begin() + 1 + num_operations, compute_args.end());
    uint32_t last_down_pass = 0;
    for (uint32_t i = 0; i < operations.size(); ++i) {
        if ((operations[i] & 0xFF) == 0) {
            last_down_pass = i;
        }
    }
    const uint32_t prep_units_index = u32(num_prefix_args + 1 + num_operations + 1 + rest.size());
    const auto units = prep_units(core_x, core_y);
    operations.insert(operations.begin() + last_down_pass + 1, 3 | (prep_units_index << 8));
    TT_FATAL(units.size() % sizing.row_chunks == 0, "moe_head: dispatch prep units not whole chunks");
    std::vector<uint32_t> out = {num_operations + 1};
    out.insert(out.end(), operations.begin(), operations.end());
    out.insert(out.end(), rest.begin(), rest.end());
    out.push_back(u32(units.size() / sizing.row_chunks));
    for (const auto& [offset, tiles, latent, _row, _column] : units) {
        out.push_back(offset);
        out.push_back(tiles | (latent << 16));
    }
    return out;
}

Args ChipProgram::prep_column_dataflow_args(uint32_t core_x, uint32_t core_y) const {
    const auto units = prep_units(core_x, core_y);
    Args out = {u32(units.size()), source_arg(SOURCE_DISPATCH_BUFFER)};
    for (const auto& [_offset, tiles, _latent, row, column_offset] : units) {
        add(out, Args{row, column_offset, tiles * 64});
    }
    return out;
}

// The matmul cores' router logits: per row tile its destinations (NoC x | y << 16, base, count address)
std::vector<Args> ChipProgram::engine_destinations() const {
    std::vector<Args> all_destinations;
    for (uint32_t row_tile = 0; row_tile < model.tp_size * model.router_row_tiles; ++row_tile) {
        // (the owner's local row tile, row_tile % router_row_tiles: its router iteration that /
        // router_row_tiles_per_iteration, its router core that % router_row_tiles_per_iteration)
        const uint32_t owner = row_tile / model.router_row_tiles,
                       router_core = (row_tile % model.router_row_tiles) % model.router_row_tiles_per_iteration;
        Args row_destinations;
        if (owner == position) {
            const auto target_core = model.virtual_core(model.router_cores[router_core]);
            row_destinations.push_back(u32(target_core.x | target_core.y << 16));
            row_destinations.push_back(router_arena_arg("hold") + owner * model.expert_tiles * kFp32TileBytes);
            row_destinations.push_back(router_arrival1);
        } else {
            const uint32_t link = router_core % model.gather_links;
            const auto router_core_indices = model.port_layout(link);
            const XDirection direction = model.direction_toward(position, owner);
            const auto direction_destinations = model.destinations(position, direction);
            const uint32_t sub_block =
                u32(std::find(direction_destinations.begin(), direction_destinations.end(), owner) -
                    direction_destinations.begin()) *
                    u32(router_core_indices.size()) +
                u32(std::find(router_core_indices.begin(), router_core_indices.end(), router_core) -
                    router_core_indices.begin());
            const auto& target_core = port_virtual_cores.at({direction, link});
            row_destinations.push_back(u32(target_core.x | target_core.y << 16));
            row_destinations.push_back(arena_base + sub_block * model.expert_tiles * kFp32TileBytes);
            row_destinations.push_back(mailbox + 4 * model.logits_words_offset + 16 * sub_block);
        }
        all_destinations.push_back(row_destinations);
    }
    return all_destinations;
}

std::vector<KernelDescriptor> ChipProgram::engine(std::vector<CBDescriptor>& cbs) {
    const uint32_t engine_columns = sizing.engine_columns, engine_rows = sizing.engine_rows,
                   row_tiles_per_core = sizing.row_tiles_per_core;
    const auto all_cores = model.engine_cores();
    Buffer* x_buffer = buffers[SOURCE_X];
    const Arg x_address = source_arg(SOURCE_X);
    const auto bf16 = tt::DataFormat::Float16_b, fp32 = tt::DataFormat::Float32;
    auto add_cb =
        [&](uint8_t cb_index, const std::string& key, int64_t size, tt::DataFormat data_format, uint32_t page) {
            cbs.push_back(arena_cb(cb_index, sizing.at(key), u32(size), data_format, page, all_cores));
        };
    add_cb(
        SHARED_EXPERT_CB_X, "xin", 2 * row_tiles_per_core * sizing.k_step_tiles * kBf16TileBytes, bf16, kBf16TileBytes);
    add_cb(
        SHARED_EXPERT_CB_W,
        "win",
        kWSlots * sizing.w_slot_tiles * sizing.w_tile_bytes,
        weight_format(model.config.w_dtype),
        sizing.w_tile_bytes);
    add_cb(
        SHARED_EXPERT_CB_H_LOCAL,
        "hown",
        row_tiles_per_core * sizing.max_column_pairs * kBf16TileBytes,
        bf16,
        kBf16TileBytes);
    add_cb(
        SHARED_EXPERT_CB_H_GATHERED,
        "hall",
        sizing.h_slots * engine_columns * row_tiles_per_core * sizing.max_column_pairs * kBf16TileBytes,
        bf16,
        kBf16TileBytes);
    auto reduce_scatter_size = [&](const std::string& region) {
        for (const auto& [name, size] : sizing.reduce_scatter_sizes) {
            if (name == region) {
                return size;
            }
        }
        TT_THROW("moe_head: no RS region {}", region);
    };
    add_cb(kReduceScatterHoldCb, "hold", reduce_scatter_size("hold"), fp32, kFp32TileBytes);
    add_cb(kReduceScatterFinalBlockCb, "finb", reduce_scatter_size("finb"), bf16, kBf16TileBytes);
    add_cb(kReduceScatterArrivalForwardCb, "arrf", reduce_scatter_size("arrf"), bf16, kBf16TileBytes);
    add_cb(kReduceScatterArrivalBackwardCb, "arrb", reduce_scatter_size("arrb"), bf16, kBf16TileBytes);
    add_cb(kReduceScatterOutputForwardCb, "outf", reduce_scatter_size("outf"), bf16, kBf16TileBytes);
    add_cb(kReduceScatterFinalCounterCb, "finc", reduce_scatter_size("finc"), bf16, 16);
    add_cb(kReduceScatterPassSyncCb, "psync", reduce_scatter_size("psync"), bf16, 16);
    if (sizing.row_chunks > 1) {
        add_cb(kReduceScatterFinalFreeCb, "ffree", reduce_scatter_size("ffree"), bf16, 16);
    }
    if (sizing.latent_tiles) {
        add_cb(kLatentCb, "lat", sizing.row_chunks * sizing.latent_pages * kBf16TileBytes, bf16, kBf16TileBytes);
    }
    {  // dispatch prep (always: the head plans it with the routing exchange)
        const std::string prep_region = sizing.latent_tiles ? "lat" : "xown";
        const int64_t prep_end = sizing.latent_tiles
                                     ? sizing.at("latz") +
                                           int64_t(sizing.shards - 1) * row_tiles_per_core *
                                               sizing.max_latent_column_tiles * kBf16TileBytes +
                                           int64_t(sizing.row_chunks - 1) * sizing.latent_pages * kBf16TileBytes
                                     : sizing.at("xown") + sizing.x_own_tiles * kBf16TileBytes;
        add_cb(kDispatchPrepUnitCb, prep_region, prep_end - sizing.at(prep_region), bf16, kBf16TileBytes);
        add_cb(kDispatchPrepOutCb, "po", 2 * sizing.prep_full_column_tiles * kBf16TileBytes, bf16, kBf16TileBytes);
        if (sizing.latent_tiles) {
            add_cb(kLatentReadyCb, "lrdy", sizing.latent_ready_bytes(), bf16, 16);
        }
    }
    const uint32_t router_core_row_tiles =
        row_tiles_per_core * sizing.row_chunks / model.router_iterations;  // (a core's rows of one router iteration)
    add_cb(
        kRouterXCb,
        "rx",
        2 * sizing.router_k_chunk_tiles * router_core_row_tiles * kBf16TileBytes,
        bf16,
        kBf16TileBytes);
    add_cb(
        kRouterWCb,
        "rw",
        2 * sizing.router_expert_tiles_per_column * sizing.router_k_chunk_tiles * kBf16TileBytes,
        bf16,
        kBf16TileBytes);
    add_cb(
        kRouterOutCb,
        "ro",
        sizing.router_expert_tiles_per_column * router_core_row_tiles * kFp32TileBytes,
        fp32,
        kFp32TileBytes);
    add_cb(SHARED_EXPERT_CB_PARTIAL_SUMS, "acc", sizing.accumulator_pages * kFp32TileBytes, fp32, kFp32TileBytes);

    std::vector<uint32_t> x_accessor_args, w_accessor_args, y_accessor_args;
    append_accessor(x_accessor_args, x_buffer);
    append_accessor(w_accessor_args, buffers[SOURCE_W]);
    append_accessor(y_accessor_args, buffers[SOURCE_Y]);
    const auto shard_order = shard_order_args();
    // the gather's ports (NoC) to tell once my own-shard x is loaded, their counter, my X-loaded semaphore id, the
    // "router done" word
    const Arg router_done = sizing.router_overlay ? zone_free + 32 : Arg(0u);
    Args tail = {u32(model.chip(coord).ports.size()), x_loaded};
    for (const auto& [direction, link, port] : model.chip(coord).ports) {
        tail.push_back(u32(model.virtual_core(port).x));
        tail.push_back(u32(model.virtual_core(port).y));
    }
    tail.push_back(kXLoadedSemaphore);
    tail.push_back(router_done);
    const uint32_t half_k_steps = sizing.num_k_steps / 2;
    std::vector<uint32_t> family_table = sizing.family0_columns;
    family_table.insert(family_table.end(), sizing.family1_columns.begin(), sizing.family1_columns.end());
    for (uint32_t column = 0; column < engine_columns; ++column) {
        const uint32_t family0_slots = sizing.in_family0(column) ? half_k_steps / sizing.family0_size : 0;
        family_table.push_back(family0_slots + (sizing.in_family1(column) ? half_k_steps / sizing.family1_size : 0));
        family_table.push_back(family0_slots);
    }
    const auto logits_destinations = engine_destinations();
    const uint32_t num_dram_banks = u32(model.mesh->dram_grid_size().x);
    std::vector<std::pair<CoreCoord, Args>> row_args, column_args;
    for (uint32_t core_x = 0; core_x < engine_columns; ++core_x) {
        const uint32_t family0_index =
            sizing.in_family0(core_x)
                ? u32(std::find(sizing.family0_columns.begin(), sizing.family0_columns.end(), core_x) -
                      sizing.family0_columns.begin())
                : engine_columns;
        const uint32_t family1_index =
            sizing.in_family1(core_x)
                ? u32(std::find(sizing.family1_columns.begin(), sizing.family1_columns.end(), core_x) -
                      sizing.family1_columns.begin())
                : engine_columns;
        for (uint32_t core_y = 0; core_y < engine_rows; ++core_y) {
            Args brisc_args = {
                core_x, core_y, family0_index, family1_index, u32(model.engine_virtual_core(core_x, core_y).y)};
            add(brisc_args, x_address);
            add(brisc_args,
                Args{
                    model.k_shard_tiles,
                    position * model.k_shard_tiles,
                    arena_arg("xown"),
                    arena_arg("xin"),
                    arena_arg("hown"),
                    arena_arg("hall"),
                    u32(model.engine_virtual_core(0, core_y).x),
                    u32(model.engine_virtual_core(engine_columns - 1, core_y).x)});
            add(brisc_args, arena_arg("rx"));
            add(brisc_args, family_table);
            for (uint32_t column = 0; column < engine_columns; ++column) {
                add(brisc_args, u32(model.engine_virtual_core(column, core_y).x));
            }
            add(brisc_args, shard_order);
            add(brisc_args, tail);
            add(brisc_args, source_arg(SOURCE_Y));
            add(brisc_args, reduce_scatter_row_dataflow_args(core_x, core_y));
            add(brisc_args, latent_row_dataflow_args(core_x, core_y));
            Args router_logits_args;
            // (the router logits' args)
            const uint32_t first_expert_tile = core_x * sizing.router_expert_tiles_per_column;
            const uint32_t num_expert_tiles = u32(std::max<int64_t>(
                0,
                std::min<int64_t>(
                    sizing.router_expert_tiles_per_column, int64_t(model.expert_tiles) - first_expert_tile)));
            router_logits_args = {
                source_arg(SOURCE_X),
                source_arg(SOURCE_ROUTER_W),
                num_expert_tiles,
                first_expert_tile,
                arena_arg("rx"),
                arena_arg("rw"),
                u32(model.engine_virtual_core(engine_columns - 1, core_y).x),
                u32(model.engine_virtual_core(0, core_y).x),
                u32(model.engine_virtual_core(0, core_y).y),
                u32(num_expert_tiles > 0)};
            for (uint32_t row_tile = 0; row_tile < row_tiles_per_core * sizing.row_chunks; ++row_tile) {
                add(router_logits_args,
                    logits_destinations[sizing.global_row_tile(
                        row_tile / row_tiles_per_core, core_y * row_tiles_per_core + row_tile % row_tiles_per_core)]);
            }
            Args ncrisc_args = {
                core_y,
                u32(model.engine_virtual_core(core_x, core_y).x),
                u32(((core_x * engine_rows + core_y) / num_dram_banks) * sizing.w_own_tiles * sizing.w_tile_bytes),
                (core_x * engine_rows + core_y) % num_dram_banks};
            add(ncrisc_args, router_logits_args);
            add(ncrisc_args, sizing.gate_up_pass_widths[core_x]);
            add(ncrisc_args, sizing.down_pass_widths[core_x]);
            add(ncrisc_args,
                Args{
                    u32(model.engine_virtual_core(core_x, 0).y),
                    u32(model.engine_virtual_core(core_x, engine_rows - 1).y)});
            add(ncrisc_args, source_arg(SOURCE_W));
            add(ncrisc_args, Args{arena_arg("wown"), arena_arg("win"), arena_arg("words")});
            add(ncrisc_args, source_arg(SOURCE_Y));
            for (uint32_t row = 0; row < engine_rows; ++row) {
                add(ncrisc_args, u32(model.engine_virtual_core(core_x, row).y));
            }
            add(ncrisc_args, shard_order);
            add(ncrisc_args, tail);
            add(ncrisc_args, reduce_scatter_column_dataflow_args(core_x, core_y));
            add(ncrisc_args, latent_column_dataflow_args(core_y));
            add(ncrisc_args, prep_column_dataflow_args(core_x, core_y));
            row_args.emplace_back(CoreCoord(core_x, core_y), brisc_args);
            column_args.emplace_back(CoreCoord(core_x, core_y), ncrisc_args);
        }
    }
    // (each core's full list as its args - they go to the DRAM tables, head_args_to_dram; ROUTER_PHASE_WORDS_ADDR:
    // common[0])
    const Args engine_words_common_args = {word("ewords")};

    const std::vector<uint32_t> row_compile_time_args = {
        engine_columns,
        row_tiles_per_core,
        sizing.k_step_tiles,
        sizing.num_k_steps,
        sizing.gate_up_passes,
        sizing.row_chunks,
        sizing.max_column_pairs,
        sizing.family0_size,
        sizing.family1_size,
        kXReadySemaphore};
    const std::vector<uint32_t> column_compile_time_args = {
        engine_rows,
        row_tiles_per_core,
        sizing.k_step_tiles,
        sizing.num_k_steps,
        sizing.gate_up_passes,
        sizing.row_chunks,
        sizing.down_k_step_tiles,
        sizing.down_k_steps,
        sizing.down_passes,
        kWSlots,
        sizing.w_slot_tiles,
        sizing.w_tile_bytes,
        kWValidSemaphore,
        kWFreeSemaphore};
    // the defines every engine kernel takes
    std::vector<std::pair<std::string, std::string>> engine_defines = {
        {"SHARED_EXPERT_H_SLOTS", integer_literal(sizing.h_slots)},
        {"SHARED_EXPERT_ACCUMULATOR_PAGES", unsigned_literal(sizing.accumulator_pages)},
        {"SHARED_EXPERT_DEST_TILES", integer_literal(kDestTiles)}};
    if (sizing.row_chunks > 1 && sizing.shards > 1) {
        engine_defines.emplace_back(
            "SHARED_EXPERT_OWNER_CHUNK_ROW_TILES", integer_literal(sizing.chunk_row_tiles / sizing.shards));
        engine_defines.emplace_back("SHARED_EXPERT_OWNER_ROW_TILES", integer_literal(sizing.rows / 32 / sizing.shards));
    } else {
        engine_defines.emplace_back(
            "SHARED_EXPERT_CHUNK_ROW_TILES",
            integer_literal(sizing.chunk_row_tiles));  // (global_row_tile's chunk-major form: row tiles a chunk)
    }
    if (sizing.w_ring_slots) {
        engine_defines.emplace_back("SHARED_EXPERT_W_RING_SLOTS", integer_literal(sizing.w_ring_slots));
    }
    if (sizing.row_chunks > 1) {
        engine_defines.emplace_back("DISPATCH_PREP_LAST", "");
    }
    engine_defines.emplace_back("REDUCE_SCATTER_ENTRIES", integer_literal(reduce_scatter.tp_size));
    for (const auto& [name, cb_index] : std::vector<std::pair<std::string, uint32_t>>{
             {"REDUCE_SCATTER_HOLD_CB", kReduceScatterHoldCb},
             {"REDUCE_SCATTER_FINAL_BLOCK_CB", kReduceScatterFinalBlockCb},
             {"REDUCE_SCATTER_ARRIVAL_FORWARD_CB", kReduceScatterArrivalForwardCb},
             {"REDUCE_SCATTER_ARRIVAL_BACKWARD_CB", kReduceScatterArrivalBackwardCb},
             {"REDUCE_SCATTER_FINAL_COUNTER_CB", kReduceScatterFinalCounterCb},
             {"REDUCE_SCATTER_OUTPUT_FORWARD_CB", kReduceScatterOutputForwardCb},
             {"REDUCE_SCATTER_PASS_SYNC_CB", kReduceScatterPassSyncCb},
             {"REDUCE_SCATTER_FINAL_FREE_CB", kReduceScatterFinalFreeCb}}) {
        engine_defines.emplace_back(name, integer_literal(cb_index));
    }
    if (sizing.final_rows()) {
        engine_defines.emplace_back("REDUCE_SCATTER_FINAL_ROWS", "");
    }
    engine_defines.emplace_back(
        "REDUCE_SCATTER_ZONE_TILES_FORWARD",
        integer_literal(reduce_scatter.zone_tiles(ReduceScatterDirection::Forward)));
    engine_defines.emplace_back(
        "REDUCE_SCATTER_ZONE_TILES_BACKWARD",
        integer_literal(reduce_scatter.zone_tiles(ReduceScatterDirection::Backward)));
    if (reduce_scatter.pass_major) {
        engine_defines.emplace_back("SHARED_EXPERT_REDUCE_SCATTER_PASSES", unsigned_literal(reduce_scatter.num_passes));
    }
    if (sizing.latent_tiles) {
        engine_defines.emplace_back("LATENT_CB", integer_literal(kLatentCb));
        engine_defines.emplace_back(
            "SHARED_EXPERT_GATE_UP_PASSES", integer_literal(sizing.gate_up_passes_without_latent));
        engine_defines.emplace_back("LATENT_MAX_COLUMN_TILES", integer_literal(sizing.max_latent_column_tiles));
        engine_defines.emplace_back("LATENT_PAGES", integer_literal(sizing.latent_pages));
    }
    const size_t prep_accessor_offset =
        column_compile_time_args.size() + w_accessor_args.size() + y_accessor_args.size();
    const size_t router_accessor_offset = prep_accessor_offset + accessor_length(buffers[SOURCE_DISPATCH_BUFFER]);
    engine_defines.emplace_back("ROUTER_X_CB", integer_literal(kRouterXCb));
    engine_defines.emplace_back("ROUTER_W_CB", integer_literal(kRouterWCb));
    engine_defines.emplace_back("ROUTER_OUT_CB", integer_literal(kRouterOutCb));
    engine_defines.emplace_back("ROUTER_K_TILES", integer_literal(model.router_k_tiles));
    engine_defines.emplace_back(
        "ROUTER_EXPERT_TILES_PER_COLUMN", integer_literal(sizing.router_expert_tiles_per_column));
    engine_defines.emplace_back(
        "ROUTER_CORE_ROW_TILES",
        integer_literal(
            row_tiles_per_core * sizing.row_chunks /
            model.router_iterations));  // (a core's rows of one router iteration)
    if (model.router_iterations >
        1) {  // (the router phase router_iterations times; ewords RGO: router core 0's go count)
        engine_defines.emplace_back("ROUTER_ITERATIONS", integer_literal(model.router_iterations));
        engine_defines.emplace_back("ROUTER_GO_FLAG_ID", integer_literal(160));
    }
    engine_defines.emplace_back("ROUTER_ROW_TILES_PER_GROUP", integer_literal(sizing.router_row_tiles_per_group));
    engine_defines.emplace_back("ROUTER_DEPTH", integer_literal(3));
    engine_defines.emplace_back("X_ACKNOWLEDGE_FLAG_ID", integer_literal(kXAckFlag));
    engine_defines.emplace_back("W_ACKNOWLEDGE_FLAG_ID", integer_literal(kWAckFlag));
    engine_defines.emplace_back("ROUTER_X_READY_FLAG_ID", integer_literal(kRouterXReadyFlag));
    engine_defines.emplace_back("ROUTER_W_READY_FLAG_ID", integer_literal(kRouterWReadyFlag));
    if (sizing.router_overlay) {
        engine_defines.emplace_back("ROUTER_DONE_CREDIT", "");
    }
    engine_defines.emplace_back("ROUTER_X_SENT_FLAG_ID", integer_literal(kRouterXSentFlag));
    engine_defines.emplace_back("ROUTER_K_CHUNK_TILES", integer_literal(sizing.router_k_chunk_tiles));
    engine_defines.emplace_back("X_READ_LOCAL_FLAG_ID", integer_literal(kXReadLocalFlag));
    engine_defines.emplace_back("ROUTER_ACCESSOR_CT_OFFSET", integer_literal(router_accessor_offset));
    engine_defines.emplace_back("DISPATCH_PREP_UNIT_CB", integer_literal(kDispatchPrepUnitCb));
    engine_defines.emplace_back("DISPATCH_PREP_OUT_CB", integer_literal(kDispatchPrepOutCb));
    engine_defines.emplace_back("DISPATCH_PREP_FULL_COLUMN_TILES", integer_literal(sizing.prep_full_column_tiles));
    engine_defines.emplace_back("DISPATCH_PREP_BLOCK_COLUMN_TILES", integer_literal(sizing.prep_block_column_tiles));
    engine_defines.emplace_back(
        "DISPATCH_PREP_ROW_BYTES", integer_literal(buffers[SOURCE_DISPATCH_BUFFER]->aligned_page_size()));
    engine_defines.emplace_back("DISPATCH_PREP_ACCESSOR_CT_OFFSET", integer_literal(prep_accessor_offset));
    if (sizing.latent_tiles) {
        engine_defines.emplace_back("LATENT_READY_CB", integer_literal(kLatentReadyCb));
    }

    std::vector<KernelDescriptor> kernels;
    auto row_dataflow = kernel(
        "dataflow/shared_expert_row_dataflow.cpp",
        all_cores,
        data_movement(DataMovementProcessor::RISCV_0, NOC::NOC_0));
    row_dataflow.compile_time_args = row_compile_time_args;
    row_dataflow.compile_time_args.insert(
        row_dataflow.compile_time_args.end(), x_accessor_args.begin(), x_accessor_args.end());
    row_dataflow.defines = KernelDescriptor::Defines(engine_defines.begin(), engine_defines.end());
    for (const auto& [core, args] : row_args) {
        set_runtime_args(row_dataflow, core, args);
    }
    if (skip_padding()) {  // (the live row chunks: BRISC reads them, after its own common args)
        Args row_common = engine_words_common_args;
        skip_defines(row_dataflow, u32(row_common.size()));
        add(row_common, live_args());
        set_common(row_dataflow, row_common);
    } else {
        set_common(row_dataflow, engine_words_common_args);
    }
    auto column_dataflow = kernel(
        "dataflow/shared_expert_column_dataflow.cpp",
        all_cores,
        data_movement(DataMovementProcessor::RISCV_1, NOC::NOC_1));
    column_dataflow.compile_time_args = column_compile_time_args;
    auto& column_dataflow_compile_time_args = column_dataflow.compile_time_args;
    column_dataflow_compile_time_args.insert(
        column_dataflow_compile_time_args.end(), w_accessor_args.begin(), w_accessor_args.end());
    column_dataflow_compile_time_args.insert(
        column_dataflow_compile_time_args.end(), y_accessor_args.begin(), y_accessor_args.end());
    append_accessor(column_dataflow_compile_time_args, buffers[SOURCE_DISPATCH_BUFFER]);
    append_accessor(column_dataflow_compile_time_args, buffers[SOURCE_X]);
    append_accessor(column_dataflow_compile_time_args, buffers[SOURCE_ROUTER_W]);
    column_dataflow.defines = row_dataflow.defines;  // (SKIP_PADDING_CHUNKS: NCRISC reads the live chunks from ewords)
    for (const auto& [core, args] : column_args) {
        set_runtime_args(column_dataflow, core, args);
    }
    set_common(column_dataflow, engine_words_common_args);
    push(kernels, row_dataflow);
    push(kernels, column_dataflow);

    // (the activation: common/kernels/expert_activation.hpp - its kind and its constants named compile-time args)
    const uint32_t activation_code = expert_activation_code(model.config.activation.kind);
    const auto activation_constant_args = expert_activation_constant_args(model.config.activation);
    std::vector<uint32_t> pre_base;
    for (const auto& [key, columns] : sizing.groups) {
        std::vector<CoreRange> core_ranges;
        for (uint32_t column : columns) {
            core_ranges.emplace_back(CoreCoord(column, 0), CoreCoord(column, engine_rows - 1));
        }
        ComputeConfigDescriptor compute_config{
            .math_fidelity = model.config.fidelity,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = kDestTiles == 8,
            .math_approx_mode = false,
        };
        compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
        compute_config.unpack_to_dest_mode[SHARED_EXPERT_CB_PARTIAL_SUMS] =
            tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        std::vector<uint32_t> prefix_args = key.first;
        prefix_args.insert(prefix_args.end(), key.second.begin(), key.second.end());
        prefix_args.insert(prefix_args.end(), sizing.column_first_pair.begin(), sizing.column_first_pair.end());
        prefix_args.push_back(sizing.shared_expert_tiles);
        // each core's full list: prefix_args + its own (SharedExpertComputeArgs' indices)
        std::vector<std::pair<CoreCoord, Args>> core_args;
        for (uint32_t column : columns) {
            for (uint32_t core_y = 0; core_y < engine_rows; ++core_y) {
                Args full;
                add(full, prefix_args);
                add(full,
                    prep_compute(
                        reduce_scatter.compute_args(position, column, core_y), prefix_args.size(), column, core_y));
                core_args.emplace_back(CoreCoord(column, core_y), std::move(full));
            }
        }
        auto compute = kernel("compute/shared_expert_compute.cpp", CoreRangeSet(core_ranges), compute_config);
        compute.compile_time_args = {
            row_tiles_per_core,
            sizing.k_step_tiles,
            sizing.gate_up_passes,
            sizing.row_chunks,
            sizing.max_column_pairs,
            sizing.down_k_step_tiles,
            sizing.down_k_steps,
            sizing.down_passes,
            sizing.w_slot_tiles,
            shard_order[1],
            shard_order[0],
            engine_columns};
        compute.named_compile_time_args.emplace_back(kExpertActivationArgName, activation_code);
        compute.named_compile_time_args.insert(
            compute.named_compile_time_args.end(), activation_constant_args.begin(), activation_constant_args.end());
        compute.defines.insert(compute.defines.end(), engine_defines.begin(), engine_defines.end());

        for (const auto& [core, args] : core_args) {
            set_runtime_args(compute, core, args);
        }
        if (skip_padding()) {  // (the live chunks from ewords: its address, common arg 0)
            skip_defines(compute, 0);
            set_common(compute, Args{word("ewords")});
        }
        push(kernels, compute);
    }
    return kernels;
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head
