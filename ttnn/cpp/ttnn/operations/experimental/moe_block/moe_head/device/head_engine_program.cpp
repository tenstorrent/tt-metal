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
    const Arg latent_words =
        mailbox + model.handoff_bytes * sizing.engine_columns * sizing.engine_rows + kLatentHandoffsGap;
    Args out = {u32(peer_positions.size()), latent_words + model.handoff_bytes * core_index};
    for (uint32_t peer_position : peer_positions) {
        const XDirection direction = latent_direction(position, peer_position);
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
    const std::vector<std::pair<uint32_t, uint32_t>> zone_slots[] = {
        reduce_scatter.zone_slots(position, core_x, core_y, ReduceScatterDirection::Forward),
        reduce_scatter.zone_slots(position, core_x, core_y, ReduceScatterDirection::Backward)};
    auto [first_output_tile, stride, width] = reduce_scatter.final_map(position, core_x, core_y);
    Args out;
    for (int d = 0; d < 2; ++d) {
        out.push_back(u32(arrivals[d].size()));
        for (const auto& scatter_arrival : arrivals[d]) {
            out.push_back(
                moe_head_reduce_scatter_words::kArrivalThreshold.pack(scatter_arrival.threshold) |
                moe_head_reduce_scatter_words::kArrivalZonePages.pack(zone_slots[d][scatter_arrival.slot].second));
        }
    }
    const size_t first_output_tile_at = out.size() + 1;  // (its arg below)
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
        const uint32_t core_row_tile = core_y * sizing.row_tiles_per_core;
        auto chunk_first = [&](uint32_t chunk) {
            return first_output_tile +
                   (sizing.global_row_tile(chunk, core_row_tile) - core_row_tile) * reduce_scatter.block_columns;
        };
        if (reduce_scatter.count(core_x, core_y, position)) {
            out[first_output_tile_at] = chunk_first(0);
        }
        for (uint32_t chunk = 1; chunk < sizing.row_chunks; ++chunk) {
            chunk_first_output_tiles.push_back(chunk_first(chunk));
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
        return noc_xy_word(port_core);
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
    const uint32_t family0_slots = sizing.family0_slots(core_x), owned_slots = sizing.owned_slots(core_x);
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
            row_destinations.push_back(noc_xy_word(target_core));
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
            row_destinations.push_back(noc_xy_word(target_core));
            row_destinations.push_back(arena_base + sub_block * model.expert_tiles * kFp32TileBytes);
            row_destinations.push_back(mailbox + 4 * model.logits_words_offset + kFlagBytes * sub_block);
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
            cbs.push_back(
                arena_cb(buffers[SOURCE_ARENA], cb_index, sizing.at(key), u32(size), data_format, page, all_cores));
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
    auto reduce_scatter_size = [&](const std::string& name) { return region_bytes(sizing.reduce_scatter_sizes, name); };
    add_cb(kReduceScatterHoldCb, "hold", reduce_scatter_size("hold"), fp32, kFp32TileBytes);
    add_cb(kReduceScatterFinalBlockCb, "finb", reduce_scatter_size("finb"), bf16, kBf16TileBytes);
    add_cb(kReduceScatterArrivalForwardCb, "arrf", reduce_scatter_size("arrf"), bf16, kBf16TileBytes);
    add_cb(kReduceScatterArrivalBackwardCb, "arrb", reduce_scatter_size("arrb"), bf16, kBf16TileBytes);
    add_cb(kReduceScatterOutputForwardCb, "outf", reduce_scatter_size("outf"), bf16, kBf16TileBytes);
    add_cb(kReduceScatterFinalCounterCb, "finc", reduce_scatter_size("finc"), bf16, kFlagBytes);
    add_cb(kReduceScatterPassSyncCb, "psync", reduce_scatter_size("psync"), bf16, kFlagBytes);
    if (sizing.row_chunks > 1) {
        add_cb(kReduceScatterFinalFreeCb, "ffree", reduce_scatter_size("ffree"), bf16, kFlagBytes);
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
            add_cb(kLatentReadyCb, "lrdy", sizing.latent_ready_bytes(), bf16, kFlagBytes);
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
    const Arg router_done = sizing.router_overlay ? zone_free + kRouterDoneWord : Arg(0u);
    Args tail = {u32(model.chip(coord).ports.size()), x_loaded};
    for (const auto& [direction, link, port] : model.chip(coord).ports) {
        tail.push_back(u32(model.virtual_core(port).x));
        tail.push_back(u32(model.virtual_core(port).y));
    }
    tail.push_back(kXLoadedSemaphore);
    tail.push_back(router_done);
    const auto family_table = sizing.family_table();
    const auto logits_destinations = engine_destinations();
    const uint32_t num_dram_banks = u32(model.mesh->dram_grid_size().x);
    std::vector<std::pair<CoreCoord, Args>> row_args, column_args;
    // (core_x's index in a family's columns; engine_columns when it is not one of them)
    const auto family_index = [&](const std::vector<uint32_t>& columns, uint32_t core_x) {
        const auto it = std::find(columns.begin(), columns.end(), core_x);
        return it != columns.end() ? u32(it - columns.begin()) : engine_columns;
    };
    for (uint32_t core_x = 0; core_x < engine_columns; ++core_x) {
        const uint32_t family0_index = family_index(sizing.family0_columns, core_x),
                       family1_index = family_index(sizing.family1_columns, core_x);
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
    // (each core's full list as its args - they go to the DRAM tables, head_args_to_dram; the router phase words'
    // address: common[0])
    const Args engine_words_common_args = {word("ewords")};

    // the accessors' positional compile-time args (TensorAccessorArgs): the kernels' only ones - their own values are
    // named args (kernels/head_named_args.hpp), the one list every engine kernel takes (a kernel reads what it needs)
    namespace names = moe_head_named_args;
    const size_t prep_accessor_offset = w_accessor_args.size() + y_accessor_args.size();
    const size_t router_accessor_offset = prep_accessor_offset + accessor_length(buffers[SOURCE_DISPATCH_BUFFER]);
    const bool owner_form = sizing.row_chunks > 1 && sizing.shards > 1;
    const bool latent = sizing.latent_tiles != 0;
    std::vector<std::pair<const char*, int64_t>> engine_named_args = {
        // (the row / column dataflow's and the compute's grid and steps)
        {names::kEngineColumns, engine_columns},
        {names::kEngineRows, engine_rows},
        {names::kRowTilesPerCore, row_tiles_per_core},
        {names::kKStepTiles, sizing.k_step_tiles},
        {names::kGateUpKSteps, sizing.num_k_steps},
        {names::kGateUpPasses, sizing.gate_up_passes},
        {names::kRowChunks, sizing.row_chunks},
        {names::kMaxColumnPairs, sizing.max_column_pairs},
        {names::kFamily0Columns, sizing.family0_size},
        {names::kFamily1Columns, sizing.family1_size},
        {names::kXReadySemaphoreId, kXReadySemaphore},
        {names::kDownKStepTiles, sizing.down_k_step_tiles},
        {names::kDownKSteps, sizing.down_k_steps},
        {names::kDownPasses, sizing.down_passes},
        {names::kWSlots, kWSlots},
        {names::kWSlotTiles, sizing.w_slot_tiles},
        {names::kWTileBytes, sizing.w_tile_bytes},
        {names::kWValidSemaphoreId, kWValidSemaphore},
        {names::kWFreeSemaphoreId, kWFreeSemaphore},
        {names::kKShards, shard_order[1]},
        {names::kKStepsPerShard, shard_order[0]},
        // (the shared expert's sizes; global_row_tile's owner form - row chunks over TP shards - or its chunk-major
        // form, the other's values 0)
        {names::kSharedExpertHSlots, sizing.h_slots},
        {names::kSharedExpertAccumulatorPages, sizing.accumulator_pages},
        {names::kSharedExpertDestTiles, kDestTiles},
        {names::kSharedExpertOwnerChunkRowTiles, owner_form ? sizing.chunk_row_tiles / sizing.shards : 0},
        {names::kSharedExpertOwnerRowTiles, owner_form ? sizing.rows / 32 / sizing.shards : 0},
        {names::kSharedExpertChunkRowTiles, owner_form ? 0 : sizing.chunk_row_tiles},
        {names::kSharedExpertWRingSlots, sizing.w_ring_slots},
        {names::kDispatchPrepLast, sizing.row_chunks > 1},
        // (the reduce-scatter; pass-major scatter passes, 0: none)
        {names::kReduceScatterEntries, reduce_scatter.tp_size},
        {names::kReduceScatterHoldCb, kReduceScatterHoldCb},
        {names::kReduceScatterFinalBlockCb, kReduceScatterFinalBlockCb},
        {names::kReduceScatterArrivalForwardCb, kReduceScatterArrivalForwardCb},
        {names::kReduceScatterArrivalBackwardCb, kReduceScatterArrivalBackwardCb},
        {names::kReduceScatterFinalCounterCb, kReduceScatterFinalCounterCb},
        {names::kReduceScatterOutputForwardCb, kReduceScatterOutputForwardCb},
        {names::kReduceScatterPassSyncCb, kReduceScatterPassSyncCb},
        {names::kReduceScatterFinalFreeCb, kReduceScatterFinalFreeCb},
        {names::kReduceScatterFinalRows, sizing.final_rows()},
        {names::kReduceScatterZoneTilesForward, reduce_scatter.zone_tiles(ReduceScatterDirection::Forward)},
        {names::kReduceScatterZoneTilesBackward, reduce_scatter.zone_tiles(ReduceScatterDirection::Backward)},
        {names::kSharedExpertReduceScatterPasses, reduce_scatter.pass_major ? reduce_scatter.num_passes : 0},
        // (the latent's all-to-all, the other values 0 without it)
        {names::kLatentEnabled, latent},
        {names::kLatentCb, latent ? kLatentCb : 0},
        {names::kSharedExpertGateUpPasses, latent ? sizing.gate_up_passes_without_latent : 0},
        {names::kLatentMaxColumnTiles, latent ? sizing.max_latent_column_tiles : 0},
        {names::kLatentPages, latent ? sizing.latent_pages : 0},
        {names::kLatentReadyCb, latent ? kLatentReadyCb : 0},
        // (the router's logits on the matmul cores: the router phase router_iterations times - the ewords' go flag
        // is router core 0's go count - and its CBs, rounds and flags)
        {names::kRouterXCb, kRouterXCb},
        {names::kRouterWCb, kRouterWCb},
        {names::kRouterOutCb, kRouterOutCb},
        {names::kRouterKTiles, model.router_k_tiles},
        {names::kRouterExpertTilesPerColumn, sizing.router_expert_tiles_per_column},
        {names::kRouterCoreRowTiles,
         row_tiles_per_core * sizing.row_chunks / model.router_iterations},  // (a core's rows of one iteration)
        {names::kRouterIterations, model.router_iterations},
        {names::kRouterGoFlagId, model.router_iterations > 1 ? kRouterGoFlag : 0},
        {names::kRouterRowTilesPerGroup, sizing.router_row_tiles_per_group},
        {names::kRouterDepth, 3},
        {names::kXAcknowledgeFlagId, kXAckFlag},
        {names::kWAcknowledgeFlagId, kWAckFlag},
        {names::kRouterXReadyFlagId, kRouterXReadyFlag},
        {names::kRouterWReadyFlagId, kRouterWReadyFlag},
        {names::kRouterDoneCredit, sizing.router_overlay},
        {names::kRouterXSentFlagId, kRouterXSentFlag},
        {names::kRouterKChunkTiles, sizing.router_k_chunk_tiles},
        {names::kXReadLocalFlagId, kXReadLocalFlag},
        {names::kRouterAccessorOffset, router_accessor_offset},
        // (the dispatch prep)
        {names::kDispatchPrepUnitCb, kDispatchPrepUnitCb},
        {names::kDispatchPrepOutCb, kDispatchPrepOutCb},
        {names::kDispatchPrepFullColumnTiles, sizing.prep_full_column_tiles},
        {names::kDispatchPrepBlockColumnTiles, sizing.prep_block_column_tiles},
        {names::kDispatchPrepRowBytes, buffers[SOURCE_DISPATCH_BUFFER]->aligned_page_size()},
        {names::kDispatchPrepAccessorOffset, prep_accessor_offset}};
    auto set_engine_named_args = [&](KernelDescriptor& kernel_descriptor) {
        for (const auto& [name, value] : engine_named_args) {
            named_arg(kernel_descriptor, name, value);
        }
    };

    std::vector<KernelDescriptor> kernels;
    auto row_dataflow = kernel(
        "dataflow/shared_expert_row_dataflow.cpp",
        all_cores,
        data_movement(DataMovementProcessor::RISCV_0, NOC::NOC_0));
    row_dataflow.compile_time_args = x_accessor_args;
    set_engine_named_args(row_dataflow);
    for (const auto& [core, args] : row_args) {
        set_runtime_args(row_dataflow, core, args);
    }
    // (a padded forward: the live row chunks BRISC reads, after its own common args)
    const std::optional<uint32_t> live_args_at =
        skip_padding() ? std::optional<uint32_t>(u32(engine_words_common_args.size())) : std::nullopt;
    skip_padding_args(row_dataflow, live_args_at);
    if (skip_padding()) {
        Args row_common = engine_words_common_args;
        add(row_common, live_args());
        set_common(row_dataflow, row_common);
    } else {
        set_common(row_dataflow, engine_words_common_args);
    }
    auto column_dataflow = kernel(
        "dataflow/shared_expert_column_dataflow.cpp",
        all_cores,
        data_movement(DataMovementProcessor::RISCV_1, NOC::NOC_1));
    auto& column_dataflow_compile_time_args = column_dataflow.compile_time_args;
    column_dataflow_compile_time_args.insert(
        column_dataflow_compile_time_args.end(), w_accessor_args.begin(), w_accessor_args.end());
    column_dataflow_compile_time_args.insert(
        column_dataflow_compile_time_args.end(), y_accessor_args.begin(), y_accessor_args.end());
    append_accessor(column_dataflow_compile_time_args, buffers[SOURCE_DISPATCH_BUFFER]);
    append_accessor(column_dataflow_compile_time_args, buffers[SOURCE_X]);
    append_accessor(column_dataflow_compile_time_args, buffers[SOURCE_ROUTER_W]);
    set_engine_named_args(column_dataflow);
    // (skip_padding_chunks: NCRISC reads the live chunks from ewords)
    skip_padding_args(column_dataflow, live_args_at);
    for (const auto& [core, args] : column_args) {
        set_runtime_args(column_dataflow, core, args);
    }
    set_common(column_dataflow, engine_words_common_args);
    push(kernels, row_dataflow);
    push(kernels, column_dataflow);

    // (the activation: common/kernels/expert_activation.hpp - its kind and its constants named compile-time args)
    const uint32_t activation_code = expert_activation_code(model.config.activation.kind);
    const auto activation_constant_args = expert_activation_constant_args(model.config.activation);
    for (const auto& [key, columns] : sizing.groups) {
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
        auto compute = kernel("compute/shared_expert_compute.cpp", model.column_cores(columns), compute_config);
        set_engine_named_args(compute);
        compute.named_compile_time_args.emplace_back(kExpertActivationArgName, activation_code);
        compute.named_compile_time_args.insert(
            compute.named_compile_time_args.end(), activation_constant_args.begin(), activation_constant_args.end());

        for (const auto& [core, args] : core_args) {
            set_runtime_args(compute, core, args);
        }
        skip_padding_args(compute, skip_padding() ? std::optional<uint32_t>(0) : std::nullopt);
        if (skip_padding()) {  // (the live chunks from ewords: its address, common arg 0)
            set_common(compute, Args{word("ewords")});
        }
        push(kernels, compute);
    }
    return kernels;
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head
