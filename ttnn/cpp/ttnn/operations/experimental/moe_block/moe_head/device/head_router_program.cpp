// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router cores' kernels and the routing exchange / planner arg blocks: ChipProgram::router, routing_defines,
// planner_defines, send_block, planner_args.
#include "head_chip_program.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

// ---- the routing exchange ----

void ChipProgram::routing_defines(KernelDescriptor& kernel_descriptor) const {
    for (const auto& [name, value] : std::vector<std::pair<std::string, int64_t>>{
             {"ROUTING_EXCHANGE_ROUTER_CORES", int64_t(model.planner_cores.size())},
             {"ROUTING_EXCHANGE_ROUTER_ROW_TILES", model.router_row_tiles},
             {"ROUTING_EXCHANGE_NUM_CHIPS", model.num_chips},
             {"ROUTING_EXCHANGE_PLANNER_INSTANCES", model.planner_instances},
             {"ROUTING_EXCHANGE_EXPERTS_PER_CHIP", model.experts_per_chip},
             {"ROUTING_EXCHANGE_MY_CHIP", my_chip},
             {"ROUTING_EXCHANGE_HEADER_BYTES", model.routing_header_padded_bytes},
             {"ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES", model.expert_count_block_bytes},
             {"ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES", model.instance_block_bytes},
             {"ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES", model.max_payload_bytes}}) {
        define(kernel_descriptor, name, integer_literal(value));
    }
}

// The routing planner's defines: the planner as routing_planner::plan (its parameters as the ROUTING_PLANNER_* defines)
void ChipProgram::planner_defines(
    KernelDescriptor& kernel_descriptor, uint32_t accessor_compile_time_args_offset) const {
    define(kernel_descriptor, "ROUTING_PLANNER");
    // (row blocks in one program: both planners, the block picks - routing_planner.hpp; the defines below the
    // last's)
    const bool dual = buffers.merged && buffers.row_blocks > 1;
    const bool idle = !dual && idle_planner();
    if (idle) {  // (row blocks: an earlier block's planner plans no token - the exchange still runs, the chips' barrier
                 // between blocks; the last plans every block's)
        define(kernel_descriptor, "ROUTING_PLANNER_IDLE");
    }
    if (dual) {
        define(kernel_descriptor, "ROUTING_PLANNER_PER_BLOCK");
        define(kernel_descriptor, "ROUTING_PLANNER_IDLE_TOKENS", integer_literal(kPlannerIdleTokens));
        define(kernel_descriptor, "ROUTING_PLANNER_IDLE_ROWS_CAPACITY", integer_literal(model.idle_rows_capacity));
        define(
            kernel_descriptor, "ROUTING_PLANNER_IDLE_ENTRIES_CAPACITY", integer_literal(model.idle_entries_capacity));
    }
    routing_defines(kernel_descriptor);
    for (const auto& [name, value] : std::vector<std::pair<std::string, int64_t>>{
             {"ROUTING_PLANNER_TOKENS", idle ? kPlannerIdleTokens : model.tokens_per_chip_all_blocks},
             {"ROUTING_PLANNER_TOP_K", model.top_k},
             {"ROUTING_PLANNER_EXPERTS_PER_CHIP", model.experts_per_chip},
             {"ROUTING_PLANNER_SP_ROWS", model.mesh_rows},
             {"ROUTING_PLANNER_TP_COLUMNS", model.mesh_columns},
             {"ROUTING_PLANNER_MY_CHIP", my_chip},
             {"ROUTING_PLANNER_ACCUMULATORS", model.accumulators},
             {"ROUTING_PLANNER_LINKS", model.routing_links},
             {"ROUTING_PLANNER_PORTS", model.luts.num_ports.at(ChipCoord{coord.first, coord.second})},
             {"ROUTING_PLANNER_TOKENS_PER_PAGE", 32},
             {"ROUTING_PLANNER_INSTANCES", model.planner_instances},
             {"ROUTING_PLANNER_HEADER_BYTES", model.routing_header_padded_bytes},
             {"ROUTING_PLANNER_EXPERT_COUNT_BLOCK_BYTES", model.expert_count_block_bytes},
             {"ROUTING_PLANNER_INSTANCE_BLOCK_BYTES", model.instance_block_bytes},
             {"ROUTING_PLANNER_SLOT_BYTES", model.routing_slot_capacity_bytes},
             {"ROUTING_PLANNER_NUM_PORTS", int64_t(senders.size())},
             {"ROUTING_PLANNER_ROWS_CAPACITY", idle ? model.idle_rows_capacity : model.rows_capacity},
             {"ROUTING_PLANNER_ENTRIES_CAPACITY", idle ? model.idle_entries_capacity : model.entries_capacity},
             {"ROUTING_PLANNER_ACCESSOR_COMPILE_TIME_ARGS_OFFSET", accessor_compile_time_args_offset}}) {
        define(kernel_descriptor, name, integer_literal(value));
    }
}

Args ChipProgram::send_block(const SendKey& key) const {
    auto found = senders.find(key);
    if (found == senders.end()) {
        return {0u};
    }
    const auto& destinations = found->second;
    const uint32_t includes_self = destinations.back() == my_chip ? 1 : 0;
    const uint32_t num_remote = u32(destinations.size()) - includes_self;
    const auto first_planner = model.virtual_core(model.first_planner_core());
    std::vector<SendKey> y_keys;
    for (const auto& [other_key, _] : senders) {
        if (other_key.y) {
            y_keys.push_back(other_key);
        }
    }
    const bool second_y = key.y && key != y_keys[0];
    Args out = {
        u32(destinations.size()),
        num_remote,
        includes_self,
        arena_base + u32(model.buffer_offset(coord, key)),
        u32(model.routing_slot_capacity_bytes)};
    add(out, source_arg(SOURCE_SLOTS));
    add(out,
        Args{
            routing_words.at("c0"),
            routing_words.at("c1"),
            u32(first_planner.x),
            u32(first_planner.y),
            routing_words.at(second_y ? "calls_s2" : "calls_s"),
            routing_words.at(second_y ? "ready2" : "ready"),
            u32(*model.node(coord).mesh_id)});
    for (uint32_t destination :
         destinations) {  // (a word a destination: its fabric chip id - the fabric routes the packet there)
        add(out, destination != my_chip ? u32(model.node_of_index(destination).chip_id) : 0u);
    }
    return out;
}

std::function<Args(uint32_t)> ChipProgram::planner_args() const {
    std::map<uint32_t, std::tuple<CoreCoord, uint32_t, int64_t>> where;
    for (const auto& [key, destinations] : senders) {
        const CoreCoord core = model.core_of(coord, key);
        const int64_t slot_base = model.buffer_offset(coord, key);
        for (uint32_t j = 0; j < destinations.size(); ++j) {
            where[destinations[j]] = {core, j, slot_base};
        }
    }
    Args head = {source_arg(SOURCE_ROWS), source_arg(SOURCE_SLOTS), routing_words.at("c0"), routing_words.at("c1"), 0u};
    for (HeadSource source : {SOURCE_COUNTS, SOURCE_REGIONS, SOURCE_TOKEN_INDEX, SOURCE_TABLE, SOURCE_PLAN}) {
        add(head, source_arg(source));
    }
    const auto& lookup_table = model.luts.lut.at(ChipCoord{coord.first, coord.second});
    for (uint8_t entry : lookup_table) {
        add(head, uint32_t(entry));
    }
    // (routing_planner.hpp PlannerArgs::first_hop: the fabric's first hop from me to each chip, kFirstHopsPerWord a
    // word)
    add(head, FirstHops{.N = model.num_chips, .dirs = model.config.shape.first_hops}.packed_row(my_chip));
    // (the planner cores: an instance on planner_cores[instance_slot(instance).first], its RISC
    // instance_slot(instance).second)
    auto on_risc0 = [this](uint32_t instance) { return model.instance_slot(instance).second == 0; };
    std::vector<uint32_t> planner_core_coordinates;
    for (uint32_t instance = 0; instance < model.planner_instances; ++instance) {
        const auto planner_core = model.virtual_core(model.planner_cores[model.instance_slot(instance).first]);
        planner_core_coordinates.push_back(u32(planner_core.x));
        planner_core_coordinates.push_back(u32(planner_core.y));
    }
    Args instance_args;
    for (int64_t planner_base : planner_bases()) {
        instance_args.push_back(arena_base + u32(planner_base));
    }
    for (uint32_t instance = 0; instance < model.planner_instances; ++instance) {
        instance_args.push_back(on_risc0(instance) ? routing_words.at("ex") : routing_words.at("ex2"));
    }
    for (uint32_t instance = 0; instance < model.planner_instances; ++instance) {
        instance_args.push_back(on_risc0(instance) ? routing_words.at("go") : routing_words.at("go2"));
    }
    if (dual_planner()) {  // (an earlier row block's instances: routing_planner.hpp reads them after the go words)
        for (int64_t planner_base : model.idle_planner_bases) {
            instance_args.push_back(arena_base + u32(planner_base));
        }
    }
    Args destination_args;
    for (uint32_t chip = 0; chip < model.num_chips; ++chip) {
        const auto& [core, slot, slot_base] = where.at(chip);
        const auto destination_core = model.virtual_core(core);
        add(destination_args,
            Args{
                u32(destination_core.x),
                u32(destination_core.y),
                arena_base + u32(slot_base) + slot * u32(model.routing_slot_capacity_bytes)});
    }
    std::vector<SendKey> y_keys;
    for (const auto& [other_key, _] : senders) {
        if (other_key.y) {
            y_keys.push_back(other_key);
        }
    }
    Args port_args;
    for (const auto& [key, _] : senders) {
        const auto sender_core = model.virtual_core(model.core_of(coord, key));
        const bool second_y = key.y && key != y_keys[0];
        add(port_args, Args{u32(sender_core.x), u32(sender_core.y), routing_words.at(second_y ? "ready2" : "ready")});
    }
    const Args tail = {routing_words.at("ready")};
    auto routing_words_copy = routing_words;
    return [=, this](uint32_t instance) {
        Args args = head;
        args[4] = on_risc0(instance) ? routing_words_copy.at("calls_p") : routing_words_copy.at("calls_p2");
        add(args, Args{instance});
        add(args, planner_core_coordinates);
        add(args, instance_args);
        add(args, destination_args);
        add(args, port_args);
        add(args, tail);
        add(args,
            Args{
                on_risc0(instance) ? routing_words_copy.at("go") : routing_words_copy.at("go2"),
                routing_words_copy.at("parts"),
                on_risc0(instance) ? routing_words_copy.at("ex") : routing_words_copy.at("ex2")});
        return args;
    };
}

Args ChipProgram::routing_output_writer_args(uint32_t router_core_index) const {
    std::vector<uint32_t> planner_core_coordinates;
    for (const auto& planner_core : model.planner_cores) {  // (every planner core: told when my rows are written)
        const auto planner_virtual_core = model.virtual_core(planner_core);
        planner_core_coordinates.push_back(u32(planner_virtual_core.x));
        planner_core_coordinates.push_back(u32(planner_virtual_core.y));
    }
    const int32_t instance = model.instance_at(
        router_core_index, 0);  // (the router core planner_cores[router_core_index]: its BRISC's instance)
    if (instance < 0) {
        Args args = {routing_words.at("rows"), routing_words.at("rows2"), 0u};
        add(args, planner_core_coordinates);
        return args;
    }
    Args args = {routing_words.at("rows"), routing_words.at("rows2"), arena_base + u32(planner_bases()[instance])};
    add(args, planner_core_coordinates);
    add(args, planner_args()(u32(instance)));
    return args;
}

Args ChipProgram::routing_input_feeder_args(uint32_t router_core_index) const {
    const int32_t instance = model.instance_at(router_core_index, 1);  // (its NCRISC's)
    if (instance < 0) {
        return {routing_words.at("rows2"), 0u};
    }
    Args args = {routing_words.at("rows2"), arena_base + u32(planner_bases()[instance])};
    add(args, planner_args()(u32(instance)));
    return args;
}

// ---- the router cores' kernels ----

std::vector<KernelDescriptor> ChipProgram::router(std::vector<CBDescriptor>& cbs) {
    const uint32_t tp_size = model.tp_size, expert_tiles = model.expert_tiles,
                   router_row_tiles = model.router_row_tiles, top_k = model.top_k;
    const auto router_core_set = cells(model.router_cores);
    const auto bf16 = tt::DataFormat::Float16_b, fp32 = tt::DataFormat::Float32;
    auto add_cb =
        [&](uint8_t cb_index, const std::string& key, uint32_t size, tt::DataFormat data_format, uint32_t page) {
            cbs.push_back(arena_cb(cb_index, region_of_router(key), size, data_format, page, router_core_set));
        };
    add_cb(ROUTER_CB_LOGITS, "hold", tp_size * expert_tiles * kFp32TileBytes, fp32, kFp32TileBytes);
    add_cb(ROUTER_CB_ARRIVALS, "arr", (tp_size - 1) * expert_tiles * kFp32TileBytes, fp32, kFp32TileBytes);
    add_cb(ROUTER_CB_BIAS, "bias", model.expert_tiles * kFp32TileBytes, fp32, kFp32TileBytes);
    add_cb(ROUTER_CB_SCORES, "sc", expert_tiles * kFp32TileBytes, fp32, kFp32TileBytes);
    add_cb(ROUTER_CB_EXPERT_INDEX, "idx", expert_tiles * kBf16TileBytes, tt::DataFormat::UInt16, kBf16TileBytes);
    add_cb(ROUTER_CB_BEST, "best", kFp32TileBytes, fp32, kFp32TileBytes);
    add_cb(ROUTER_CB_PACK_TO_UNPACK, "psync", 16 * 2, bf16, 16);
    add_cb(ROUTER_CB_ARRIVAL_ORDER, "arrb", 16 * std::max<uint32_t>(1, tp_size - 1), tt::DataFormat::UInt32, 16);
    add_cb(ROUTER_CB_READY, "rdy", 16 * 2, bf16, 16);
    float scale = model.config.route_scale;
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, 4);
    const Args tail = {router_arena_arg("words"), router_arena_arg("best"), router_arena_arg("out"), scale_bits};
    const Arg own_hold_block = router_arena_arg("hold") + position * expert_tiles * kFp32TileBytes;
    const auto order = model.arrival_order(position);
    // router iterations: the router BRISC's args - the "rit" word, router core 0 (counts every router core and logits
    // port of the chip done with an iteration, then increments every matmul core's go word), the active ports' count,
    // the ports (told my iteration is done), the ewords and the matmul cores
    Args iteration_args;
    if (model.router_iterations > 1) {
        const auto first_router_core = model.virtual_core(model.router_cores[0]);
        uint32_t num_active_ports = 0;  // (the ports that relay logits: each counts an iteration's sub-blocks sent)
        for (const auto& [direction, link, core] : model.chip(coord).ports) {
            num_active_ports += model.chip(coord).gather_ring.peer(direction) &&
                                        !model.destinations(position, direction).empty() &&
                                        !model.port_layout(link).empty()
                                    ? 1
                                    : 0;
        }
        add(iteration_args, Args{word("rit"), u32(first_router_core.x | first_router_core.y << 16), num_active_ports});
        const auto ports = model.all_ports();
        add(iteration_args, u32(ports.size()));
        for (const auto& port : ports) {
            const auto port_core = model.virtual_core(port);
            add(iteration_args, u32(port_core.x | port_core.y << 16));
        }
        add(iteration_args,
            Args{word("ewords"), u32(sizing.engine_columns)});  // (the matmul cores: their NoC x's, then y's)
        for (uint32_t core_x = 0; core_x < sizing.engine_columns; ++core_x) {
            add(iteration_args, u32(model.virtual_core(CoreCoord(core_x, 0)).x));
        }
        add(iteration_args, u32(sizing.engine_rows));
        for (uint32_t core_y = 0; core_y < sizing.engine_rows; ++core_y) {
            add(iteration_args, u32(model.virtual_core(CoreCoord(0, core_y)).y));
        }
    }
    std::vector<std::pair<CoreCoord, Args>> input_feeder_runtime_args, router_compute_runtime_args,
        output_writer_runtime_args;
    for (uint32_t router_core_index = 0; router_core_index < model.router_cores.size(); ++router_core_index) {
        const CoreCoord core = model.router_cores[router_core_index];
        Args feeder_args = {source_arg(SOURCE_ROUTER_BIAS), source_arg(SOURCE_ROUTER_INDEX)};
        for (uint32_t owner = 0; owner < tp_size; ++owner) {
            add(feeder_args, owner * router_row_tiles + router_core_index % router_row_tiles);
        }
        add(feeder_args, Args{tp_size - 1, router_arrival0});
        add(feeder_args, order);
        add(feeder_args, tail);
        add(feeder_args, own_hold_block);
        add(feeder_args, routing_input_feeder_args(router_core_index));
        input_feeder_runtime_args.emplace_back(core, feeder_args);
        Args compute_args = {
            1u, 2u};  // (one op: the fin of my owner block - the compute's TP position, ROUTER_CHIP_FROM_READY_PAGE)
        for (uint32_t owner = 0; owner < tp_size; ++owner) {
            add(compute_args, tp_size - 1);  // (its sources: only my own block's entry is read)
        }
        add(compute_args, tail);
        add(compute_args,
            Args{
                router_arena_arg("hold"),
                router_arena_arg("route"),
                32 * (router_core_index %
                      router_row_tiles)});  // (HOLD's base: my block position x expert_tiles pages on)
        router_compute_runtime_args.emplace_back(core, compute_args);
        const uint32_t router_row_tile = router_core_index % router_row_tiles;
        Args writer_args = {source_arg(SOURCE_ROWS)};
        add(writer_args,
            router_row_tile +
                buffers.block * router_row_tiles);  // (row blocks: this block's owned rows after the earlier blocks')
        add(writer_args,
            Args{
                scale_bits,
                router_arena_arg("out"),
                router_arena_arg("words"),
                router_arena_arg("best"),
                router_arena_arg("route")});
        writer_args.push_back(
            source_arg(SOURCE_ROUTE) + buffers.block * model.tokens_per_chip * 4 *
                                           route_words(model.num_chips));  // (one page: this block's words on)
        add(writer_args, Args{32 * router_row_tile, own_hold_block});
        if (model.router_iterations > 1) {  // (router iterations: ROUTER_ITERATION_ARGS args before the planner's - am
                                            // I router core 0, then iteration_args)
            add(writer_args, u32(router_core_index == 0));
            add(writer_args, iteration_args);
        }
        add(writer_args, routing_output_writer_args(router_core_index));
        output_writer_runtime_args.emplace_back(core, writer_args);
    }
    for (const auto& runtime_args_list :
         {input_feeder_runtime_args, router_compute_runtime_args, output_writer_runtime_args}) {
        for (const auto& [core, args] : runtime_args_list) {
            TT_FATAL(args.size() <= 341, "moe_head: router core runtime args {} > 341", args.size());
        }
    }
    const std::vector<uint32_t> router_compile_time_args = {model.router_k_tiles, expert_tiles, tp_size, top_k};
    std::vector<uint32_t> routing_compile_time_args;
    append_accessor(routing_compile_time_args, buffers[SOURCE_ROWS]);
    append_accessor(routing_compile_time_args, buffers[SOURCE_SLOTS]);
    std::vector<uint32_t> feeder_compile_time_args = router_compile_time_args;
    for (Buffer* buffer : {buffers[SOURCE_ROUTER_BIAS], buffers[SOURCE_ROUTER_INDEX]}) {
        append_accessor(feeder_compile_time_args, buffer);
    }
    const uint32_t feeder_routing_accessor_offset = u32(feeder_compile_time_args.size());
    feeder_compile_time_args.insert(
        feeder_compile_time_args.end(), routing_compile_time_args.begin(), routing_compile_time_args.end());
    std::vector<uint32_t> writer_compile_time_args = router_compile_time_args;
    append_accessor(writer_compile_time_args, buffers[SOURCE_ROWS]);
    const uint32_t writer_routing_accessor_offset = u32(writer_compile_time_args.size());
    writer_compile_time_args.insert(
        writer_compile_time_args.end(), routing_compile_time_args.begin(), routing_compile_time_args.end());
    const uint32_t route_accessor_compile_time_args_offset = u32(writer_compile_time_args.size());
    ComputeConfigDescriptor compute_config{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi3,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = false,
        .math_approx_mode = false,
    };
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    for (uint8_t i : {ROUTER_CB_SCORES, ROUTER_CB_BEST, ROUTER_CB_LOGITS, ROUTER_CB_ARRIVALS, ROUTER_CB_BIAS}) {
        compute_config.unpack_to_dest_mode[i] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    const Args common = {arena_base, router_arrival1};
    // a padded chunk (routing_planner.hpp PADDED_CHUNK): the planners and the route words' writer read actual_start
    // / actual_end (common [2], [3]) and bound this chip's tokens by its SP row's real rows (main's rotated layout: [4]
    // the SP row, [5] SP rows, [6] tokens a row) less the row's tokens before mine ([7] position x tokens_per_chip,
    // my owned block position), [8] tokens_per_chip; absent, the programs are the unpadded ones. Row blocks: the row
    // holds row-block sub-chunks of tokens_per_chip x mesh_columns tokens one after another, this call's the
    // block-th - [6] the whole row's tokens, [7] the sub-chunk's first row in it plus position x tokens_per_chip
    // ([9] the sub-chunk's rows, [10] the row blocks, [11] position x tokens_per_chip: the planner counts every
    // block's real tokens of mine)
    const bool padded = buffers[SOURCE_START] != nullptr;
    Args common_with_padding = common;
    if (padded) {  // (splits: the row holds splits x blocks calls' rows; this split's first is split x row_blocks)
        const uint32_t sub_chunk_tokens = model.tokens_per_chip * model.mesh_columns,
                       first_row_block = buffers.split * buffers.row_blocks;
        add(common_with_padding,
            Args{
                source_arg(SOURCE_START),
                source_arg(SOURCE_END),
                Arg(coord.first),
                Arg(model.mesh_rows),
                Arg(model.config.shape.row_tokens ? model.config.shape.row_tokens
                                                  : sub_chunk_tokens * buffers.row_blocks * model.splits),
                Arg((first_row_block + buffers.block) * sub_chunk_tokens + position * model.tokens_per_chip),
                Arg(model.tokens_per_chip),
                Arg(sub_chunk_tokens),
                Arg(buffers.row_blocks),
                Arg(first_row_block * sub_chunk_tokens + position * model.tokens_per_chip)});
    }
    auto iteration_defines =
        [&](KernelDescriptor& kernel_descriptor) {  // (router iterations: ROUTER_ITERATIONS_ON_ROUTER of them,
                                                    // ROUTER_ROW_TILES_PER_ITERATION router cores)
            if (model.router_iterations > 1) {
                define(kernel_descriptor, "ROUTER_ITERATIONS_ON_ROUTER", integer_literal(model.router_iterations));
                define(
                    kernel_descriptor,
                    "ROUTER_ROW_TILES_PER_ITERATION",
                    integer_literal(model.router_row_tiles_per_iteration));
                define(
                    kernel_descriptor,
                    "ROUTER_GO_WORD",
                    integer_literal(
                        160));  // (the matmul cores' go word in their ewords: the engine's ROUTER_GO_FLAG_ID)
            }
        };
    auto router_defines = [&](KernelDescriptor& kernel_descriptor) {
        iteration_defines(kernel_descriptor);
        define(kernel_descriptor, "ROUTER_MATH_SYNC_OFFSET", unsigned_literal(region_of_router("msync")));
        define(kernel_descriptor, "ROUTER_FOLD", unsigned_literal(model.config.shape.fold));
    };
    const SpanTables spans = span_subs(model.mesh_rows, model.mesh_columns, model.ring, model.wrap_y);
    auto join = [](const auto& items) {
        std::string joined;
        for (size_t i = 0; i < items.size(); ++i) {
            joined += (i ? "," : "") + std::to_string(uint32_t(items[i]));
        }
        return joined;
    };
    std::vector<KernelDescriptor> kernels;
    {
        auto set_core_runtime_args = [&](const std::vector<std::pair<CoreCoord, Args>>& runtime_args_list,
                                         KernelDescriptor& kernel_descriptor) {
            for (const auto& [core, args] : runtime_args_list) {
                set_runtime_args(kernel_descriptor, core, args);
            }
        };
        auto input_feeder = kernel(
            "dataflow/router_input_feeder.cpp",
            router_core_set,
            data_movement(DataMovementProcessor::RISCV_1, NOC::NOC_0));
        router_defines(input_feeder);
        planner_defines(input_feeder, feeder_routing_accessor_offset);
        router_planner_none(input_feeder);
        input_feeder.compile_time_args = feeder_compile_time_args;
        set_core_runtime_args(input_feeder_runtime_args, input_feeder);
        if (padded) {
            padded_chunk_defines(input_feeder);
        }
        set_common(input_feeder, common_with_padding);
        routing_accessors(input_feeder);
        auto router_compute = kernel("compute/router_compute.cpp", router_core_set, compute_config);
        router_defines(router_compute);
        if (model.config.sqrt_softplus) {
            define(router_compute, "ROUTER_SQRTSOFTPLUS");
            define(router_compute, "INP_FLOAT32", "1");
        }
        define(router_compute, "ROUTER_ROUTE_WORDS");
        if (padded) {  // (its row tiles of padding only: no sums, top-K, rows or route words - router_compute.cpp)
            padded_chunk_defines(router_compute);
        }
        define(router_compute, "ROUTE_WORDS_SP_ROWS", integer_literal(model.mesh_rows));
        define(router_compute, "ROUTE_WORDS_TP_COLUMNS", integer_literal(model.mesh_columns));
        define(router_compute, "ROUTE_WORDS_MY_CHIP", integer_literal(my_chip));
        define(
            input_feeder,
            "ROUTER_MY_CHIP",
            integer_literal(my_chip));  // (the compute's chip index and TP position, through the RDY page: one program
        define(input_feeder, "ROUTER_TP_POSITION", integer_literal(position));  // for the mesh)
        define(router_compute, "ROUTER_SCORE_STRIDE", integer_literal(int64_t(expert_tiles) * kFp32TileBytes));
        define(router_compute, "ROUTE_WORDS_TOP_K", integer_literal(top_k));
        define(router_compute, "ROUTE_WORDS_PER_TOKEN", integer_literal(route_words(model.num_chips)));
        define(
            router_compute, "ROUTE_WORDS_EXPERTS", integer_literal(model.expert_tiles * 32 / model.config.shape.fold));
        define(router_compute, "ROUTE_WORDS_WRAP_X", spans.wrap_x ? "true" : "false");
        define(router_compute, "ROUTE_WORDS_WRAP_Y", spans.wrap_y ? "true" : "false");
        define(router_compute, "ROUTE_WORDS_X_RING_INIT", join(spans.xring));
        define(router_compute, "ROUTE_WORDS_Y_RING_INIT", join(spans.yring));
        std::vector<uint32_t> expert_chips;
        for (uint32_t expert = 0; expert < model.expert_tiles * 32 / model.config.shape.fold; ++expert) {
            expert_chips.push_back(expert / model.experts_per_chip);
        }
        define(router_compute, "ROUTE_WORDS_EXPERT_CHIPS_INIT", join(expert_chips));
        router_compute.compile_time_args = {
            router_compile_time_args.begin() + 1,
            router_compile_time_args.end()};  // (no router_k_tiles: the readers' alone)
        set_core_runtime_args(router_compute_runtime_args, router_compute);
        set_common(router_compute, common);
        auto output_writer = kernel(
            "dataflow/router_output_writer.cpp",
            router_core_set,
            data_movement(DataMovementProcessor::RISCV_0, NOC::NOC_1));
        define(output_writer, "ROUTER_FOLD", unsigned_literal(model.config.shape.fold));
        output_writer.compile_time_args = writer_compile_time_args;
        define(output_writer, "ROUTER_ROUTE_WORDS");
        define(
            output_writer, "ROUTER_ROUTE_ACCESSOR_CT_OFFSET", integer_literal(route_accessor_compile_time_args_offset));
        define(output_writer, "ROUTER_ROUTE_PAGE_BYTES", integer_literal(buffers[SOURCE_ROUTE]->aligned_page_size()));
        define(output_writer, "ROUTER_ROUTE_WORDS_PER_TOKEN", integer_literal(route_words(model.num_chips)));
        define(
            output_writer,
            "ROUTER_ITERATION_ARGS",
            integer_literal(model.router_iterations > 1 ? 1 + iteration_args.size() : 0));
        iteration_defines(output_writer);
        append_accessor(output_writer.compile_time_args, buffers[SOURCE_ROUTE]);
        planner_defines(output_writer, writer_routing_accessor_offset);
        router_planner_none(output_writer);
        set_core_runtime_args(output_writer_runtime_args, output_writer);
        if (padded) {
            padded_chunk_defines(output_writer);
        }
        set_common(output_writer, common_with_padding);
        routing_accessors(output_writer);
        push(kernels, input_feeder);
        push(kernels, router_compute);
        push(kernels, output_writer);
    }
    // row blocks: the planner-only cores (instances: their index past the router cores, and that index + planner_cores)
    if (model.planner_cores.size() > model.router_cores.size()) {
        const std::vector<CoreCoord> planner_only_cores(
            model.planner_cores.begin() + model.router_cores.size(), model.planner_cores.end());
        const auto planner_args_of = planner_args();
        for (const bool second : {false, true}) {
            auto planner_kernel = kernel(
                "dataflow/routing_planner_core.cpp",
                cells(planner_only_cores),
                second ? data_movement(DataMovementProcessor::RISCV_1, NOC::NOC_0)
                       : data_movement(DataMovementProcessor::RISCV_0, NOC::NOC_1));
            planner_kernel.compile_time_args = routing_compile_time_args;
            planner_defines(planner_kernel, 0);
            if (padded) {
                padded_chunk_defines(planner_kernel);
            }
            for (uint32_t i = 0; i < planner_only_cores.size(); ++i) {
                const int32_t instance = model.instance_at(u32(model.router_cores.size()) + i, second ? 1 : 0);
                Args args = {routing_words.at(second ? "rows2" : "rows")};
                if (instance >= 0) {
                    args.push_back(arena_base + u32(planner_bases()[instance]));
                    add(args, planner_args_of(u32(instance)));
                } else {
                    args.push_back(0u);
                }
                set_runtime_args(planner_kernel, planner_only_cores[i], args);
            }
            set_common(planner_kernel, common_with_padding);
            routing_accessors(planner_kernel);
            push(kernels, planner_kernel);
        }
    }
    return kernels;
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head
