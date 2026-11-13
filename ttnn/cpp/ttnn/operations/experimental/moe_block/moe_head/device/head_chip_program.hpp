// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One chip's program of the MoE head (ChipProgram): one builder a core role - the matmul cores' (shared expert)
// kernels (head_engine_program.cpp), the x gather's port kernels and the Y senders (head_gather_program.cpp), the
// router and the routing exchange (head_router_program.cpp) - assembled by build (head_program.cpp). Internal to
// moe_head.
#pragma once

#include <functional>
#include <set>

#include "head_model.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

// One chip's program.
class ChipProgram {
public:
    ChipProgram(const HeadModel& model, const HeadBuffers& buffers, Coord coord) :
        model(model), sizing(*model.sizing), reduce_scatter(*model.reduce_scatter), buffers(buffers), coord(coord) {
        position = model.position(coord);
        my_chip = coord.first * model.mesh_columns + coord.second;
        arena_base = source_arg(SOURCE_ARENA);
        mailbox = word("mbox");
        for (uint32_t i = 0; i < (sizing.row_chunks > 1 ? 4u : 3u); ++i) {
            engine_arrival_words.push_back(word("arr") + 16 * i);
        }
        zone_free = mailbox + 4 * model.zone_words_offset;
        const Arg x_gather_word = word("xg");
        arrival = x_gather_word;
        ready = x_gather_word + 16;
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t link = 0; link < model.gather_links; ++link) {
                watch[{static_cast<XDirection>(i), link}] = x_gather_word + 16 * (2 + i * model.gather_links + link);
            }
        }
        x_loaded = x_gather_word + 16 * (2 + 2 * model.gather_links);
        router_arrival0 = word("slot_words");
        router_arrival1 = word("rt") + 16;
        slots_free = word("sfree");
        const std::vector<std::string> planner_word_keys = {
            "calls_p", "calls_p2", "go", "go2", "parts", "ex", "ex2", "rows", "rows2"};
        for (uint32_t i = 0; i < planner_word_keys.size(); ++i) {
            routing_words[planner_word_keys[i]] = word("ra_p") + 16 * i;
        }
        routing_words["c0"] = word("ra_c");
        routing_words["c1"] = word("ra_c") + 16;
        const std::vector<std::string> sender_word_keys = {"calls_s", "calls_s2", "ready", "ready2"};
        for (uint32_t i = 0; i < sender_word_keys.size(); ++i) {
            routing_words[sender_word_keys[i]] = word("ra_s") + 16 * i;
        }
        for (const auto& [direction, link, core] : model.chip(coord).ports) {
            port_virtual_cores[{direction, link}] = model.virtual_core(core);
        }
        senders = model.serves.at(my_chip);
    }

    ProgramDescriptor build(HeadPatches* patches);
    // One program for the mesh: this chip's own scalars (its index, its ports, the watcher's second Y block) as
    // runtime values - a data-movement kernel's ROUTING_EXCHANGE_MY_CHIP / ROUTING_PLANNER_MY_CHIP /
    // ROUTING_PLANNER_PORTS / ROUTING_PLANNER_NUM_PORTS / ROUTER_MY_CHIP / ROUTER_TP_POSITION /
    // WATCHER_ROUTING_ARGS_OFFSET read trailing common args, which head_args_to_dram moves into each chip's DRAM table
    // (the values differ by chip, the defines do not); the router compute gets its index and TP position in the RDY
    // page (ROUTER_CHIP_FROM_READY_PAGE).
    void chip_values();

private:
    const HeadModel& model;
    const SharedExpertSizing& sizing;
    const ReduceScatterPlan& reduce_scatter;
    const HeadBuffers& buffers;
    Coord coord;
    uint32_t position = 0, my_chip = 0;
    Arg arena_base, mailbox, zone_free, arrival, ready, x_loaded, router_arrival0, router_arrival1, slots_free;
    std::vector<Arg> engine_arrival_words;
    std::map<std::pair<XDirection, uint32_t>, Arg> watch;
    std::map<std::string, Arg> routing_words;
    // the address args of each kernel: by its (local) descriptor's address while it is built, then - push moves it into
    // its function's kernel list - in pushed, one entry per kernel in list order (a descriptor's address does not
    // survive the moves; take_pushed hands a list's entries to the final assembly)
    std::map<const KernelDescriptor*, std::vector<HeadPatches::Arg>> kernel_patches;
    std::vector<std::vector<HeadPatches::Arg>> pushed;
    void push(std::vector<KernelDescriptor>& kernels, KernelDescriptor& kernel_descriptor) {
        auto patches_entry = kernel_patches.find(&kernel_descriptor);
        pushed.push_back(
            patches_entry == kernel_patches.end() ? std::vector<HeadPatches::Arg>{} : std::move(patches_entry->second));
        if (patches_entry != kernel_patches.end()) {
            kernel_patches.erase(patches_entry);
        }
        kernels.push_back(std::move(kernel_descriptor));
    }
    std::vector<std::vector<HeadPatches::Arg>> take_pushed() { return std::exchange(pushed, {}); }

    // a source's address arg (0 for an absent tensor)
    Arg source_arg(HeadSource source) const {
        return Arg(buffers.addresses[source], buffers.addresses[source] ? int32_t(source) : -1, 0);
    }
    // a core's / the common args of a kernel, their addresses recorded
    void set_runtime_args(KernelDescriptor& kernel_descriptor, const CoreCoord& core, const Args& args) {
        for (uint32_t i = 0; i < args.size(); ++i) {
            if (args[i].source >= 0) {
                kernel_patches[&kernel_descriptor].push_back(
                    {0, false, core, i, uint16_t(args[i].source), args[i].offset});
            }
        }
        kernel_descriptor.runtime_args.emplace_back(core, values(args));
    }
    void set_common(KernelDescriptor& kernel_descriptor, const Args& args) {
        for (uint32_t i = 0; i < args.size(); ++i) {
            if (args[i].source >= 0) {
                kernel_patches[&kernel_descriptor].push_back(
                    {0, true, CoreCoord{}, i, uint16_t(args[i].source), args[i].offset});
            }
        }
        kernel_descriptor.common_runtime_args = values(args);
    }
    std::map<std::pair<XDirection, uint32_t>, CoreCoord> port_virtual_cores;
    std::map<SendKey, std::vector<uint32_t>> senders;
    ProgramDescriptor program_descriptor;

    Arg word(const std::string& name) const {
        const uint32_t index = model.word_index(name);
        return Arg(buffers.addresses[SOURCE_FIRST_WORD + index], int32_t(SOURCE_FIRST_WORD + index), 0);
    }
    Arg arena_arg(const std::string& region) const { return arena_base + u32(sizing.at(region)); }
    Arg router_arena_arg(const std::string& region) const { return arena_base + u32(region_of_router(region)); }
    int64_t region_of_router(const std::string& region) const {
        for (const auto& [name, offset] : model.router_offsets) {
            if (name == region) {
                return offset;
            }
        }
        TT_THROW("moe_head: no router region {}", region);
    }
    CBDescriptor arena_cb(
        uint8_t cb_index,
        int64_t offset,
        uint32_t size,
        tt::DataFormat data_format,
        uint32_t page_size,
        const CoreRangeSet& cores) const {
        CBDescriptor descriptor;
        descriptor.total_size = size;
        descriptor.core_ranges = cores;
        descriptor.format_descriptors = {
            CBFormatDescriptor{.buffer_index = cb_index, .data_format = data_format, .page_size = page_size}};
        descriptor.buffer = buffers[SOURCE_ARENA];
        descriptor.address_offset = u32(offset);
        return descriptor;
    }

    // the pieces
    std::vector<uint32_t> shard_order_args() const;
    std::vector<KernelDescriptor> engine(std::vector<CBDescriptor>& cbs);
    std::vector<KernelDescriptor> gather(std::vector<CBDescriptor>& cbs);
    std::vector<KernelDescriptor> router(std::vector<CBDescriptor>& cbs);
    std::vector<KernelDescriptor> y_kernels();
    Args reduce_scatter_row_dataflow_args(uint32_t core_x, uint32_t core_y) const;
    Args latent_row_dataflow_args(uint32_t core_x, uint32_t core_y) const;
    Args latent_column_dataflow_args(uint32_t core_y) const;
    Args reduce_scatter_column_dataflow_args(uint32_t core_x, uint32_t core_y) const;
    Args prep_column_dataflow_args(uint32_t core_x, uint32_t core_y) const;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> prep_units(
        uint32_t core_x, uint32_t core_y) const;
    std::vector<uint32_t> prep_compute(
        const std::vector<uint32_t>& compute_args, size_t num_prefix_args, uint32_t core_x, uint32_t core_y) const;
    std::vector<Args> engine_destinations() const;
    Args reduce_scatter_block(XDirection direction, uint32_t link) const;
    Args latent_block(XDirection direction, uint32_t link) const;
    Args logits_block(XDirection direction, uint32_t link) const;
    Args port_block(XDirection direction, uint32_t link) const;
    Args send_block(const SendKey& key) const;
    Args y_args(const SendKey& key, const CoreCoord& core, bool with_hellos = true);
    void routing_defines(KernelDescriptor& kernel_descriptor) const;
    // The routing kernels' DRAM tensors (routing_planner.hpp routing_accessors): their accessor args after the kernel's
    // own compile-time args, from ROUTING_ACCESSORS_COMPILE_TIME_ARGS_OFFSET on - plan, counts, regions, token index,
    // table, slots, forwarding slots, the padded chunk's start / end scalars (an interleaved DRAM placeholder for one
    // this call has none of).
    void routing_accessors(KernelDescriptor& kernel_descriptor) const {
        define(
            kernel_descriptor,
            "ROUTING_ACCESSORS_COMPILE_TIME_ARGS_OFFSET",
            integer_literal(kernel_descriptor.compile_time_args.size()));
        for (HeadSource source :
             {SOURCE_PLAN,
              SOURCE_COUNTS,
              SOURCE_REGIONS,
              SOURCE_TOKEN_INDEX,
              SOURCE_TABLE,
              SOURCE_SLOTS,
              SOURCE_START,
              SOURCE_END}) {
            (buffers[source] ? tt::tt_metal::TensorAccessorArgs(*buffers[source])
                             : tt::tt_metal::TensorAccessorArgs::create_dram_interleaved())
                .append_to(kernel_descriptor.compile_time_args);
        }
    }
    void planner_defines(KernelDescriptor& kernel_descriptor, uint32_t accessor_compile_time_args_offset) const;
    // SKIP_PADDING_CHUNKS (a padded forward; no latent, row chunks): the engine, the gather's ports and the watcher run
    // the row chunks up to the last with a real row only (padded_rows.hpp padded_live_chunks) - the matmul cores' BRISC
    // tells its NCRISC and compute (ewords + LIVE_CHUNKS_FLAG_ID)
    bool skip_padding() const {
        return buffers[SOURCE_START] != nullptr && !sizing.latent_tiles && sizing.row_chunks > 1;
    }
    void skip_defines(KernelDescriptor& kernel_descriptor, uint32_t args_offset) const {
        define(kernel_descriptor, "SKIP_PADDING_CHUNKS");
        define(kernel_descriptor, "SKIP_PADDING_ARGS_OFFSET", integer_literal(args_offset));
        define(kernel_descriptor, "LIVE_CHUNKS_FLAG_ID", integer_literal(240));
        define(kernel_descriptor, "PADDED_FORWARD_CHUNKS", integer_literal(model.config.shape.forward_chunks));
        if (buffers.padded_forward_metadata) {
            define(kernel_descriptor, "PADDED_FORWARD_METADATA");
        }
    }
    Args live_args() const {
        const uint32_t sub_chunk_tokens = model.tokens_per_chip * model.mesh_columns,
                       first_row_block = buffers.split * buffers.row_blocks;
        const bool per_owner = sizing.row_chunks > 1 && sizing.shards > 1;
        return Args{
            source_arg(SOURCE_START),
            buffers[SOURCE_END] ? source_arg(SOURCE_END) : Arg(0u),
            Arg(coord.first),
            Arg(model.mesh_rows),
            Arg(model.config.shape.row_tokens ? model.config.shape.row_tokens
                                              : sub_chunk_tokens * buffers.row_blocks * model.splits),
            Arg((first_row_block + buffers.block) * sub_chunk_tokens),
            Arg(sizing.row_chunks),
            Arg(sizing.chunk_row_tiles),
            Arg(per_owner ? sizing.chunk_row_tiles / sizing.shards : 0u),
            Arg(per_owner ? sizing.rows / 32 / sizing.shards : 0u),
            word("valid")};
    }
    // a padded forward (routing_planner.hpp PADDED_CHUNK): its chunks a forward (PADDED_FORWARD_CHUNKS) and whether
    // it comes as the forward's metadata words (PADDED_FORWARD_METADATA) rather than one chunk's actual_start /
    // actual_end
    void padded_chunk_defines(KernelDescriptor& kernel_descriptor) const {
        define(kernel_descriptor, "PADDED_CHUNK");

        define(kernel_descriptor, "PADDED_FORWARD_CHUNKS", integer_literal(model.config.shape.forward_chunks));
        if (buffers.padded_forward_metadata) {
            define(kernel_descriptor, "PADDED_FORWARD_METADATA");
        }
    }
    // row blocks in one program with every planner instance on the planner-only cores: the router kernels compile no
    // planner (routing_planner.hpp ROUTING_PLANNER_NONE)
    void router_planner_none(KernelDescriptor& kernel_descriptor) const {
        if (buffers.merged && buffers.row_blocks > 1 && model.planner_instances <= model.num_planner_only_cores) {
            define(kernel_descriptor, "ROUTING_PLANNER_NONE");
        }
    }
    // row blocks: an earlier block's planner plans no token (the exchange runs empty: the chips' barrier)
    bool idle_planner() const { return buffers.block + 1 < buffers.row_blocks; }
    // (row blocks in one program: the last block's - every block's tokens - with the earlier blocks' after them, the
    // kernel picks: routing_planner.hpp ROUTING_PLANNER_PER_BLOCK)
    bool dual_planner() const { return buffers.merged && buffers.row_blocks > 1; }
    const std::vector<int64_t>& planner_bases() const {
        return idle_planner() && !dual_planner() ? model.idle_planner_bases : model.planner_bases;
    }
    std::function<Args(uint32_t)> planner_args() const;
    Args routing_output_writer_args(uint32_t router_core_index) const;
    Args routing_input_feeder_args(uint32_t router_core_index) const;
    std::vector<uint32_t> fabric_connection_args(const Coord& peer, uint32_t link, const CoreCoord& core) {
        std::vector<uint32_t> out;
        tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
            *model.mesh, model.node(coord), model.node(peer), link, program_descriptor, core, out);
        return out;
    }
};

// Row blocks (head_row_blocks.cpp)
namespace row_block_detail {
std::vector<CoreCoord> cores_in(const CoreRangeSet& core_ranges);
void check_row_block_args(
    const ProgramDescriptor& first_block,
    const ProgramDescriptor& second_block,
    const ProgramDescriptor& last_block,
    uint32_t row_blocks,
    const std::set<uint32_t>& deltas);
void add_row_block_barrier(
    ProgramDescriptor& program,
    uint32_t row_blocks,
    uint32_t words,
    const std::function<CoreCoord(const CoreCoord&)>& virtual_core_of,
    const std::vector<std::pair<std::string, std::string>>& defines);
}  // namespace row_block_detail

}  // namespace ttnn::operations::experimental::moe_block::moe_head
