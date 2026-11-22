// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The head's kernel args into per-chip DRAM tables, the mesh workload (build_head_workload) with what a cache hit
// writes, and the engine's mcast families.
#include "head_chip_program.hpp"

namespace ttnn::operations::bringup::moe_block::moe_head {

namespace head_args_detail {

using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::DataMovementProcessor;
using Chips = std::vector<std::pair<ttnn::MeshCoordinate, ProgramDescriptor>>;
using Key = std::pair<size_t, size_t>;

// the compute kernels' args page (a CB index no head piece uses)
constexpr uint8_t kArgsCb = 31;

bool is_compute(const KernelDescriptor& kernel_descriptor) {
    return std::holds_alternative<ComputeConfigDescriptor>(kernel_descriptor.config);
}
bool on_riscv0(const KernelDescriptor& kernel_descriptor) {
    const auto* config = std::get_if<DataMovementConfigDescriptor>(&kernel_descriptor.config);
    return config != nullptr && config->processor == DataMovementProcessor::RISCV_0;
}

}  // namespace head_args_detail

// Every kernel's args into per-chip DRAM tables (common/args_table.hpp runtime_args_to_dram), so the chips'
// programs become alike - one program for the mesh, nothing per chip or per core written on a launch.
//  * A data-movement kernel's block on a core: its runtime args (padded to the kernel's longest), then its common args
//    (from the named arg moe_block_args_common_at: moe_block_args::common reads them there - they hold chip values
//    too, chip_values), then on the RISCV_0 kernel of a core with a compute kernel the compute kernel's runtime +
//    common args (from moe_block_args_peer_at): a compute kernel cannot read DRAM, so its data-movement kernel hands
//    them over as one page of CB kArgsCb.
//  * Every address arg (HeadAddressArgs) becomes a fixup: the table holds its offset, the device adds its source's
//    address from the kernel's common args - which a cache hit rewrites (write_program_addresses).
//  * The blocks land in the arena's args region (HeadModel.args_offset, kHeadArgsBytes a core, past every other arena
//    region): the RISCV_0 kernels' blocks at its start, the RISCV_1 kernels' after the largest of those, the compute
//    page after the largest RISCV_1 block. The region is checked against kHeadArgsBytes here.
// Returns the tables and the converted kernels (what a cache hit writes into their common args).
ProgramAddresses head_args_to_dram(
    const HeadModel& model,
    const HeadBuffers& buffers,
    head_args_detail::Chips& chips,
    const std::vector<HeadAddressArgs>& address_args) {
    using namespace head_args_detail;
    const size_t num_chips = chips.size();
    TT_FATAL(
        num_chips > 0 && address_args.size() == num_chips,
        "moe_head: args to DRAM: {} programs, {} address arg lists",
        num_chips,
        address_args.size());
    const size_t num_kernels = chips[0].second.kernels.size();
    const auto& first_kernels = chips[0].second.kernels;
    // each kernel's longest runtime / common args over every chip and core
    std::vector<uint32_t> runtime_words(num_kernels, 0), common_words(num_kernels, 0);
    for (const auto& [_, program] : chips) {
        TT_FATAL(
            program.kernels.size() == num_kernels,
            "moe_head: args to DRAM: the chips' programs have different kernels");
        for (size_t i = 0; i < num_kernels; ++i) {
            for (const auto& [core, args] : program.kernels[i].runtime_args) {
                runtime_words[i] = std::max(runtime_words[i], u32(args.size()));
            }
            common_words[i] = std::max(common_words[i], u32(program.kernels[i].common_runtime_args.size()));
        }
    }
    // a compute kernel's args ride the RISCV_0 kernel over its cores (page_words: the longest compute list - the page)
    std::vector<int> host_kernel(num_kernels, -1);
    uint32_t page_words = 0;
    CoreRangeSet compute_cores;
    for (size_t compute_index = 0; compute_index < num_kernels; ++compute_index) {
        if (!is_compute(first_kernels[compute_index])) {
            continue;
        }
        for (size_t j = 0; j < num_kernels && host_kernel[compute_index] < 0; ++j) {
            if (on_riscv0(first_kernels[j]) &&
                first_kernels[j].core_ranges.contains(first_kernels[compute_index].core_ranges)) {
                host_kernel[compute_index] = int(j);
            }
        }
        TT_FATAL(
            host_kernel[compute_index] >= 0,
            "moe_head: compute kernel {} ({}): no RISCV_0 kernel on its cores",
            compute_index,
            first_kernels[compute_index].kernel_source);
        page_words = std::max(page_words, runtime_words[compute_index] + common_words[compute_index]);
        compute_cores = compute_cores.merge(first_kernels[compute_index].core_ranges);
    }
    std::vector<bool> is_host(num_kernels, false);
    for (size_t compute_index = 0; compute_index < num_kernels; ++compute_index) {
        if (host_kernel[compute_index] >= 0) {
            is_host[host_kernel[compute_index]] = true;
        }
    }
    // a data-movement kernel's block words: [runtime_words][common_words][the compute page's page_words, if it hosts
    // one]
    auto words = [&](size_t kernel_index) {
        return runtime_words[kernel_index] + common_words[kernel_index] + (is_host[kernel_index] ? page_words : 0u);
    };

    // each chip's new per-core lists (the args as built - addresses whole) and fixups
    std::vector<ArgsTableKernel> fixup_kernels;
    std::vector<int> fixup_kernel_of(num_kernels, -1);
    for (size_t k = 0; k < num_kernels; ++k) {
        if (!is_compute(first_kernels[k])) {
            fixup_kernel_of[k] = int(fixup_kernels.size());
            ArgsTableKernel fixup_kernel;
            fixup_kernel.kernel = k;
            fixup_kernel.fixups.resize(num_chips);
            fixup_kernel.kept.resize(num_chips);
            fixup_kernels.push_back(std::move(fixup_kernel));
        }
    }
    std::vector<std::vector<std::map<Key, std::vector<uint32_t>>>> lists(
        num_chips, std::vector<std::map<Key, std::vector<uint32_t>>>(num_kernels));
    for (size_t chip_index = 0; chip_index < num_chips; ++chip_index) {
        auto& program = chips[chip_index].second;
        auto runtime_args_of = [&](size_t kernel_index, const CoreCoord& core) -> std::vector<uint32_t> {
            for (const auto& [arg_core, args] : program.kernels[kernel_index].runtime_args) {
                if (arg_core == core) {
                    return std::vector<uint32_t>(args.begin(), args.end());
                }
            }
            return {};
        };
        auto put = [](std::vector<uint32_t>& list, size_t offset, const auto& source_words) {
            std::copy(source_words.begin(), source_words.end(), list.begin() + offset);
        };
        for (size_t k = 0; k < num_kernels; ++k) {
            if (is_compute(first_kernels[k])) {
                continue;
            }
            for (const auto& core : row_block_detail::cores_in(program.kernels[k].core_ranges)) {
                auto& list = lists[chip_index][k][{core.x, core.y}];
                list.assign(words(k), 0u);
                put(list, 0, runtime_args_of(k, core));
                put(list, runtime_words[k], program.kernels[k].common_runtime_args);
            }
        }
        for (size_t compute_index = 0; compute_index < num_kernels; ++compute_index) {
            if (host_kernel[compute_index] < 0) {
                continue;
            }
            const size_t host_index = host_kernel[compute_index];
            for (const auto& core : row_block_detail::cores_in(program.kernels[compute_index].core_ranges)) {
                auto& list = lists[chip_index][host_index].at({core.x, core.y});
                put(list, runtime_words[host_index] + common_words[host_index], runtime_args_of(compute_index, core));
                put(list,
                    runtime_words[host_index] + common_words[host_index] + runtime_words[compute_index],
                    program.kernels[compute_index].common_runtime_args);
            }
        }
        // the address args: a fixup each (a common arg's on every core of its kernel)
        for (const auto& patch_arg : address_args[chip_index].args) {
            if (patch_arg.fabric_connection) {  // (a data-movement kernel's own runtime args: kept, from index 0)
                fixup_kernels[fixup_kernel_of[patch_arg.kernel]]
                    .kept[chip_index][{patch_arg.core.x, patch_arg.core.y}] = {patch_arg.index, kFabricConnectionArgs};
                continue;
            }
            const bool compute = host_kernel[patch_arg.kernel] >= 0;
            const size_t block_kernel = compute ? size_t(host_kernel[patch_arg.kernel]) : patch_arg.kernel;
            const uint32_t word_index = (compute ? runtime_words[block_kernel] + common_words[block_kernel] : 0u) +
                                        (patch_arg.common ? runtime_words[patch_arg.kernel] : 0u) + patch_arg.index;
            const auto cores = patch_arg.common
                                   ? row_block_detail::cores_in(program.kernels[patch_arg.kernel].core_ranges)
                                   : std::vector<CoreCoord>{patch_arg.core};
            for (const auto& core : cores) {
                fixup_kernels[fixup_kernel_of[block_kernel]].fixups[chip_index][{core.x, core.y}].push_back(
                    {word_index, patch_arg.source});
            }
        }
    }
    // the args region: RISCV_0 blocks, then RISCV_1 blocks, then the compute page
    std::vector<uint32_t> bytes(num_kernels, 0);
    uint32_t riscv0_bytes = 0, riscv1_bytes = 0;
    for (size_t k = 0; k < num_kernels; ++k) {
        if (is_compute(first_kernels[k])) {
            continue;
        }
        size_t num_fixups = 0;
        for (const auto& per_chip : fixup_kernels[fixup_kernel_of[k]].fixups) {
            for (const auto& [_, fixups] : per_chip) {
                num_fixups = std::max(num_fixups, fixups.size());
            }
        }
        bytes[k] = args_block_bytes(words(k), u32(num_fixups));
        (on_riscv0(first_kernels[k]) ? riscv0_bytes : riscv1_bytes) =
            std::max(on_riscv0(first_kernels[k]) ? riscv0_bytes : riscv1_bytes, bytes[k]);
    }
    const uint32_t page = u32(align_to_64(4 * int64_t(page_words)));
    TT_FATAL(
        int64_t(riscv0_bytes) + riscv1_bytes + page <= kHeadArgsBytes,
        "moe_head: the kernels' args blocks need {} (RISCV_0) + {} (RISCV_1) + {} (compute page) B a core, the arena's "
        "args region holds {} (kHeadArgsBytes)",
        riscv0_bytes,
        riscv1_bytes,
        page,
        kHeadArgsBytes);
    const uint32_t args_offset = u32(model.args_offset);
    for (auto& fixup_kernel : fixup_kernels) {
        fixup_kernel.l1_offset = on_riscv0(first_kernels[fixup_kernel.kernel]) ? 0u : riscv0_bytes;
        fixup_kernel.bytes = on_riscv0(first_kernels[fixup_kernel.kernel]) ? riscv0_bytes : riscv1_bytes;
    }

    // the programs: the lists as each kernel's runtime args, no common args left (the converter adds its own), the
    // named args saying where things sit, the compute page's CB
    for (size_t chip_index = 0; chip_index < num_chips; ++chip_index) {
        auto& program = chips[chip_index].second;
        for (size_t k = 0; k < num_kernels; ++k) {
            auto& kernel_descriptor = program.kernels[k];
            kernel_descriptor.common_runtime_args.clear();
            kernel_descriptor.runtime_args.clear();
            if (is_compute(kernel_descriptor)) {
                if (host_kernel[k] >= 0) {
                    set_args_table_compute_reads_page(kernel_descriptor, kArgsCb, runtime_words[k]);
                }
                fill_args_table_named_args(kernel_descriptor);
                continue;
            }
            for (auto& [core_key, list] : lists[chip_index][k]) {
                kernel_descriptor.runtime_args.emplace_back(
                    CoreCoord(core_key.first, core_key.second), std::move(list));
            }
            set_args_table_common_in_block(kernel_descriptor, runtime_words[k]);
            if (is_host[k]) {
                set_args_table_peer_page_sender(
                    kernel_descriptor, kArgsCb, runtime_words[k] + common_words[k], page_words);
            }
        }
        if (page_words > 0) {
            program.cbs.push_back(arena_cb(
                buffers[SOURCE_ARENA],
                kArgsCb,
                args_offset + riscv0_bytes + riscv1_bytes,
                page,
                tt::DataFormat::UInt32,
                page,
                compute_cores));
        }
    }
    std::string why;
    auto converted = runtime_args_to_dram(
        *model.mesh, chips, fixup_kernels, buffers.addresses, buffers.addresses[SOURCE_ARENA] + args_offset, why);
    TT_FATAL(converted.tables != nullptr, "moe_head: the kernels' args to DRAM: {}", why);
    // (no common arg is left before the converter's: every kernel's common index is 0, on every chip)
    for (const auto& common_index : converted.common_index) {
        for (uint32_t index : common_index) {
            TT_FATAL(index == 0, "moe_head: args to DRAM: a kernel kept common args ({})", index);
        }
    }
    return ProgramAddresses{
        .tables = converted.tables,
        .table_kernels = table_kernels(converted, fixup_kernels, 0, SOURCE_ARENA, args_offset),
        .args = {},
        .arena_cbs = {}};
}

// The engine BRISC's kernel_lib families: its x slots 0 / 1 and h (a receiver group a row). Attached to the finished
// program (the args in DRAM, the chips merged): their runtime args stay runtime args - the same on every chip.
void attach_engine_mcasts(const HeadModel& model, ProgramDescriptor& program_descriptor) {
    namespace kernel_lib_host = ttnn::kernel_lib::host;
    const SharedExpertSizing& sizing = *model.sizing;
    auto engine_kernel = std::find_if(
        program_descriptor.kernels.begin(),
        program_descriptor.kernels.end(),
        [](const KernelDescriptor& kernel_descriptor) {
            return kernel_descriptor.kernel_source.ends_with("dataflow/shared_expert_row_dataflow.cpp");
        });
    TT_FATAL(engine_kernel != program_descriptor.kernels.end(), "moe_head: no engine BRISC kernel");
    const auto engine_rectangle = model.engine_cores();
    auto owners = [&](const std::vector<uint32_t>& family) -> kernel_lib_host::McastSenderConfig {
        if (family.size() > 1) {
            return kernel_lib_host::McastRotatingSenderConfig{.sender_cores = model.column_cores(family)};
        }
        return kernel_lib_host::McastFixedSenderConfig{.sender_index = family[0]};
    };
    uint32_t next_id = kNumOwnSemaphores;
    for (const auto& [name, sender_config] : std::vector<std::pair<std::string, kernel_lib_host::McastSenderConfig>>{
             {"x0", owners(sizing.family0_columns)},
             {"x1", owners(sizing.family1_columns)},
             {"h", kernel_lib_host::McastRotatingSenderConfig{}}}) {
        kernel_lib_host::Mcast mcast(
            *model.mesh,
            kernel_lib_host::McastConfig{.noc = NOC::NOC_0, .handshake = true},
            engine_rectangle,
            sizing.engine_columns,
            sender_config);
        const std::array kernels{std::ref(*engine_kernel)};
        mcast.attach(program_descriptor, name, kernels, next_id);
        next_id = mcast.next_semaphore_id();
    }
}

HeadWorkload build_head_workload(
    const HeadModel& model, const HeadBuffers& buffers, const std::vector<ttnn::MeshCoordinate>& coords) {
    // (the arena: an L1 tensor of the arena cores, a shard of at least arena_bytes each - the CBs over its regions bind
    // to it, the kernels take base + offset; the kernels' args blocks land past its regions, at args_offset)
    const Buffer* arena = buffers[SOURCE_ARENA];
    const uint64_t need = uint64_t(model.args_offset) + kHeadArgsBytes;
    TT_FATAL(
        arena != nullptr && arena->is_l1() && arena->aligned_size_per_bank() >= need,
        "moe_head: the arena must be an L1 tensor of at least {} B a core ({} B of regions, {} B of args blocks)",
        need,
        model.args_offset,
        kHeadArgsBytes);
    TT_FATAL(
        arena->shard_spec().grid().contains(head_arena_cores(model)),
        "moe_head: the arena must span the matmul cores, the router cores, the ports and the watcher");
    // row blocks in one program: a block's buffers - x / y `block` strides further, the block's offsets
    auto block_buffers = [&](uint32_t block) {
        HeadBuffers shifted = buffers;
        shifted.block = block;
        shifted.addresses[SOURCE_X] += block * buffers.x_stride;
        shifted.addresses[SOURCE_Y] += block * buffers.y_stride;
        return shifted;
    };
    const bool merged = buffers.merged && buffers.row_blocks > 1;
    // (row blocks in one program: what moves from one block to the next, per block - row_block.hpp's named args)
    namespace names = moe_head_named_args;
    const uint32_t route_stride = model.tokens_per_chip * 4 * route_words(model.num_chips);
    const std::vector<std::pair<const char*, uint32_t>> block_strides = {
        {names::kRowBlockXStrideBytes, buffers.x_stride},
        {names::kRowBlockYStrideBytes, buffers.y_stride},
        {names::kRowBlockDispatchRowStride, model.tokens_per_chip},
        {names::kRowBlockPaddedRowStride, model.tokens_per_chip * model.mesh_columns},
        {names::kRowBlockRouterPageStride, model.router_row_tiles},
        {names::kRowBlockRouteWordsStride, route_stride}};
    std::set<uint32_t> block_deltas = {
        buffers.x_stride,
        buffers.y_stride,
        model.tokens_per_chip,
        model.tokens_per_chip * model.mesh_columns,
        model.router_row_tiles,
        route_stride};
    block_deltas.erase(0u);
    // each chip's program and where its address args sit
    head_args_detail::Chips chips;
    std::vector<HeadAddressArgs> address_args(coords.size());
    for (size_t i = 0; i < coords.size(); ++i) {
        const auto& coord = coords[i];
        const ChipCoord chip_coord = chip_of(coord, model.topology());  // (the block's chip of this mesh coordinate)
        if (merged) {  // (block 0's program; its kernels block over the row blocks themselves - row_block.hpp)
            const uint32_t row_blocks = buffers.row_blocks;
            const HeadBuffers first_block_buffers = block_buffers(0), second_block_buffers = block_buffers(1),
                              last_block_buffers = block_buffers(row_blocks - 1);  // (ChipProgram
            // keeps a reference)
            ChipProgram first_block_chip(model, first_block_buffers, {chip_coord.r, chip_coord.c});
            auto first_block_program = first_block_chip.build(&address_args[i]);
            // (block 1's and the last block's programs: every arg that differs by the block is one the kernels compute)
            ChipProgram second_block_chip(model, second_block_buffers, {chip_coord.r, chip_coord.c});
            const auto second_block_program = second_block_chip.build(nullptr);
            ProgramDescriptor last_block_program;
            if (row_blocks > 2) {
                ChipProgram last_block_chip(model, last_block_buffers, {chip_coord.r, chip_coord.c});
                last_block_program = last_block_chip.build(nullptr);
            }
            row_block_detail::check_row_block_args(
                first_block_program,
                second_block_program,
                row_blocks > 2 ? last_block_program : second_block_program,
                row_blocks,
                block_deltas);
            chips.emplace_back(coord, std::move(first_block_program));
        } else {
            ChipProgram chip_program(model, buffers, {chip_coord.r, chip_coord.c});
            chips.emplace_back(coord, chip_program.build(&address_args[i]));
        }
    }
    // every kernel's args into the chips' DRAM tables: the programs are then alike
    const ProgramAddresses converted = head_args_to_dram(model, buffers, chips, address_args);
    tt::tt_metal::WorkloadDescriptor workload;
    for (auto& [coord, program] : chips) {
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(program)});
    }
    // one program for the mesh (common/chip_program.hpp: the union of the chips' semaphores - the fabric connections'
    // differ by core - and their eth channels per-device runtime args)
    std::string why;
    auto one = merge_chip_workload(std::move(workload), "moe_head", &why);
    workload = std::move(one.workload);
    if (workload.programs.size() > 1) {
        log_warning(tt::LogOp, "moe_head: one program per chip: {}", why);
    }
    // (row blocks: the barrier's table holds the block words' address, a common arg after the converter's)
    std::optional<uint32_t> row_block_words_source;
    if (merged) {
        const auto specs = head_words(model);
        for (size_t i = 0; i < specs.size(); ++i) {
            if (specs[i].name == "row_blocks") {
                row_block_words_source = u32(SOURCE_FIRST_WORD + i);
            }
        }
        TT_FATAL(
            row_block_words_source && buffers.addresses.at(*row_block_words_source) == buffers.row_block_words,
            "moe_head: row blocks: no block words among the persistent words");
    }
    HeadWorkload out;
    for (auto& program : workload.programs) {
        attach_engine_mcasts(model, program.descriptor);
        // (every kernel's row-block count: its kernel_main's loop - row_block.hpp; the strides and the barrier's args
        // are 0 with one block: add_row_block_barrier sets them)
        for (auto& kernel_descriptor : program.descriptor.kernels) {
            named_arg(kernel_descriptor, names::kRowBlocks, buffers.row_blocks);
            for (const char* name : names::kRowBlockStrides) {
                named_arg(kernel_descriptor, name, 0);
            }
            named_arg(kernel_descriptor, names::kRowBlockBarrierArg, 0);
            named_arg(kernel_descriptor, names::kRowBlockBarrierLead, 0);
            named_arg(kernel_descriptor, names::kRowBlockBarrierCoordinator, 0);
        }
        ProgramAddresses addresses = converted;
        if (merged) {
            row_block_detail::add_row_block_barrier(
                program.descriptor,
                buffers.row_blocks,
                buffers.row_block_words,
                [&](const CoreCoord& logical_core) { return model.mesh->worker_core_from_logical_core(logical_core); },
                block_strides);
            for (uint32_t k = 0; k < program.descriptor.kernels.size(); ++k) {
                for (const auto& [name, value] : program.descriptor.kernels[k].named_compile_time_args) {
                    if (name == names::kRowBlockBarrierArg) {
                        addresses.args.push_back(ProgramAddresses::AddressArg{
                            .kernel = k,
                            .common = true,
                            .core = {},
                            .index = value,
                            .source = *row_block_words_source,
                            .offset = 0});
                    }
                }
            }
        }
        addresses.arena_cbs = arena_backed_cbs(program.descriptor, buffers[SOURCE_ARENA]);
        out.addresses.push_back(std::move(addresses));
    }
    out.workload = std::move(workload);
    out.per_device_runtime_args = std::move(one.per_device_runtime_args);
    return out;
}

}  // namespace ttnn::operations::bringup::moe_block::moe_head
