// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The head's kernel args into per-chip DRAM tables, the mesh workload (build_head_workload), the engine's mcast
// families, and the program-cache hit's address patching (patch_head_program).
#include "head_chip_program.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

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
// a block of `words` args and `num_fixups` fixups, as runtime_args_to_dram_fixup lays it out
uint32_t block_bytes(uint32_t words, uint32_t num_fixups) {
    return u32(align_to_64(4 * (int64_t(words) + 1 + num_fixups)));
}

}  // namespace head_args_detail

// Every kernel's args into per-chip DRAM tables (common/args_table.hpp runtime_args_to_dram_fixup), so the chips'
// programs become alike - one program for the mesh, nothing per chip or per core written on a launch.
//  * A data-movement kernel's block on a core: its runtime args (padded to the kernel's longest), then its common args
//    (from MOE_BLOCK_ARGS_CMN: moe_block_args::common reads them there - they hold chip values too, chip_values), then
//    on the RISCV_0 kernel of a core with a compute kernel the compute kernel's runtime + common args (from
//    MOE_BLOCK_ARGS_PEER_AT): a compute kernel cannot read DRAM, so its data-movement kernel hands them over as one
//    page of CB kArgsCb.
//  * Every address arg (HeadPatches) becomes a fixup: the table holds its offset, the device adds its source's address
//    from the kernel's common args - which a cache hit rewrites (patch_head_program).
//  * The blocks land in the arena's args region (HeadModel.args_offset, kHeadArgsBytes a core, past every other arena
//    region): the RISCV_0 kernels' blocks at its start, the RISCV_1 kernels' after the largest of those, the compute
//    page after the largest RISCV_1 block. The region is checked against kHeadArgsBytes here.
// Fills `layout` (what a cache hit rewrites); returns the tables, or nullptr with layout_only (the programs are then
// only half converted: thrown away).
std::shared_ptr<const void> head_args_to_dram(
    const HeadModel& model,
    const HeadBuffers& buffers,
    head_args_detail::Chips& chips,
    const std::vector<HeadPatches>& patches,
    HeadPatches& layout,
    bool layout_only) {
    using namespace head_args_detail;
    const size_t num_chips = chips.size();
    TT_FATAL(
        num_chips > 0 && patches.size() == num_chips,
        "moe_head: args to DRAM: {} programs, {} patch lists",
        num_chips,
        patches.size());
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
    std::vector<ArgsFixupKernel> fixup_kernels;
    std::vector<int> fixup_kernel_of(num_kernels, -1);
    for (size_t k = 0; k < num_kernels; ++k) {
        if (!is_compute(first_kernels[k])) {
            fixup_kernel_of[k] = int(fixup_kernels.size());
            ArgsFixupKernel fixup_kernel;
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
        for (const auto& patch_arg : patches[chip_index].args) {
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
        bytes[k] = block_bytes(words(k), u32(num_fixups));
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
    layout = HeadPatches{};
    layout.args_offset = u32(model.args_offset);
    for (auto& fixup_kernel : fixup_kernels) {
        fixup_kernel.l1_offset = on_riscv0(first_kernels[fixup_kernel.kernel]) ? 0u : riscv0_bytes;
        fixup_kernel.bytes = on_riscv0(first_kernels[fixup_kernel.kernel]) ? riscv0_bytes : riscv1_bytes;
        layout.dram.push_back({u32(fixup_kernel.kernel), 0u, fixup_kernel.l1_offset});
    }

    // the programs: the lists as each kernel's runtime args, no common args left (the converter adds its own), the
    // defines saying where things sit, the compute page's CB
    for (size_t chip_index = 0; chip_index < num_chips; ++chip_index) {
        auto& program = chips[chip_index].second;
        for (size_t k = 0; k < num_kernels; ++k) {
            auto& kernel_descriptor = program.kernels[k];
            kernel_descriptor.common_runtime_args.clear();
            kernel_descriptor.runtime_args.clear();
            if (is_compute(kernel_descriptor)) {
                if (host_kernel[k] >= 0) {
                    define(kernel_descriptor, "MOE_BLOCK_ARGS_TRISC", "1");
                    define(kernel_descriptor, "MOE_BLOCK_ARGS_CMN", integer_literal(runtime_words[k]));
                    define(kernel_descriptor, "MOE_BLOCK_ARGS_PEER_CB", integer_literal(kArgsCb));
                }
                continue;
            }
            for (auto& [core_key, list] : lists[chip_index][k]) {
                kernel_descriptor.runtime_args.emplace_back(
                    CoreCoord(core_key.first, core_key.second), std::move(list));
            }
            define(kernel_descriptor, "MOE_BLOCK_ARGS_CMN", integer_literal(runtime_words[k]));
            if (is_host[k]) {
                define(
                    kernel_descriptor, "MOE_BLOCK_ARGS_PEER_AT", integer_literal(runtime_words[k] + common_words[k]));
                define(kernel_descriptor, "MOE_BLOCK_ARGS_PEER_N", integer_literal(page_words));
                define(kernel_descriptor, "MOE_BLOCK_ARGS_PEER_CB", integer_literal(kArgsCb));
            }
        }
        if (page_words > 0) {
            CBDescriptor cb;
            cb.total_size = page;
            cb.core_ranges = compute_cores;
            cb.format_descriptors = {
                CBFormatDescriptor{.buffer_index = kArgsCb, .data_format = tt::DataFormat::UInt32, .page_size = page}};
            cb.buffer = buffers[SOURCE_ARENA];
            cb.address_offset = layout.args_offset + riscv0_bytes + riscv1_bytes;
            program.cbs.push_back(cb);
        }
    }
    for (uint32_t i = 0; i < chips[0].second.cbs.size(); ++i) {
        if (chips[0].second.cbs[i].buffer != nullptr) {
            layout.arena_cbs.push_back({i, chips[0].second.cbs[i].address_offset});
        }
    }
    if (layout_only) {
        return nullptr;
    }
    std::string why;
    const std::vector<uint32_t> sources(buffers.addresses.begin(), buffers.addresses.end());
    auto converted = runtime_args_to_dram_fixup(
        *model.mesh, chips, fixup_kernels, sources, buffers.addresses[SOURCE_ARENA] + layout.args_offset, why);
    TT_FATAL(converted.tables != nullptr, "moe_head: the kernels' args to DRAM: {}", why);
    for (const auto& kept_common_args : converted.ca) {
        for (uint32_t kept_count : kept_common_args) {
            TT_FATAL(kept_count == 0, "moe_head: args to DRAM: a kernel kept common args ({})", kept_count);
        }
    }
    return converted.tables;
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
    auto columns_of = [&](const std::vector<uint32_t>& columns) {
        std::vector<CoreRange> ranges;
        for (uint32_t column : columns) {
            ranges.emplace_back(CoreCoord(column, 0), CoreCoord(column, sizing.engine_rows - 1));
        }
        return CoreRangeSet(ranges);
    };
    auto owners = [&](const std::vector<uint32_t>& family) -> kernel_lib_host::McastSenderConfig {
        if (family.size() > 1) {
            return kernel_lib_host::McastRotatingSenderConfig{.sender_cores = columns_of(family)};
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

tt::tt_metal::WorkloadDescriptor build_head_workload(
    const HeadModel& model,
    const HeadBuffers& buffers,
    const std::vector<ttnn::MeshCoordinate>& coords,
    HeadPatches* layout,
    bool layout_only) {
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
    // (row blocks in one program: what moves from one block to the next, per block - row_block.hpp's defines)
    const uint32_t route_stride = model.tokens_per_chip * 4 * route_words(model.num_chips);
    const std::vector<std::pair<std::string, std::string>> block_defines = {
        {"ROW_BLOCK_X_STRIDE_BYTES", std::to_string(buffers.x_stride)},
        {"ROW_BLOCK_Y_STRIDE_BYTES", std::to_string(buffers.y_stride)},
        {"ROW_BLOCK_DISPATCH_ROW_STRIDE", std::to_string(model.tokens_per_chip)},
        {"ROW_BLOCK_PADDED_ROW_STRIDE", std::to_string(model.tokens_per_chip * model.mesh_columns)},
        {"ROW_BLOCK_ROUTER_PAGE_STRIDE", std::to_string(model.router_row_tiles)},
        {"ROW_BLOCK_ROUTE_WORDS_STRIDE", std::to_string(route_stride)}};
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
    std::vector<HeadPatches> patches(coords.size());
    for (size_t i = 0; i < coords.size(); ++i) {
        const auto& coord = coords[i];
        const ChipCoord chip_coord = chip_of(coord, model.topology());  // (the block's chip of this mesh coordinate)
        if (merged) {  // (block 0's program; its kernels block over the row blocks themselves - row_block.hpp)
            const uint32_t row_blocks = buffers.row_blocks;
            const HeadBuffers first_block_buffers = block_buffers(0), second_block_buffers = block_buffers(1),
                              last_block_buffers = block_buffers(row_blocks - 1);  // (ChipProgram
            // keeps a reference)
            ChipProgram first_block_chip(model, first_block_buffers, {chip_coord.r, chip_coord.c});
            auto first_block_program = first_block_chip.build(&patches[i]);
            if (!layout_only) {  // (block 1's and the last block's programs: every arg that differs by the block is one
                                 // the kernels compute)
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
            }
            chips.emplace_back(coord, std::move(first_block_program));
        } else {
            ChipProgram chip_program(model, buffers, {chip_coord.r, chip_coord.c});
            chips.emplace_back(coord, chip_program.build(&patches[i]));
        }
    }
    // every kernel's args into the chips' DRAM tables: the programs are then alike
    HeadPatches patch_layout;
    const auto tables = head_args_to_dram(model, buffers, chips, patches, patch_layout, layout_only);
    if (layout) {
        *layout = patch_layout;
    }
    tt::tt_metal::WorkloadDescriptor workload;
    if (layout_only) {
        return workload;
    }
    for (auto& [coord, program] : chips) {
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(program)});
    }
    // one program for the mesh (common/chip_program.hpp: the union of the chips' semaphores - the fabric connections'
    // differ by core)
    std::string why;
    workload = merge_chip_workload(std::move(workload), "moe_head", &why);
    if (workload.programs.size() > 1) {
        log_warning(tt::LogOp, "moe_head: one program per chip: {}", why);
    }
    for (auto& program : workload.programs) {
        attach_engine_mcasts(model, program.descriptor);
        // (every kernel's row-block count: its kernel_main's loop - row_block.hpp)
        for (auto& kernel_descriptor : program.descriptor.kernels) {
            kernel_descriptor.named_compile_time_args.emplace_back("row_blocks", buffers.row_blocks);
        }
        if (merged) {
            row_block_detail::add_row_block_barrier(
                program.descriptor,
                buffers.row_blocks,
                buffers.row_block_words,
                [&](const CoreCoord& logical_core) { return model.mesh->worker_core_from_logical_core(logical_core); },
                block_defines);
        }
    }
    // (the tables live as long as the cached workload)
    workload.buffers.push_back(
        tt::tt_metal::WorkloadBuffer{.owner = std::const_pointer_cast<void>(tables), .buffer = nullptr});
    return workload;
}

void patch_head_program(
    tt::tt_metal::Program& program, const HeadPatches& patches, const HeadBuffers& buffers, HeadLastAddresses& last) {
    // (a source's args only when its address differs from the one this program last got)
    std::array<bool, kHeadMaxSources> moved{};
    for (size_t source = 0; source < kHeadMaxSources; ++source) {
        moved[source] = last.addresses[source] != buffers.addresses[source];
        last.addresses[source] = buffers.addresses[source];
    }
    // the kernels' args in DRAM: each source's address in their common args, and where the arena's args region is
    for (const auto& dram_kernel : patches.dram) {
        auto& common = tt::tt_metal::GetCommonRuntimeArgs(program, dram_kernel.kernel);
        for (size_t source = 0; source < kHeadMaxSources; ++source) {
            if (moved[source]) {
                common[dram_kernel.converter_args_index + kArgsFixupSources + source] = buffers.addresses[source];
            }
        }
        if (moved[SOURCE_ARENA]) {
            common[dram_kernel.converter_args_index + kArgsFixupL1] =
                buffers.addresses[SOURCE_ARENA] + patches.args_offset + dram_kernel.l1_offset;
        }
    }
    // (address args left as runtime / common args, grouped by source: none once every kernel's args are in DRAM)
    const auto& args = patches.args;
    for (const auto& [group_begin, group_end] : patches.groups) {
        const uint16_t source = args[group_begin].source;
        if (!moved[source]) {
            continue;
        }
        for (size_t k = group_begin; k < group_end; ++k) {
            const auto& patch_arg = args[k];
            auto& data = patch_arg.common ? tt::tt_metal::GetCommonRuntimeArgs(program, patch_arg.kernel)
                                          : tt::tt_metal::GetRuntimeArgs(program, patch_arg.kernel, patch_arg.core);
            data[patch_arg.index] = buffers.addresses[source] + patch_arg.offset;
        }
    }
    if (moved[SOURCE_ARENA]) {
        const auto cbs = program.circular_buffers();
        for (const auto& cb : patches.arena_cbs) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(
                program, cbs[cb.index]->id(), *buffers[SOURCE_ARENA], cb.offset);
        }
    }
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head
