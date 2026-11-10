// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row blocks: the per-block argument check and the barrier tables (kernels/row_block.hpp).
#include "head_chip_program.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

namespace row_block_detail {

using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::DataMovementProcessor;
using tt::tt_metal::ReaderConfigDescriptor;

// a kernel's RISCs (row_block.hpp's done words: bit 0 BRISC, 1 NCRISC, 2 .. 4 TRISC 0 .. 2)
uint32_t risc_bits(const KernelDescriptor& kernel_descriptor) {
    return std::visit(
        [](const auto& config) -> uint32_t {
            using ConfigType = std::decay_t<decltype(config)>;
            if constexpr (std::is_same_v<ConfigType, ComputeConfigDescriptor>) {
                return 0x1C;
            } else if constexpr (std::is_same_v<ConfigType, DataMovementConfigDescriptor>) {
                return config.processor == DataMovementProcessor::RISCV_0 ? 1u : 2u;
            } else if constexpr (std::is_same_v<ConfigType, ReaderConfigDescriptor>) {
                return 2u;
            } else {
                return 1u;
            }
        },
        kernel_descriptor.config);
}

uint32_t key_of(const CoreCoord& core) { return uint32_t(core.x) | (uint32_t(core.y) << 8); }

std::vector<CoreCoord> cores_in(const CoreRangeSet& core_ranges) {
    std::vector<CoreCoord> cores;
    for (const auto& range : core_ranges.ranges()) {
        for (auto y = range.start_coord.y; y <= range.end_coord.y; ++y) {
            for (auto x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                cores.emplace_back(x, y);
            }
        }
    }
    return cores;
}

// The kernels compute every argument that differs from one row block to the next (row_block.hpp: base + l x one of
// the deltas). Proven on a cache miss from the blocks' own programs: each argument of block 1's and the last block's
// program is block 0's plus a multiple of the block, by one of the declared deltas - an argument that changes otherwise
// (one no kernel computes) stops here instead of running stale.
void check_row_block_args(
    const ProgramDescriptor& first_block,
    const ProgramDescriptor& second_block,
    const ProgramDescriptor& last_block,
    uint32_t row_blocks,
    const std::set<uint32_t>& deltas) {
    TT_FATAL(
        second_block.kernels.size() == first_block.kernels.size() &&
            last_block.kernels.size() == first_block.kernels.size(),
        "moe_head: row blocks: the blocks' programs differ in their kernels");
    auto same = [&](const std::vector<uint32_t>& first_args,
                    const std::vector<uint32_t>& second_args,
                    const std::vector<uint32_t>& last_args,
                    const std::string& what) {
        TT_FATAL(
            second_args.size() == first_args.size() && last_args.size() == first_args.size(),
            "moe_head: row blocks: {}: the blocks' arg counts differ",
            what);
        for (size_t i = 0; i < first_args.size(); ++i) {
            const uint32_t delta = second_args[i] - first_args[i];
            TT_FATAL(
                last_args[i] == first_args[i] + (row_blocks - 1) * delta && (delta == 0 || deltas.contains(delta)),
                "moe_head: row blocks: {} arg {} differs by the block ({} / {} / {}) in a way no kernel computes",
                what,
                i,
                first_args[i],
                second_args[i],
                last_args[i]);
        }
    };
    for (size_t i = 0; i < first_block.kernels.size(); ++i) {
        const auto &first_kernel = first_block.kernels[i], &second_kernel = second_block.kernels[i],
                   &last_kernel = last_block.kernels[i];
        TT_FATAL(
            second_kernel.kernel_source == first_kernel.kernel_source &&
                last_kernel.kernel_source == first_kernel.kernel_source &&
                second_kernel.defines == first_kernel.defines && last_kernel.defines == first_kernel.defines &&
                second_kernel.compile_time_args == first_kernel.compile_time_args &&
                last_kernel.compile_time_args == first_kernel.compile_time_args &&
                second_kernel.core_ranges == first_kernel.core_ranges &&
                last_kernel.core_ranges == first_kernel.core_ranges &&
                second_kernel.runtime_args.size() == first_kernel.runtime_args.size() &&
                last_kernel.runtime_args.size() == first_kernel.runtime_args.size(),
            "moe_head: row blocks: kernel {} ({}) differs between the row_blocks but for its args",
            i,
            first_kernel.kernel_source);
        same(
            first_kernel.common_runtime_args,
            second_kernel.common_runtime_args,
            last_kernel.common_runtime_args,
            first_kernel.kernel_source + " common");
        for (size_t j = 0; j < first_kernel.runtime_args.size(); ++j) {
            TT_FATAL(
                second_kernel.runtime_args[j].first == first_kernel.runtime_args[j].first &&
                    last_kernel.runtime_args[j].first == first_kernel.runtime_args[j].first,
                "moe_head: row blocks: kernel {}: the blocks' cores differ",
                first_kernel.kernel_source);
            same(
                first_kernel.runtime_args[j].second,
                second_kernel.runtime_args[j].second,
                last_kernel.runtime_args[j].second,
                first_kernel.kernel_source + " core " + first_kernel.runtime_args[j].first.str());
        }
    }
}

// The row blocks' defines and barrier tables (kernels/row_block.hpp) on every kernel of one
// chip's program: the block count and deltas; each kernel's table after its own common args (ROW_BLOCK_BARRIER_ARG); a
// lead kernel's roles (ROW_BLOCK_BARRIER_LEAD: the cores whose barrier it leads - their RISCs and program semaphores);
// the coordinator's release list (ROW_BLOCK_BARRIER_COORDINATOR).
void add_row_block_barrier(
    ProgramDescriptor& program,
    uint32_t row_blocks,
    uint32_t words,
    const std::function<CoreCoord(const CoreCoord&)>& virtual_core_of,
    const std::vector<std::pair<std::string, std::string>>& defines) {
    TT_FATAL(row_blocks >= 2 && words != 0, "moe_head: row blocks: {} blocks, block words at {}", row_blocks, words);
    std::map<uint32_t, uint32_t> riscs, semaphores;
    std::map<uint32_t, CoreCoord> cores;
    for (const auto& kernel_descriptor : program.kernels) {
        for (const auto& core : cores_in(kernel_descriptor.core_ranges)) {
            const uint32_t bits = risc_bits(kernel_descriptor);
            TT_FATAL(
                !(riscs[key_of(core)] & bits), "moe_head: row blocks: two kernels on one RISC of core {}", core.str());
            riscs[key_of(core)] |= bits;
            cores[key_of(core)] = core;
        }
    }
    for (const auto& semaphore : program.semaphores) {
        TT_FATAL(
            semaphore.initial_value == 0,
            "moe_head: row blocks: semaphore {} starts at {} (the barrier resets them to 0)",
            semaphore.id,
            semaphore.initial_value);
        for (const auto& core : cores_in(semaphore.core_ranges)) {
            semaphores[key_of(core)] |= 1u << semaphore.id;
        }
    }
    auto lead_of = [&](uint32_t core_key) {
        const uint32_t risc_mask = riscs.at(core_key);
        TT_FATAL(risc_mask & 3, "moe_head: row blocks: core {} runs no data movement kernel", cores.at(core_key).str());
        return (risc_mask & 2) ? 2u : 1u;
    };
    // (the coordinator: the first core of the lead kernel with the fewest cores)
    int coordinator_kernel = -1;
    CoreCoord coordinator;
    size_t fewest = SIZE_MAX;
    for (size_t i = 0; i < program.kernels.size(); ++i) {
        const auto& kernel_descriptor = program.kernels[i];
        const auto kernel_cores = cores_in(kernel_descriptor.core_ranges);
        if (kernel_cores.empty() || !(risc_bits(kernel_descriptor) & 3)) {
            continue;
        }
        bool lead = true;
        for (const auto& core : kernel_cores) {
            lead = lead && lead_of(key_of(core)) == risc_bits(kernel_descriptor);
        }
        if (lead && kernel_cores.size() < fewest) {
            fewest = kernel_cores.size();
            coordinator_kernel = int(i);
            coordinator = kernel_cores[0];
        }
    }
    TT_FATAL(coordinator_kernel >= 0, "moe_head: row blocks: no kernel leads every core it runs on");
    const CoreCoord coordinator_virtual = virtual_core_of(coordinator);
    const uint32_t coordinator_word =
        key_of(coordinator) | (uint32_t(coordinator_virtual.x) << 16) | (uint32_t(coordinator_virtual.y) << 24);
    for (size_t i = 0; i < program.kernels.size(); ++i) {
        auto& kernel_descriptor = program.kernels[i];
        const uint32_t kernel_bits = risc_bits(kernel_descriptor);
        const bool compute = kernel_bits & 0x1C;
        std::vector<uint32_t> table = {words, coordinator_word, uint32_t(cores.size())};
        std::vector<uint32_t> roles;
        if (!compute) {
            const auto kernel_cores = cores_in(kernel_descriptor.core_ranges);
            for (const auto& core : kernel_cores) {
                const uint32_t core_key = key_of(core);
                if (lead_of(core_key) == kernel_bits) {
                    roles.insert(
                        roles.end(),
                        {core_key, riscs.at(core_key), semaphores.contains(core_key) ? semaphores.at(core_key) : 0u});
                }
            }
            bool alike = !roles.empty() && roles.size() / 3 == kernel_cores.size();
            for (size_t j = 3; alike && j < roles.size(); j += 3) {
                alike = roles[j + 1] == roles[1] && roles[j + 2] == roles[2];
            }
            if (alike) {
                table.insert(table.end(), {0x80000000u, roles[1], roles[2]});
            } else if (!roles.empty()) {
                table.push_back(uint32_t(roles.size() / 3));
                table.insert(table.end(), roles.begin(), roles.end());
            }
            if (int(i) == coordinator_kernel) {  // (the cores it releases: every core of the program)
                table.push_back(uint32_t(cores.size()));
                for (const auto& [core_key, core] : cores) {
                    const CoreCoord release_core = virtual_core_of(core);
                    table.push_back(uint32_t(release_core.x) | (uint32_t(release_core.y) << 16));
                }
            }
        }
        const uint32_t table_offset = uint32_t(kernel_descriptor.common_runtime_args.size());
        kernel_descriptor.common_runtime_args.insert(
            kernel_descriptor.common_runtime_args.end(), table.begin(), table.end());
        kernel_descriptor.defines.emplace_back("ROW_BLOCK_BARRIER_ARG", std::to_string(table_offset));
        for (const auto& block_define : defines) {
            kernel_descriptor.defines.push_back(block_define);
        }
        if (!roles.empty()) {
            kernel_descriptor.defines.emplace_back("ROW_BLOCK_BARRIER_LEAD", "1");
        }
        if (int(i) == coordinator_kernel) {
            kernel_descriptor.defines.emplace_back("ROW_BLOCK_BARRIER_COORDINATOR", "1");
        }
    }
}

}  // namespace row_block_detail

}  // namespace ttnn::operations::experimental::moe_block::moe_head
