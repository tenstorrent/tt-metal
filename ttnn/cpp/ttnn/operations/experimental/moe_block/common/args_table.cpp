// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args_table.hpp"

#include <algorithm>
#include <map>
#include <tuple>
#include <type_traits>

#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shard_data_transfer.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/experimental/moe_block/common/kernels/args_table_names.hpp"

namespace ttnn::operations::experimental::moe_block {

namespace args_table_detail {

using tt::tt_metal::CoreCoord;

std::vector<CoreCoord> cores_of(const tt::tt_metal::CoreRangeSet& set) {
    std::vector<CoreCoord> out;
    for (const auto& r : set.ranges()) {
        for (auto y = r.start_coord.y; y <= r.end_coord.y; ++y) {
            for (auto x = r.start_coord.x; x <= r.end_coord.x; ++x) {
                out.emplace_back(x, y);
            }
        }
    }
    return out;
}

}  // namespace args_table_detail

namespace {

ArgsTable convert_runtime_args_to_dram(
    tt::tt_metal::distributed::MeshDevice& mesh,
    ChipPrograms& chips,
    const std::vector<ArgsTableKernel>& kernels,
    std::span<const uint32_t> sources,
    uint32_t block_base,
    std::string& why) {
    ZoneScopedN("moe_block: runtime args to DRAM");
    using namespace args_table_detail;
    using tt::tt_metal::distributed::DeviceLocalBufferConfig;
    using tt::tt_metal::distributed::MeshBuffer;
    using tt::tt_metal::distributed::ReplicatedBufferConfig;
    using Key = std::pair<size_t, size_t>;
    TT_FATAL(!chips.empty(), "runtime_args_to_dram: no program");
    TT_FATAL(sources.size() < 0x10000, "runtime_args_to_dram: {} sources", sources.size());
    const size_t N = chips.size();
    const uint32_t nbanks = mesh.allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    constexpr uint32_t kAlign = 64;  // (a DRAM page and its L1 landing: whole 64 B)

    // (checks first: nothing changes unless every kernel converts)
    struct Plan {
        uint32_t gx = 1, gy = 1, fixups_at = 0, nfix = 0, block = 0;
    };
    std::vector<Plan> plans(kernels.size());
    for (size_t j = 0; j < kernels.size(); ++j) {
        const auto& kk = kernels[j];
        auto& pl = plans[j];
        if (kk.fixups.size() != N || kk.kept.size() != N) {
            why = fmt::format("kernel {}: fixups / kept for {} / {} chips, {} programs", kk.kernel, kk.fixups.size(),
                              kk.kept.size(), N);  // fmt: skip
            return {};
        }
        if (kk.l1_offset % kAlign != 0) {
            why = fmt::format("kernel {}: L1 offset {} not 64 B aligned", kk.kernel, kk.l1_offset);
            return {};
        }
        TT_FATAL(
            kk.kernel < chips[0].second.kernels.size(),
            "runtime_args_to_dram: kernel {} of {}",
            kk.kernel,
            chips[0].second.kernels.size());
        const auto& k0 = chips[0].second.kernels[kk.kernel];
        for (size_t c = 0; c < N; ++c) {
            const auto& k = chips[c].second.kernels[kk.kernel];
            // (the chips may place the kernel on different cores - e.g. the combine's port kernels on each chip's
            // own port cores: each chip's table holds the blocks of its own cores)
            if (!k.buffer_bindings.empty() || !k.common_buffer_bindings.empty()) {
                why = fmt::format(
                    "kernel {} ({}): it has buffer bindings (fixups_from_buffer_bindings turns them into fixups)",
                    kk.kernel,
                    k0.kernel_source);
                return {};
            }
            for (const auto& core : cores_of(k.core_ranges)) {
                pl.gx = std::max<uint32_t>(pl.gx, core.x + 1);
                pl.gy = std::max<uint32_t>(pl.gy, core.y + 1);
            }
            for (const auto& [core, a] : k.runtime_args) {
                pl.fixups_at = std::max<uint32_t>(pl.fixups_at, static_cast<uint32_t>(a.size()));
                const Key key{core.x, core.y};
                if (const auto it = kk.fixups[c].find(key); it != kk.fixups[c].end()) {
                    pl.nfix = std::max<uint32_t>(pl.nfix, static_cast<uint32_t>(it->second.size()));
                    for (const auto& f : it->second) {
                        if (f.index >= a.size() || f.source >= sources.size() || f.index > 0xFFFF) {
                            why = fmt::format(
                                "kernel {} core {}: fixup ({}, {}) out of range",
                                kk.kernel,
                                core.str(),
                                f.index,
                                f.source);  // fmt: skip
                            return {};
                        }
                    }
                }
                if (const auto it = kk.kept[c].find(key); it != kk.kept[c].end()) {
                    if (it->second.first + it->second.second > a.size()) {
                        why = fmt::format(
                            "kernel {} core {}: kept args past its {} args", kk.kernel, core.str(), a.size());
                        return {};
                    }
                }
            }
        }
        pl.block = args_block_bytes(pl.fixups_at, pl.nfix);
        if (pl.block > kk.bytes) {
            why = fmt::format(
                "kernel {} ({}): a block of {} args + {} fixups ({} B) does not fit its {} B in L1",
                kk.kernel,
                k0.kernel_source,
                pl.fixups_at,
                pl.nfix,
                pl.block,
                kk.bytes);
            return {};
        }
    }

    for (size_t j = 0; j < kernels.size(); ++j) {
        log_debug(
            tt::LogOp,
            "args in DRAM: kernel {} ({}): {} B a core of its {} B",
            kernels[j].kernel,
            chips[0].second.kernels[kernels[j].kernel].kernel_source,
            plans[j].block,
            kernels[j].bytes);
    }
    ArgsTable out;
    auto tables = std::make_shared<std::vector<std::shared_ptr<MeshBuffer>>>();
    for (size_t j = 0; j < kernels.size(); ++j) {
        const auto& kk = kernels[j];
        const auto& pl = plans[j];
        const size_t pages = static_cast<size_t>(pl.gx) * pl.gy, words = pl.block / 4;
        auto table = MeshBuffer::create(
            ReplicatedBufferConfig{.size = pages * pl.block},
            DeviceLocalBufferConfig{.page_size = pl.block, .buffer_type = tt::tt_metal::BufferType::DRAM},
            &mesh);
        std::vector<std::vector<uint32_t>> data(N, std::vector<uint32_t>(pages * words, 0u));
        for (size_t c = 0; c < N; ++c) {
            for (const auto& [core, a] : chips[c].second.kernels[kk.kernel].runtime_args) {
                auto* blk = data[c].data() + (static_cast<size_t>(core.y) * pl.gx + core.x) * words;
                std::copy(a.begin(), a.end(), blk);
                const auto it = kk.fixups[c].find(Key{core.x, core.y});
                const uint32_t n = it == kk.fixups[c].end() ? 0u : static_cast<uint32_t>(it->second.size());
                blk[pl.fixups_at] = n;
                for (uint32_t f = 0; f < n; ++f) {
                    const auto& fix = it->second[f];
                    blk[fix.index] -= sources[fix.source];  // (the arg as built: the source's address + its offset)
                    blk[pl.fixups_at + 1 + f] = fix.index | (fix.source << 16);
                }
            }
        }
        std::vector<tt::tt_metal::distributed::ShardDataTransfer> writes;
        writes.reserve(N);
        for (size_t c = 0; c < N; ++c) {
            writes.push_back(tt::tt_metal::distributed::ShardDataTransfer{chips[c].first}.host_data(data[c].data()));
        }
        mesh.mesh_command_queue(0).enqueue_write_shards(table, writes, /*blocking=*/true);

        // (a block in a CB: the kernel reads the CB's address, the common arg stays 0 - the same layout either way)
        const uint32_t block_address = kk.block_cb ? 0u : block_base + kk.l1_offset;
        std::vector<uint32_t> common_index;
        for (size_t c = 0; c < N; ++c) {
            auto& k = chips[c].second.kernels[kk.kernel];
            const uint32_t ca = static_cast<uint32_t>(k.common_runtime_args.size());
            common_index.push_back(ca);
            for (uint32_t x :
                 {static_cast<uint32_t>(table->address()),
                  pl.block,
                  nbanks,
                  pl.gx,
                  block_address,
                  static_cast<uint32_t>(sources.size())}) {  // fmt: skip
                k.common_runtime_args.push_back(x);
            }
            k.common_runtime_args.insert(k.common_runtime_args.end(), sources.begin(), sources.end());
            // (the kept args stay runtime args, from index 0; a core with none kept has no runtime args)
            decltype(k.runtime_args) kept;
            for (const auto& [core, a] : k.runtime_args) {
                if (const auto it = kk.kept[c].find(Key{core.x, core.y}); it != kk.kept[c].end() && it->second.second) {
                    const auto [start, count] = it->second;
                    using Args = std::remove_cvref_t<decltype(a)>;
                    kept.emplace_back(core, Args(a.begin() + start, a.begin() + start + count));
                }
            }
            k.runtime_args = std::move(kept);
            namespace names = moe_block_args_names;
            set_named_compile_time_arg(k, names::kEnabled, 1);
            set_named_compile_time_arg(k, names::kCommonIndex, ca);
            set_named_compile_time_arg(k, names::kFixupsAt, pl.fixups_at);
            set_named_compile_time_arg(k, names::kBlockInCb, kk.block_cb ? 1u : 0u);
            set_named_compile_time_arg(k, names::kBlockCb, kk.block_cb ? *kk.block_cb : 0u);
            set_named_compile_time_arg(k, names::kBlockCbOffset, kk.block_cb ? kk.l1_offset : 0u);
            fill_args_table_named_args(k);
        }
        out.common_index.push_back(std::move(common_index));
        tables->push_back(std::move(table));
    }
    out.tables = tables;
    return out;
}

}  // namespace

ArgsTable runtime_args_to_dram(
    tt::tt_metal::distributed::MeshDevice& mesh,
    ChipPrograms& chips,
    const std::vector<ArgsTableKernel>& kernels,
    std::span<const uint32_t> sources,
    uint32_t block_base,
    std::string& why) {
    ArgsTable out = convert_runtime_args_to_dram(mesh, chips, kernels, sources, block_base, why);
    if (out.tables == nullptr) {
        // (not converted: the kernels still read the table's named compile-time args, so they get them off)
        for (auto& chip : chips) {
            for (const auto& kernel : kernels) {
                if (kernel.kernel < chip.second.kernels.size()) {
                    fill_args_table_named_args(chip.second.kernels[kernel.kernel]);
                }
            }
        }
    }
    return out;
}

void set_named_compile_time_arg(tt::tt_metal::KernelDescriptor& kernel, const std::string& name, uint32_t value) {
    for (auto& [existing, existing_value] : kernel.named_compile_time_args) {
        if (existing == name) {
            existing_value = value;
            return;
        }
    }
    kernel.named_compile_time_args.emplace_back(name, value);
}

void fill_args_table_named_args(tt::tt_metal::KernelDescriptor& kernel) {
    for (const char* name : moe_block_args_names::kAll) {
        const bool has = std::any_of(
            kernel.named_compile_time_args.begin(), kernel.named_compile_time_args.end(), [&](const auto& arg) {
                return arg.first == name;
            });
        if (!has) {
            kernel.named_compile_time_args.emplace_back(name, 0u);
        }
    }
}

void set_args_table_common_in_block(tt::tt_metal::KernelDescriptor& kernel, uint32_t at) {
    set_named_compile_time_arg(kernel, moe_block_args_names::kCommonInBlock, 1);
    set_named_compile_time_arg(kernel, moe_block_args_names::kCommonAt, at);
}

void set_args_table_peer_page_sender(tt::tt_metal::KernelDescriptor& kernel, uint32_t cb, uint32_t at, uint32_t words) {
    set_named_compile_time_arg(kernel, moe_block_args_names::kPeerPageSender, 1);
    set_named_compile_time_arg(kernel, moe_block_args_names::kPeerAt, at);
    set_named_compile_time_arg(kernel, moe_block_args_names::kPeerWords, words);
    set_named_compile_time_arg(kernel, moe_block_args_names::kPeerCb, cb);
}

void set_args_table_compute_reads_page(tt::tt_metal::KernelDescriptor& kernel, uint32_t cb, uint32_t common_at) {
    set_named_compile_time_arg(kernel, moe_block_args_names::kComputeReadsPage, 1);
    set_named_compile_time_arg(kernel, moe_block_args_names::kPeerCb, cb);
    set_named_compile_time_arg(kernel, moe_block_args_names::kCommonAt, common_at);
}

uint32_t args_block_bytes(uint32_t words, uint32_t fixups) {
    constexpr uint64_t kAlign = 64;
    const uint64_t bytes = 4 * (uint64_t(words) + 1 + fixups);
    return static_cast<uint32_t>((bytes + kAlign - 1) / kAlign * kAlign);
}

void fixups_from_buffer_bindings(
    ChipPrograms& chips, ArgsTableKernel& kernel, std::span<tt::tt_metal::Buffer* const> sources) {
    kernel.fixups.resize(chips.size());
    kernel.kept.resize(chips.size());
    for (size_t c = 0; c < chips.size(); ++c) {
        TT_FATAL(
            kernel.kernel < chips[c].second.kernels.size(),
            "fixups_from_buffer_bindings: kernel {} of {}",
            kernel.kernel,
            chips[c].second.kernels.size());
        auto& k = chips[c].second.kernels[kernel.kernel];
        TT_FATAL(
            k.common_buffer_bindings.empty(),
            "fixups_from_buffer_bindings: kernel {} ({}) has common buffer bindings",
            kernel.kernel,
            k.kernel_source);
        for (const auto& b : k.buffer_bindings) {
            std::optional<uint32_t> source;
            for (uint32_t s = 0; s < sources.size(); ++s) {
                if (sources[s] != b.buffer) {
                    continue;
                }
                // (one buffer in two slots: a cache hit could not tell which one the arg follows)
                TT_FATAL(
                    !source.has_value(),
                    "kernel {} ({}): one buffer is sources {} and {} - call the op with distinct buffers",
                    kernel.kernel,
                    k.kernel_source,
                    source.value_or(0),
                    s);
                source = s;
            }
            TT_FATAL(
                source.has_value(),
                "kernel {} ({}): arg {} of core {} is bound to a buffer that is no source",
                kernel.kernel,
                k.kernel_source,
                b.arg_idx,
                b.core.str());
            kernel.fixups[c][{b.core.x, b.core.y}].push_back(ArgsTableFixup{.index = b.arg_idx, .source = *source});
        }
        k.buffer_bindings.clear();
    }
}

std::vector<ProgramAddresses::TableKernel> table_kernels(
    const ArgsTable& table,
    const std::vector<ArgsTableKernel>& kernels,
    size_t chip,
    std::optional<uint32_t> block_source,
    uint32_t block_base_offset) {
    TT_FATAL(
        table.common_index.size() == kernels.size(),
        "table_kernels: {} kernels converted, {} given",
        table.common_index.size(),
        kernels.size());
    std::vector<ProgramAddresses::TableKernel> out;
    for (size_t j = 0; j < kernels.size(); ++j) {
        TT_FATAL(
            chip < table.common_index[j].size(), "table_kernels: chip {} of {}", chip, table.common_index[j].size());
        const bool in_cb = kernels[j].block_cb.has_value();
        out.push_back(ProgramAddresses::TableKernel{
            .kernel = static_cast<tt::tt_metal::KernelHandle>(kernels[j].kernel),
            .common_index = table.common_index[j][chip],
            .block_source = in_cb ? std::nullopt : block_source,
            .block_offset = in_cb ? 0u : block_base_offset + kernels[j].l1_offset});
    }
    return out;
}

std::vector<ProgramAddresses::ArenaCb> arena_backed_cbs(
    const tt::tt_metal::ProgramDescriptor& desc, const tt::tt_metal::Buffer* arena) {
    for (const auto& k : desc.kernels) {
        TT_FATAL(
            k.buffer_bindings.empty() && k.common_buffer_bindings.empty(),
            "moe_block: kernel {} has buffer bindings no cache hit rewrites",
            k.kernel_source);
    }
    std::vector<ProgramAddresses::ArenaCb> out;
    for (uint32_t i = 0; i < desc.cbs.size(); ++i) {
        const auto& cb = desc.cbs[i];
        TT_FATAL(cb.tensor == nullptr, "moe_block: CB {} is backed by a tensor no cache hit re-points", i);
        if (cb.buffer != nullptr) {
            TT_FATAL(cb.buffer == arena, "moe_block: CB {} is backed by a buffer that is not the arena", i);
            out.push_back(ProgramAddresses::ArenaCb{.index = i, .offset = cb.address_offset});
        }
    }
    return out;
}

void write_program_addresses(
    tt::tt_metal::Program& program,
    const ProgramAddresses& addresses,
    std::span<const uint32_t> sources,
    const tt::tt_metal::Buffer* arena) {
    ZoneScopedN("moe_block: write program addresses");
    for (const auto& k : addresses.table_kernels) {
        auto& common = tt::tt_metal::GetCommonRuntimeArgs(program, k.kernel);
        TT_FATAL(
            common[k.common_index + kArgsTableSources - 1] == sources.size(),
            "moe_block: kernel {} was built with {} sources, the hit has {}",
            k.kernel,
            common[k.common_index + kArgsTableSources - 1],
            sources.size());
        for (size_t s = 0; s < sources.size(); ++s) {
            common[k.common_index + kArgsTableSources + s] = sources[s];
        }
        if (k.block_source) {
            common[k.common_index + kArgsTableBlockAddress] = sources[*k.block_source] + k.block_offset;
        }
    }
    for (const auto& a : addresses.args) {
        auto& data = a.common ? tt::tt_metal::GetCommonRuntimeArgs(program, a.kernel)
                              : tt::tt_metal::GetRuntimeArgs(program, a.kernel, a.core);
        data[a.index] = sources[a.source] + a.offset;
    }
    if (addresses.arena_cbs.empty()) {
        return;
    }
    TT_FATAL(arena != nullptr, "moe_block: the program has CBs on the arena, the call has no arena");
    const auto cbs = program.circular_buffers();
    for (const auto& cb : addresses.arena_cbs) {
        tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cbs[cb.index]->id(), *arena, cb.offset);
    }
}

}  // namespace ttnn::operations::experimental::moe_block
