// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args_table.hpp"

#include <algorithm>
#include <type_traits>
#include <map>
#include <tuple>

#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shard_data_transfer.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block {

namespace args_table_detail {

using tt::tt_metal::Buffer;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::KernelDescriptor;

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

// A kernel's buffer-bound args on one core: (arg index, buffer), by index.
std::vector<std::pair<uint32_t, Buffer*>> bound_on(const KernelDescriptor& k, const CoreCoord& core) {
    std::vector<std::pair<uint32_t, Buffer*>> out;
    for (const auto& b : k.buffer_bindings) {
        if (b.core == core) {
            out.emplace_back(b.arg_idx, b.buffer);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace args_table_detail

std::shared_ptr<const void> runtime_args_to_dram(
    tt::tt_metal::distributed::MeshDevice& mesh,
    std::vector<std::pair<tt::tt_metal::distributed::MeshCoordinate, tt::tt_metal::ProgramDescriptor>>& chips,
    const std::vector<std::optional<ArgsScratch>>& scratch,
    std::string& why) {
    ZoneScopedN("moe_block: runtime args to DRAM");
    using namespace args_table_detail;
    using tt::tt_metal::distributed::DeviceLocalBufferConfig;
    using tt::tt_metal::distributed::MeshBuffer;
    using tt::tt_metal::distributed::ReplicatedBufferConfig;
    TT_FATAL(!chips.empty(), "runtime_args_to_dram: no program");
    const size_t N = chips.size(), K = chips[0].second.kernels.size();
    TT_FATAL(scratch.size() <= K, "runtime_args_to_dram: {} scratch entries for {} kernels", scratch.size(), K);
    for (const auto& [coord, p] : chips) {
        TT_FATAL(p.kernels.size() == K, "runtime_args_to_dram: the chips' programs have different kernels");
    }
    const uint32_t nbanks = mesh.allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    constexpr uint32_t kAlign = 64;  // (a DRAM page and its L1 landing: whole 64 B)

    // (checks first: nothing changes unless every kernel converts)
    struct Plan {
        size_t i;
        ArgsScratch s;
        uint32_t gx, gy, words, block;
        std::vector<std::pair<uint32_t, Buffer*>> bound;
    };
    std::vector<Plan> plans;
    for (size_t i = 0; i < scratch.size(); ++i) {
        if (!scratch[i]) {
            continue;
        }
        const auto& k0 = chips[0].second.kernels[i];
        Plan pl{.i = i, .s = *scratch[i], .gx = 1, .gy = 1, .words = 0, .block = 0, .bound = {}};
        TT_FATAL(
            pl.s.offset % kAlign == 0, "runtime_args_to_dram: kernel {}: scratch offset {} not 64 B aligned", i,
            pl.s.offset);
        bool first = true;
        for (size_t c = 0; c < N; ++c) {
            const auto& k = chips[c].second.kernels[i];
            if (k.core_ranges != k0.core_ranges || k.common_runtime_args != k0.common_runtime_args ||
                !k.common_buffer_bindings.empty()) {
                why = fmt::format(
                    "kernel {} ({}): the chips differ in cores or common args, or it has common buffer bindings",
                    i,
                    k0.kernel_source);
                return nullptr;
            }
            for (const auto& core : cores_of(k.core_ranges)) {
                pl.gx = std::max<uint32_t>(pl.gx, core.x + 1);
                pl.gy = std::max<uint32_t>(pl.gy, core.y + 1);
            }
            for (const auto& [core, a] : k.runtime_args) {
                pl.words = std::max<uint32_t>(pl.words, std::min<uint32_t>(static_cast<uint32_t>(a.size()), pl.s.keep_from));
                auto b = bound_on(k, core);
                if (first) {
                    pl.bound = std::move(b);
                    first = false;
                } else if (b != pl.bound) {
                    why = fmt::format(
                        "kernel {} ({}): buffer-bound args differ between cores or chips", i, k0.kernel_source);
                    return nullptr;
                }
                if (std::any_of(
                        pl.bound.begin(), pl.bound.end(), [&](const auto& ib) { return ib.first >= pl.s.keep_from; })) {
                    why = fmt::format(
                        "kernel {} ({}): a buffer-bound arg at or past the kept runtime args ({})",
                        i,
                        k0.kernel_source,
                        pl.s.keep_from);
                    return nullptr;
                }
            }
            for (const auto& b : k.buffer_bindings) {  // (a bound arg on a core without runtime args: not ours)
                const bool has = std::any_of(
                    k.runtime_args.begin(), k.runtime_args.end(), [&](const auto& ra) { return ra.first == b.core; });
                if (!has) {
                    why = fmt::format("kernel {} ({}): a binding on a core without args", i, k0.kernel_source);
                    return nullptr;
                }
            }
        }
        pl.block = std::max<uint32_t>(kAlign, (4 * pl.words + kAlign - 1) / kAlign * kAlign);
        if (pl.block > pl.s.bytes) {
            why = fmt::format(
                "kernel {} ({}): {} args a core ({} B) do not fit its {} B args scratch",
                i,
                k0.kernel_source,
                pl.words,
                pl.block,
                pl.s.bytes);
            return nullptr;
        }
        plans.push_back(std::move(pl));
    }

    auto tables = std::make_shared<std::vector<std::shared_ptr<MeshBuffer>>>();
    for (const auto& pl : plans) {
        const size_t pages = static_cast<size_t>(pl.gx) * pl.gy, words = pl.block / 4;
        auto table = MeshBuffer::create(
            ReplicatedBufferConfig{.size = pages * pl.block},
            DeviceLocalBufferConfig{.page_size = pl.block, .buffer_type = tt::tt_metal::BufferType::DRAM},
            &mesh);
        // (each chip's blocks: every core's runtime args as they were, the bound ones too - the kernel overwrites
        // those with this call's addresses)
        std::vector<std::vector<uint32_t>> data(N, std::vector<uint32_t>(pages * words, 0u));
        for (size_t c = 0; c < N; ++c) {
            for (const auto& [core, a] : chips[c].second.kernels[pl.i].runtime_args) {
                const size_t n = std::min<size_t>(a.size(), pl.s.keep_from);
                std::copy(a.begin(), a.begin() + n, data[c].begin() + (static_cast<size_t>(core.y) * pl.gx + core.x) * words);
            }
        }
        std::vector<tt::tt_metal::distributed::ShardDataTransfer> writes;
        for (size_t c = 0; c < N; ++c) {
            writes.push_back(tt::tt_metal::distributed::ShardDataTransfer{chips[c].first}.host_data(data[c].data()));
        }
        mesh.mesh_command_queue().enqueue_write_shards(table, writes, /*blocking=*/true);

        for (size_t c = 0; c < N; ++c) {
            auto& k = chips[c].second.kernels[pl.i];
            const uint32_t ca = static_cast<uint32_t>(k.common_runtime_args.size());
            const uint32_t nb = static_cast<uint32_t>(pl.bound.size());
            for (uint32_t x : {static_cast<uint32_t>(table->address()), pl.block, nbanks, pl.gx, nb}) {
                k.common_runtime_args.push_back(x);
            }
            for (const auto& [idx, buf] : pl.bound) {
                k.common_runtime_args.push_back(idx);
                k.common_buffer_bindings.push_back(tt::tt_metal::CommonBufferBinding{
                    .arg_idx = static_cast<uint32_t>(k.common_runtime_args.size()), .buffer = buf});
                k.common_runtime_args.push_back(static_cast<uint32_t>(buf->address()));
            }
            // (the kept args stay runtime args, from index 0; a core with none kept has no runtime args)
            decltype(k.runtime_args) kept;
            for (const auto& [core, a] : k.runtime_args) {
                if (a.size() > pl.s.keep_from) {
                    kept.emplace_back(core, std::remove_cvref_t<decltype(a)>(a.begin() + pl.s.keep_from, a.end()));
                }
            }
            k.runtime_args = std::move(kept);
            k.buffer_bindings.clear();
            k.defines.emplace_back("MOE_BLOCK_ARGS", "1");
            k.defines.emplace_back("MOE_BLOCK_ARGS_CA", std::to_string(ca));
            k.defines.emplace_back("MOE_BLOCK_ARGS_CB", std::to_string(pl.s.cb));
            k.defines.emplace_back("MOE_BLOCK_ARGS_OFF", std::to_string(pl.s.offset));
        }
        tables->push_back(std::move(table));
    }
    return tables;
}

ArgsFixupResult runtime_args_to_dram_fixup(
    tt::tt_metal::distributed::MeshDevice& mesh,
    std::vector<std::pair<tt::tt_metal::distributed::MeshCoordinate, tt::tt_metal::ProgramDescriptor>>& chips,
    const std::vector<ArgsFixupKernel>& kernels,
    const std::vector<uint32_t>& sources,
    uint32_t scratch_base,
    std::string& why) {
    ZoneScopedN("moe_block: runtime args to DRAM (fixups)");
    using namespace args_table_detail;
    using tt::tt_metal::distributed::DeviceLocalBufferConfig;
    using tt::tt_metal::distributed::MeshBuffer;
    using tt::tt_metal::distributed::ReplicatedBufferConfig;
    using Key = std::pair<size_t, size_t>;
    TT_FATAL(!chips.empty(), "runtime_args_to_dram_fixup: no program");
    TT_FATAL(sources.size() < 0x10000, "runtime_args_to_dram_fixup: {} sources", sources.size());
    const size_t N = chips.size();
    const uint32_t nbanks = mesh.allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    constexpr uint32_t kAlign = 64;

    // (checks first: nothing changes unless every kernel converts)
    struct Plan {
        uint32_t gx = 1, gy = 1, fix_at = 0, nfix = 0, block = 0;
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
            "runtime_args_to_dram_fixup: kernel {} of {}",
            kk.kernel,
            chips[0].second.kernels.size());
        const auto& k0 = chips[0].second.kernels[kk.kernel];
        for (size_t c = 0; c < N; ++c) {
            const auto& k = chips[c].second.kernels[kk.kernel];
            // (the chips may place the kernel on different cores - e.g. the combine's port kernels on each chip's
            // own port cores: each chip's table holds the blocks of its own cores)
            if (!k.buffer_bindings.empty() || !k.common_buffer_bindings.empty()) {
                why = fmt::format("kernel {} ({}): it has buffer bindings", kk.kernel, k0.kernel_source);
                return {};
            }
            for (const auto& core : cores_of(k.core_ranges)) {
                pl.gx = std::max<uint32_t>(pl.gx, core.x + 1);
                pl.gy = std::max<uint32_t>(pl.gy, core.y + 1);
            }
            for (const auto& [core, a] : k.runtime_args) {
                pl.fix_at = std::max<uint32_t>(pl.fix_at, static_cast<uint32_t>(a.size()));
                const Key key{core.x, core.y};
                if (const auto it = kk.fixups[c].find(key); it != kk.fixups[c].end()) {
                    pl.nfix = std::max<uint32_t>(pl.nfix, static_cast<uint32_t>(it->second.size()));
                    for (const auto& f : it->second) {
                        if (f.index >= a.size() || f.src >= sources.size() || f.index > 0xFFFF) {
                            why = fmt::format("kernel {} core {}: fixup ({}, {}) out of range", kk.kernel, core.str(),
                                              f.index, f.src);  // fmt: skip
                            return {};
                        }
                    }
                }
                if (const auto it = kk.kept[c].find(key); it != kk.kept[c].end()) {
                    if (it->second.first + it->second.second > a.size()) {
                        why = fmt::format("kernel {} core {}: kept args past its {} args", kk.kernel, core.str(), a.size());
                        return {};
                    }
                }
            }
        }
        pl.block = (4 * (pl.fix_at + 1 + pl.nfix) + kAlign - 1) / kAlign * kAlign;
        if (pl.block > kk.bytes) {
            why = fmt::format(
                "kernel {} ({}): a block of {} args + {} fixups ({} B) does not fit its {} B args scratch",
                kk.kernel,
                k0.kernel_source,
                pl.fix_at,
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
    ArgsFixupResult out;
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
                blk[pl.fix_at] = n;
                for (uint32_t f = 0; f < n; ++f) {
                    const auto& fix = it->second[f];
                    blk[fix.index] -= sources[fix.src];  // (the arg as built: the source's address + its offset)
                    blk[pl.fix_at + 1 + f] = fix.index | (fix.src << 16);
                }
            }
        }
        std::vector<tt::tt_metal::distributed::ShardDataTransfer> writes;
        for (size_t c = 0; c < N; ++c) {
            writes.push_back(tt::tt_metal::distributed::ShardDataTransfer{chips[c].first}.host_data(data[c].data()));
        }
        mesh.mesh_command_queue().enqueue_write_shards(table, writes, /*blocking=*/true);

        std::vector<uint32_t> cas;
        for (size_t c = 0; c < N; ++c) {
            auto& k = chips[c].second.kernels[kk.kernel];
            const uint32_t ca = static_cast<uint32_t>(k.common_runtime_args.size());
            cas.push_back(ca);
            for (uint32_t x : {static_cast<uint32_t>(table->address()), pl.block, nbanks, pl.gx,
                               scratch_base + kk.l1_offset, static_cast<uint32_t>(sources.size())}) {  // fmt: skip
                k.common_runtime_args.push_back(x);
            }
            k.common_runtime_args.insert(k.common_runtime_args.end(), sources.begin(), sources.end());
            // (the kept args stay runtime args, from index 0; a core with none kept has no runtime args)
            decltype(k.runtime_args) kept;
            for (const auto& [core, a] : k.runtime_args) {
                if (const auto it = kk.kept[c].find(Key{core.x, core.y}); it != kk.kept[c].end() && it->second.second) {
                    const auto [start, count] = it->second;
                    kept.emplace_back(core, std::remove_cvref_t<decltype(a)>(a.begin() + start, a.begin() + start + count));
                }
            }
            k.runtime_args = std::move(kept);
            k.defines.emplace_back("MOE_BLOCK_ARGS_FIXUP", "1");
            k.defines.emplace_back("MOE_BLOCK_ARGS_CA", std::to_string(ca));
            k.defines.emplace_back("MOE_BLOCK_ARGS_FIX_AT", std::to_string(pl.fix_at));
        }
        out.ca.push_back(std::move(cas));
        tables->push_back(std::move(table));
    }
    out.tables = tables;
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block
