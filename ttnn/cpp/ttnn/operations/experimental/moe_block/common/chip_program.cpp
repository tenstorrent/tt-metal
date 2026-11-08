// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chip_program.hpp"

#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <map>
#include <set>
#include <tuple>
#include <variant>

#include <fmt/format.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shard_data_transfer.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block {

namespace chip_program_detail {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;

bool same_config(const KernelDescriptor::ConfigDescriptor& a, const KernelDescriptor::ConfigDescriptor& b) {
    if (a.index() != b.index()) {
        return false;
    }
    if (const auto* x = std::get_if<DataMovementConfigDescriptor>(&a)) {
        const auto& y = std::get<DataMovementConfigDescriptor>(b);
        return x->processor == y.processor && x->noc == y.noc && x->noc_mode == y.noc_mode;
    }
    if (const auto* x = std::get_if<ComputeConfigDescriptor>(&a)) {
        const auto& y = std::get<ComputeConfigDescriptor>(b);
        return x->math_fidelity == y.math_fidelity && x->fp32_dest_acc_en == y.fp32_dest_acc_en &&
               x->dst_full_sync_en == y.dst_full_sync_en && x->unpack_to_dest_mode == y.unpack_to_dest_mode &&
               x->bfp8_pack_precise == y.bfp8_pack_precise && x->math_approx_mode == y.math_approx_mode &&
               x->enable_trisc2_rvv == y.enable_trisc2_rvv;
    }
    return true;  // (reader / writer: no fields)
}

bool same_formats(const CBDescriptor::FormatDescriptors& a, const CBDescriptor::FormatDescriptors& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].buffer_index != b[i].buffer_index || a[i].data_format != b[i].data_format ||
            a[i].page_size != b[i].page_size || a[i].tile.has_value() != b[i].tile.has_value() ||
            a[i].face_geometry.has_value() != b[i].face_geometry.has_value()) {
            return false;
        }
    }
    return true;
}

bool same_cb(const CBDescriptor& a, const CBDescriptor& b) {
    return a.total_size == b.total_size && a.core_ranges == b.core_ranges &&
           same_formats(a.format_descriptors, b.format_descriptors) &&
           same_formats(a.remote_format_descriptors, b.remote_format_descriptors) && a.buffer == b.buffer &&
           a.tensor == b.tensor && a.address_offset == b.address_offset &&
           a.global_circular_buffer == b.global_circular_buffer;
}

bool is_compute(const KernelDescriptor& k) { return std::holds_alternative<ComputeConfigDescriptor>(k.config); }

uint32_t chip_key(const CoreCoord& c) { return static_cast<uint32_t>(c.x) | (static_cast<uint32_t>(c.y) << 16); }

std::vector<CoreCoord> chip_cores(const CoreRangeSet& s) {
    std::vector<CoreCoord> v;
    for (const auto& r : s.ranges()) {
        for (auto y = r.start_coord.y; y <= r.end_coord.y; ++y) {
            for (auto x = r.start_coord.x; x <= r.end_coord.x; ++x) {
                v.emplace_back(x, y);
            }
        }
    }
    return v;
}

// one chip's patch entries of a kernel: per core, (index | common << 31, value)
using Entries = std::vector<std::pair<uint32_t, uint32_t>>;

// (where v differs from the base, the difference v - base, added at launch: the base is what the host writes - on a
// cache hit too, an address moved with the chips' same offsets kept - so the chip's value follows it; the caller pads
// the base to the longest)
void diff(const std::vector<uint32_t>& base, const std::vector<uint32_t>& v, uint32_t tag, Entries& out) {
    for (size_t i = 0; i < base.size(); ++i) {
        const uint32_t x = i < v.size() ? v[i] : 0u;
        if (x != base[i]) {
            out.emplace_back(static_cast<uint32_t>(i) | tag, x - base[i]);
        }
    }
}

// does the kernel's source run mb_loop_main.hpp's kernel_main (the chip's values applied first): a kernel that gets
// values without it would run with the base chip's (read once a source - a cache miss only)
bool has_chip_hook(const KernelDescriptor& k) {
    if (k.source_type != KernelDescriptor::SourceType::FILE_PATH) {
        return k.kernel_source.find("mb_loop_main.hpp") != std::string::npos;
    }
    static std::mutex mu;
    static std::map<std::string, bool> known;
    std::lock_guard lock(mu);
    if (const auto it = known.find(k.kernel_source); it != known.end()) {
        return it->second;
    }
    std::filesystem::path path(k.kernel_source);
    if (path.is_relative()) {
        const char* home = std::getenv("TT_METAL_HOME");
        path = std::filesystem::path(home != nullptr ? home : ".") / path;
    }
    std::ifstream f(path);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return known[k.kernel_source] = text.find("mb_loop_main.hpp") != std::string::npos;
}

}  // namespace chip_program_detail
using namespace chip_program_detail;

std::optional<ChipProgram> merge_chip_programs(
    tt::tt_metal::distributed::MeshDevice& mesh,
    std::vector<std::pair<tt::tt_metal::distributed::MeshCoordinate, ProgramDescriptor>> chips,
    std::string& why) {
    TT_FATAL(!chips.empty(), "merge_chip_programs: no chips");
    const size_t N = chips.size();
    ProgramDescriptor base = chips.front().second;
    if (N == 1) {
        return ChipProgram{std::move(base), nullptr};
    }
    // (structure: kernels and CBs alike on every chip)
    for (size_t c = 1; c < N; ++c) {
        const auto& p = chips[c].second;
        if (p.kernels.size() != base.kernels.size() || p.cbs.size() != base.cbs.size()) {
            why = fmt::format(
                "chip {}: {} kernels / {} CBs, the base {} / {}",
                c,
                p.kernels.size(),
                p.cbs.size(),
                base.kernels.size(),
                base.cbs.size());
            return std::nullopt;
        }
        for (size_t i = 0; i < p.cbs.size(); ++i) {
            if (!same_cb(p.cbs[i], base.cbs[i])) {
                why = fmt::format("chip {}: CB {} differs", c, i);
                return std::nullopt;
            }
        }
        for (size_t i = 0; i < p.kernels.size(); ++i) {
            const auto &a = base.kernels[i], &b = p.kernels[i];
            const char* what =
                a.kernel_source != b.kernel_source || a.source_type != b.source_type ? "source"
                : a.core_ranges != b.core_ranges                                     ? "cores"
                : a.compile_time_args != b.compile_time_args || a.named_compile_time_args != b.named_compile_time_args
                    ? "compile-time args"
                : a.defines != b.defines ? "defines"
                : a.opt_level != b.opt_level || a.compiler_include_paths != b.compiler_include_paths ||
                        !same_config(a.config, b.config)
                    ? "config"
                : a.common_buffer_bindings.size() != b.common_buffer_bindings.size() ||
                        a.buffer_bindings.size() != b.buffer_bindings.size()
                    ? "buffer bindings"
                    : nullptr;
            if (what != nullptr) {
                std::string detail;
                if (a.defines != b.defines) {
                    for (const auto& d : b.defines) {
                        if (std::find(a.defines.begin(), a.defines.end(), d) == a.defines.end()) {
                            detail += fmt::format(" {}={}", d.first, d.second);
                        }
                    }
                } else if (a.compile_time_args != b.compile_time_args) {
                    for (size_t j = 0; j < std::min(a.compile_time_args.size(), b.compile_time_args.size()); ++j) {
                        if (a.compile_time_args[j] != b.compile_time_args[j]) {
                            detail += fmt::format(" [{}] {} vs {}", j, a.compile_time_args[j], b.compile_time_args[j]);
                        }
                    }
                }
                why = fmt::format("chip {}: kernel {} ({}): {} differ{}", c, i, a.kernel_source, what, detail);
                return std::nullopt;
            }
            {  // (the same bindings, in any order: a program lists its cores in its own order)
                auto set_of = [](const KernelDescriptor& k) {
                    std::set<std::tuple<uint32_t, uint32_t, const void*>> out;
                    for (const auto& x : k.buffer_bindings) {
                        out.emplace(chip_key(x.core), x.arg_idx, x.buffer);
                    }
                    return out;
                };
                if (set_of(a) != set_of(b)) {
                    why = fmt::format("chip {}: kernel {} ({}) binds other buffers", c, i, a.kernel_source);
                    return std::nullopt;
                }
            }
            for (size_t j = 0; j < a.common_buffer_bindings.size(); ++j) {
                const auto &x = a.common_buffer_bindings[j], &y = b.common_buffer_bindings[j];
                if (x.arg_idx != y.arg_idx || x.buffer != y.buffer) {
                    why = fmt::format("chip {}: kernel {} ({}) binds other buffers", c, i, a.kernel_source);
                    return std::nullopt;
                }
            }
        }
    }

    // (semaphores: the union of each id's cores; an id starts at one value everywhere)
    {
        std::map<std::tuple<uint32_t, int>, SemaphoreDescriptor> sems;
        for (const auto& [coord, p] : chips) {
            for (const auto& s : p.semaphores) {
                const auto k = std::make_tuple(s.id, static_cast<int>(s.core_type));
                auto [it, fresh] = sems.emplace(k, s);
                if (!fresh) {
                    if (it->second.initial_value != s.initial_value) {
                        why = fmt::format(
                            "semaphore {} starts at {} and {}", s.id, it->second.initial_value, s.initial_value);
                        return std::nullopt;
                    }
                    it->second.core_ranges = it->second.core_ranges.merge(s.core_ranges);
                }
            }
        }
        base.semaphores.clear();
        for (auto& [k, s] : sems) {
            base.semaphores.push_back(std::move(s));
        }
    }

    // (per kernel: the base's args padded to the longest chip's, and every chip's differences)
    const size_t K = base.kernels.size();
    // patches[chip][kernel][core key] -> entries
    std::vector<std::vector<std::map<uint32_t, Entries>>> patches(N, std::vector<std::map<uint32_t, Entries>>(K));
    std::vector<bool> patched(K, false);
    for (size_t i = 0; i < K; ++i) {
        auto& kb = base.kernels[i];
        const auto cores = chip_cores(kb.core_ranges);
        // (each chip's args by core)
        std::vector<std::map<uint32_t, const std::vector<uint32_t>*>> rt(N);
        size_t lc = 0;
        for (size_t c = 0; c < N; ++c) {
            const auto& k = chips[c].second.kernels[i];
            for (const auto& [core, a] : k.runtime_args) {
                rt[c][chip_key(core)] = &a;
            }
            lc = std::max(lc, k.common_runtime_args.size());
        }
        // the base: chip 0's args, padded
        std::map<uint32_t, std::vector<uint32_t>> padded;
        for (const auto& core : cores) {
            size_t l = 0;
            for (size_t c = 0; c < N; ++c) {
                if (const auto it = rt[c].find(chip_key(core)); it != rt[c].end()) {
                    l = std::max(l, it->second->size());
                }
            }
            std::vector<uint32_t> v;
            if (const auto it = rt[0].find(chip_key(core)); it != rt[0].end()) {
                v = *it->second;
            }
            v.resize(l, 0u);
            padded[chip_key(core)] = std::move(v);
        }
        std::vector<uint32_t> common = kb.common_runtime_args;
        common.resize(lc, 0u);
        static const std::vector<uint32_t> none;
        for (size_t c = 0; c < N; ++c) {
            const auto& k = chips[c].second.kernels[i];
            Entries ce;
            diff(common, k.common_runtime_args, 0x80000000u, ce);
            for (const auto& core : cores) {
                const auto it = rt[c].find(chip_key(core));
                Entries e = ce;
                diff(padded.at(chip_key(core)), it != rt[c].end() ? *it->second : none, 0u, e);
                if (!e.empty()) {
                    patched[i] = true;
                    patches[c][i][chip_key(core)] = std::move(e);
                }
            }
        }
        for (const auto& b : kb.buffer_bindings) {  // (a buffer address is the same on every chip)
            for (size_t c = 0; c < N; ++c) {
                if (const auto it = patches[c][i].find(chip_key(b.core)); it != patches[c][i].end()) {
                    for (const auto& [ix, v] : it->second) {
                        if (ix == b.arg_idx) {
                            why = fmt::format("kernel {} ({}): a buffer's arg differs by chip", i, kb.kernel_source);
                            return std::nullopt;
                        }
                    }
                }
            }
        }
        kb.runtime_args.clear();
        for (const auto& core : cores) {
            if (!padded.at(chip_key(core)).empty() || rt[0].contains(chip_key(core))) {
                kb.runtime_args.emplace_back(core, padded.at(chip_key(core)));
            }
        }
        kb.common_runtime_args = std::move(common);
    }
    // (a compute kernel's values: a data-movement kernel on all its cores writes them into the compute's args - tagged
    // 1 << 30 - then raises the compute's flag (its last common arg), which the compute waits on - mb_chip.hpp)
    std::vector<bool> waits(K, false);
    for (size_t i = 0; i < K; ++i) {
        auto& kc = base.kernels[i];
        if (!patched[i] || !is_compute(kc)) {
            continue;
        }
        const auto ccores = chip_cores(kc.core_ranges);
        size_t host = K;
        for (size_t j = 0; j < K && host == K; ++j) {
            if (is_compute(base.kernels[j])) {
                continue;
            }
            const auto dc = chip_cores(base.kernels[j].core_ranges);
            std::set<uint32_t> have;
            for (const auto& c : dc) {
                have.insert(chip_key(c));
            }
            if (std::all_of(
                    ccores.begin(), ccores.end(), [&](const CoreCoord& c) { return have.contains(chip_key(c)); })) {
                host = j;
            }
        }
        if (host == K) {
            why = fmt::format(
                "compute kernel {} ({}): its args differ by chip and no data-movement kernel covers its cores",
                i,
                kc.kernel_source);
            return std::nullopt;
        }
        const uint32_t flag = static_cast<uint32_t>(kc.common_runtime_args.size());
        kc.common_runtime_args.push_back(0u);
        kc.defines.emplace_back("MB_CHIP_WAIT", "1");
        kc.defines.emplace_back("MB_CHIP_FLAG", std::to_string(flag));
        for (size_t c = 0; c < N; ++c) {
            for (const auto& core : ccores) {
                auto& to = patches[c][host][chip_key(core)];
                if (const auto it = patches[c][i].find(chip_key(core)); it != patches[c][i].end()) {
                    for (const auto& [ix, v] : it->second) {
                        to.emplace_back(ix | 0x40000000u, v);
                    }
                }
                to.emplace_back(flag | 0xC0000000u, 1u);  // (last: the compute's flag)
            }
            patches[c][i].clear();
        }
        patched[i] = false;
        patched[host] = true;
        waits[i] = true;
    }
    if (std::none_of(patched.begin(), patched.end(), [](bool b) { return b; })) {
        return ChipProgram{std::move(base), nullptr};
    }
    for (size_t i = 0; i < K; ++i) {
        if ((patched[i] || waits[i]) && !has_chip_hook(base.kernels[i])) {
            why = fmt::format(
                "kernel {} ({}) takes per-chip values but has no mb_loop_main.hpp", i, base.kernels[i].kernel_source);
            return std::nullopt;
        }
    }

    // (the patch blocks: per patched kernel its own DRAM table, a page per core of the grid (its block: the kernel's
    // largest), the same page on every chip, each chip's own entries)
    uint32_t gx = 1, gy = 1;
    for (size_t i = 0; i < K; ++i) {
        for (const auto& core : chip_cores(base.kernels[i].core_ranges)) {
            gx = std::max<uint32_t>(gx, core.x + 1);
            gy = std::max<uint32_t>(gy, core.y + 1);
        }
    }
    const size_t pages = static_cast<size_t>(gx) * gy;
    auto* allocator = mesh.allocator().get();
    const uint32_t nbanks = allocator->get_num_banks(tt::tt_metal::BufferType::DRAM);
    auto tables = std::make_shared<std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>>>();
    // (the unique + common args a Tensix core takes at most: the kernel-config buffer's fit is checked by the program)
    constexpr size_t kMaxArgs = 4094;
    for (size_t i = 0; i < K; ++i) {
        if (!patched[i]) {
            continue;
        }
        size_t most = 0;
        for (size_t c = 0; c < N; ++c) {
            for (const auto& [k, e] : patches[c][i]) {
                most = std::max(most, e.size());
            }
        }
        const uint32_t block = std::max<uint32_t>(64, (4 * (1 + 2 * static_cast<uint32_t>(most)) + 63) / 64 * 64);
        const size_t words = block / 4;
        auto table = tt::tt_metal::distributed::MeshBuffer::create(
            tt::tt_metal::distributed::ReplicatedBufferConfig{.size = pages * block},
            tt::tt_metal::distributed::DeviceLocalBufferConfig{
                .page_size = block, .buffer_type = tt::tt_metal::BufferType::DRAM},
            &mesh);
        auto& kb = base.kernels[i];
        const uint32_t ca = static_cast<uint32_t>(kb.common_runtime_args.size());
        for (uint32_t x : {static_cast<uint32_t>(table->address()), block, nbanks, 0u, gx}) {
            kb.common_runtime_args.push_back(x);
        }
        kb.common_runtime_args.resize(kb.common_runtime_args.size() + words + 16, 0u);  // (the block lands here)
        size_t most_rt = 0;
        for (const auto& [core, a] : kb.runtime_args) {
            most_rt = std::max(most_rt, a.size());
        }
        if (kb.common_runtime_args.size() + most_rt > kMaxArgs) {
            why = fmt::format(
                "kernel {} ({}): {} common + {} runtime args with its chip patch block ({} entries), over {}",
                i,
                kb.kernel_source,
                kb.common_runtime_args.size(),
                most_rt,
                most,
                kMaxArgs);
            return std::nullopt;
        }
        kb.defines.emplace_back("MB_CHIP", "1");
        kb.defines.emplace_back("MB_CHIP_CA", std::to_string(ca));
        std::vector<std::vector<uint32_t>> data(N, std::vector<uint32_t>(pages * words, 0u));
        for (size_t c = 0; c < N; ++c) {
            for (const auto& [k, e] : patches[c][i]) {
                uint32_t* w = data[c].data() + ((k >> 16) * gx + (k & 0xFFFF)) * words;
                w[0] = static_cast<uint32_t>(e.size());
                for (size_t j = 0; j < e.size(); ++j) {
                    w[1 + 2 * j] = e[j].first;
                    w[2 + 2 * j] = e[j].second;
                }
            }
        }
        std::vector<tt::tt_metal::distributed::ShardDataTransfer> writes;
        for (size_t c = 0; c < N; ++c) {
            writes.push_back(tt::tt_metal::distributed::ShardDataTransfer{chips[c].first}.host_data(data[c].data()));
        }
        mesh.mesh_command_queue().enqueue_write_shards(table, writes, /*blocking=*/true);
        tables->push_back(std::move(table));
    }
    return ChipProgram{std::move(base), std::move(tables)};
}

std::pair<tt::tt_metal::WorkloadDescriptor, std::shared_ptr<const void>> merge_chip_workload(
    tt::tt_metal::WorkloadDescriptor w, tt::tt_metal::distributed::MeshDevice& mesh, const char* op) {
    using tt::tt_metal::distributed::MeshCoordinate;
    if (w.programs.size() < 2) {
        return {std::move(w), nullptr};
    }
    std::vector<std::pair<MeshCoordinate, ProgramDescriptor>> chips;
    std::vector<uint32_t> lo, hi;
    for (const auto& prog : w.programs) {
        if (prog.range.shape().mesh_size() != 1) {
            return {std::move(w), nullptr};
        }
        const auto& c = prog.range.start_coord();
        if (lo.empty()) {
            lo.assign(c.coords().begin(), c.coords().end());
            hi = lo;
        }
        for (size_t d = 0; d < c.dims(); ++d) {
            lo[d] = std::min(lo[d], c[d]);
            hi[d] = std::max(hi[d], c[d]);
        }
        chips.emplace_back(c, prog.descriptor);
    }
    const tt::tt_metal::distributed::MeshCoordinateRange range{MeshCoordinate(lo), MeshCoordinate(hi)};
    if (range.shape().mesh_size() != chips.size()) {  // (not one rectangle)
        return {std::move(w), nullptr};
    }
    std::string why;
    auto merged = merge_chip_programs(mesh, std::move(chips), why);
    if (!merged.has_value()) {
        log_debug(tt::LogOp, "{}: a program a chip ({})", op, why);
        return {std::move(w), nullptr};
    }
    log_debug(tt::LogOp, "{}: one program for {} chips", op, range.shape().mesh_size());
    tt::tt_metal::WorkloadDescriptor one;
    one.programs.push_back({range, std::move(merged->desc)});
    return {std::move(one), std::move(merged->table)};
}

}  // namespace ttnn::operations::experimental::moe_block
