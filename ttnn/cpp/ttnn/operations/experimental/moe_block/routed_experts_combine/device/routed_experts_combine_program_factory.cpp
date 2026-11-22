// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine_program_factory.hpp"
#include <tracy/Tracy.hpp>

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <variant>
#include <vector>
#include <set>
#include <tuple>
#include <unordered_map>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

namespace routed_experts_combine_factory_detail {

CombineBuffers buffers_of(const RoutedExpertsCombineInputs& t, const ttnn::Tensor& out) {
    CombineBuffers b;
    auto set = [&](CombineBuf k, tt::tt_metal::Buffer* p) { b.buf[static_cast<size_t>(k)] = p; };
    set(CombineBuf::Arena, t.flat.arena.buffer());
    set(CombineBuf::Plan, t.plan.buffer());
    set(CombineBuf::Table, t.table.buffer());
    set(CombineBuf::Out, out.buffer());
    set(CombineBuf::Bar, t.bar.buffer());
    set(CombineBuf::Hcache, t.hcache.buffer());
    set(CombineBuf::Lut, t.lut ? t.lut->buffer() : nullptr);
    return b;
}

// The address sources of the kernels' args in DRAM: the flat expert's FlatAddr buffers, then the combine's CombineBufs.
constexpr uint32_t kFlatSources = static_cast<uint32_t>(flat_routed_expert::FlatAddr::Count);
constexpr uint32_t kSources = kFlatSources + static_cast<uint32_t>(CombineBuf::Count);
std::array<uint32_t, kSources> arg_sources(const RoutedExpertsCombineInputs& t, const CombineBuffers& b) {
    std::array<uint32_t, kSources> s{};
    const auto fa = flat_routed_expert::flat_source_addresses(t.flat);
    std::copy(fa.begin(), fa.end(), s.begin());
    for (size_t i = 0; i < static_cast<size_t>(CombineBuf::Count); ++i) {
        s[kFlatSources + i] = b.buf[i] != nullptr ? static_cast<uint32_t>(b.buf[i]->address()) : 0u;
    }
    return s;
}

// Where the kernels' args blocks land: the arena's args region (combine_host.hpp kArgsRows) on every core.
uint32_t args_base(const RoutedExpertsCombineParams& p, const CombineBuffers& b) {
    return static_cast<uint32_t>(b[CombineBuf::Arena]->address()) + args_offset(p.plan->arena_bytes);
}

// One chip's program as built: the flat expert's and the combine's address args and report.
struct ChipBuild {
    CombineHandles combine;
    flat_routed_expert::FlatProgramHandles flat;
};

// Moves every data-movement kernel's per-core args of the chips' programs into per-chip DRAM tables (common/
// args_table.hpp runtime_args_to_dram): each address arg the flat expert or the combine recorded becomes a fixup (its
// source's address added on the device). The compute kernels keep their one or two per-core args (the same on every
// chip); their address args, if any, stay in the builds' patches.
struct ArgsToDram {
    ArgsTable table;
    std::vector<ArgsTableKernel> kernels;
};
ArgsToDram args_to_dram(
    tt::tt_metal::WorkloadDescriptor& chips,
    std::vector<ChipBuild>& builds,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const RoutedExpertsCombineParams& p,
    const RoutedExpertsCombineInputs& t,
    const CombineBuffers& bufs) {
    ZoneScopedN("routed_experts_combine: args to DRAM");
    const auto sources = arg_sources(t, bufs);
    const size_t N = chips.programs.size();
    // (the flat expert's descriptors hold its address args' offsets only: the converter takes every address arg whole,
    // so add the addresses now)
    for (size_t c = 0; c < N; ++c) {
        auto& kernels = chips.programs[c].descriptor.kernels;
        for (const auto& pa : builds[c].flat.patches) {
            TT_FATAL(
                pa.kernel < kernels.size(),
                "routed_experts_combine: patch for kernel {} of {}",
                pa.kernel,
                kernels.size());
            for (auto& [core, a] : kernels[pa.kernel].runtime_args) {
                if (core == pa.core) {
                    a.at(pa.index) += sources[static_cast<size_t>(pa.src)];
                }
            }
        }
    }
    ArgsToDram out;
    auto& ak = out.kernels;
    const auto& k0 = chips.programs.at(0).descriptor.kernels;
    for (size_t i = 0; i < k0.size(); ++i) {
        const auto* dmc = std::get_if<tt::tt_metal::DataMovementConfigDescriptor>(&k0[i].config);
        if (dmc == nullptr) {
            continue;  // (a compute kernel)
        }
        ArgsTableKernel kk;
        kk.kernel = i;
        kk.l1_offset = dmc->processor == tt::tt_metal::DataMovementProcessor::RISCV_0 ? 0u : kArgsHalfBytes;
        kk.bytes = kArgsHalfBytes;
        kk.fixups.resize(N);
        kk.kept.resize(N);
        for (size_t c = 0; c < N; ++c) {
            for (const auto& pa : builds[c].flat.patches) {
                if (pa.kernel == i) {
                    kk.fixups[c][{pa.core.x, pa.core.y}].push_back({pa.index, static_cast<uint32_t>(pa.src)});
                }
            }
            for (const auto& pa : builds[c].combine.patches) {
                if (pa.kernel == i) {
                    kk.fixups[c][{pa.core.x, pa.core.y}].push_back(
                        {pa.index, kFlatSources + static_cast<uint32_t>(pa.buf)});
                }
            }
            // the y writers' report blocks (per-core args - their destinations differ by chip): the counts address
            // and the L1 hand-off's buffer addresses (set_report wrote them whole)
            const auto& fl = builds[c].flat;
            for (const auto& [k, core] : fl.y_writers) {
                if (k != i) {
                    continue;
                }
                auto& fixups = kk.fixups[c][{core.x, core.y}];
                fixups.push_back({fl.y_report_arg + 2, kFlatSources + static_cast<uint32_t>(CombineBuf::Arena)});
                const auto& la = builds[c].combine.l1_args;
                for (size_t j = 0; j < la.size(); ++j) {
                    if (la[j].buf) {
                        fixups.push_back(
                            {fl.y_report_arg + flat_routed_expert::kFlatReportArgs + static_cast<uint32_t>(j),
                             kFlatSources + static_cast<uint32_t>(*la[j].buf)});
                    }
                }
            }
            // the senders' fabric connection args stay runtime args too (kept: from index 0), opened with
            // build_from_args; their eth channel differs by chip - a per-device runtime arg of the one program
            for (const auto& f : builds[c].combine.fabric_args) {
                if (f.kernel == i) {
                    kk.kept[c][{f.core.x, f.core.y}] = {f.first, f.count};
                }
            }
        }
        ak.push_back(std::move(kk));
    }
    ChipPrograms cv;
    for (auto& pr : chips.programs) {
        cv.emplace_back(pr.range.start_coord(), std::move(pr.descriptor));
    }
    std::string why;
    out.table = runtime_args_to_dram(mesh_device, cv, ak, sources, args_base(p, bufs), why);
    for (size_t c = 0; c < N; ++c) {
        chips.programs[c].descriptor = std::move(cv[c].second);
    }
    TT_FATAL(out.table.tables != nullptr, "routed_experts_combine: the kernels' args to DRAM: {}", why);
    // (the converted kernels' address args are now the sources in their common args: no per-core arg left)
    auto converted = [&](uint32_t k) {
        return std::any_of(ak.begin(), ak.end(), [&](const ArgsTableKernel& x) { return x.kernel == k; });
    };
    for (auto& build : builds) {
        std::erase_if(build.flat.patches, [&](const auto& pa) { return converted(pa.kernel); });
        std::erase_if(build.combine.patches, [&](const auto& pa) { return converted(pa.kernel); });
    }
    return out;
}

// The y writers' report into chip's descriptor (before its args move to DRAM).
void set_report(
    tt::tt_metal::ProgramDescriptor& program,
    const ChipBuild& build,
    const RoutedExpertsCombineParams& p,
    const CombineBuffers& b) {
    flat_routed_expert::set_flat_combine_report(
        program,
        build.flat,
        p.flat_config,
        flat_routed_expert::FlatCombineReport{
            .collector = build.combine.collector,
            .counts_addr = static_cast<uint32_t>(b[CombineBuf::Arena]->address()) + build.combine.counts_offset,
            .l1_args = resolve_l1_args(build.combine, b)});
}

// What a cache hit writes into program `chip`'s (chip 0's for one program for the mesh): the converted kernels' common
// args, the address args left in other kernels, the arena's CBs.
ProgramAddresses program_addresses(
    const ArgsToDram& converted,
    const ChipBuild& build,
    size_t chip,
    const RoutedExpertsCombineParams& p,
    const tt::tt_metal::ProgramDescriptor& desc,
    const tt::tt_metal::Buffer* arena) {
    ProgramAddresses out{
        .tables = converted.table.tables,
        .table_kernels = table_kernels(
            converted.table,
            converted.kernels,
            chip,
            kFlatSources + static_cast<uint32_t>(CombineBuf::Arena),
            args_offset(p.plan->arena_bytes)),
        .args = {},
        .arena_cbs = arena_backed_cbs(desc, arena)};
    for (const auto& pa : build.flat.patches) {
        out.args.push_back(ProgramAddresses::AddressArg{
            .kernel = pa.kernel,
            .common = false,
            .core = pa.core,
            .index = pa.index,
            .source = static_cast<uint32_t>(pa.src),
            .offset = pa.offset});
    }
    for (const auto& pa : build.combine.patches) {
        out.args.push_back(ProgramAddresses::AddressArg{
            .kernel = pa.kernel,
            .common = false,
            .core = pa.core,
            .index = pa.index,
            .source = kFlatSources + static_cast<uint32_t>(pa.buf),
            .offset = pa.offset});
    }
    return out;
}

}  // namespace routed_experts_combine_factory_detail

RoutedExpertsCombineFactory::cached_mesh_workload_t RoutedExpertsCombineFactory::create_mesh_workload(
    const RoutedExpertsCombineParams& p,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const RoutedExpertsCombineInputs& t,
    ttnn::Tensor& output) {
    ZoneScopedN("routed_experts_combine: create (miss)");
    using namespace routed_experts_combine_factory_detail;
    TT_FATAL(p.plan && p.flat_plan, "routed_experts_combine: no plans (call through the host entry point)");
    auto& mesh_device = *t.out.device();
    const auto bufs = buffers_of(t, output);
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    // (each chip's program descriptor, then one program for the mesh where they merge - common/chip_program.hpp)
    tt::tt_metal::WorkloadDescriptor chips;
    std::vector<ChipBuild> builds;
    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& coord : range) {
            tt::tt_metal::ProgramDescriptor program;
            ChipBuild build;
            // the flat routed expert on its cores (the same fragment on every chip), then the combine on the rest
            build.flat = flat_routed_expert::add_flat_routed_expert_program(
                program,
                mesh_device,
                p.flat_config,
                *flat_routed_expert::flat_routed_expert_plan(mesh_device, p.flat_config),
                t.flat);
            TT_FATAL(
                build.flat.y_writers.size() == p.flat.writers.size(),
                "routed_experts_combine: {} y writers in the program, {} planned",
                build.flat.y_writers.size(),
                p.flat.writers.size());
            build.combine =
                add_combine_program(program, *p.plan, chip_of(coord, p.plan->topo), mesh_device, bufs, p.flat);
            set_report(program, build, p, bufs);
            chips.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(program)});
            builds.push_back(std::move(build));
        }
    }
    // every data-movement kernel's per-core args into the per-chip DRAM tables: the chips' programs then differ in
    // their tables' contents only
    const auto converted = args_to_dram(chips, builds, mesh_device, p, t, bufs);
    // (a merged program keeps chip 0's address args: every chip's args that the host rewrites from an address must be
    // the same args from the same source - the chips differ in offsets only; the CBs merge_chip_workload compares)
    bool alike = true;
    for (size_t c = 1; c < builds.size() && alike; ++c) {
        const auto &a = builds[0], &b = builds[c];
        alike = a.flat.patches.size() == b.flat.patches.size() &&
                a.combine.patches.size() == b.combine.patches.size() && a.flat.y_writers == b.flat.y_writers;
        // (the same args from the same sources, in any order)
        auto flat_set = [](const flat_routed_expert::FlatProgramHandles& f) {
            std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, int>> out;
            for (const auto& x : f.patches) {
                out.emplace(x.kernel, x.core.x, x.core.y, x.index, static_cast<int>(x.src));
            }
            return out;
        };
        auto comb_set = [](const CombineHandles& h) {
            std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, int>> out;
            for (const auto& x : h.patches) {
                out.emplace(x.kernel, x.core.x, x.core.y, x.index, static_cast<int>(x.buf));
            }
            return out;
        };
        alike = alike && flat_set(a.flat) == flat_set(b.flat) && comb_set(a.combine) == comb_set(b.combine);
    }
    // (which args still differ between the chips, for the error below: every difference, the first kMaxDiffs listed -
    // kernel, common / runtime, core, index, chip 0's value and the other chip's; a missing arg reads as 0, as the
    // merge pads it)
    auto differences = [&]() -> std::string {
        constexpr size_t kMaxDiffs = 20;
        const auto& p0 = chips.programs.at(0).descriptor;
        std::vector<std::string> lines;
        size_t total = 0;
        auto note = [&](std::string line) {
            ++total;
            if (lines.size() < kMaxDiffs) {
                lines.push_back(std::move(line));
            }
        };
        auto at = [](const std::vector<uint32_t>& v, size_t i) { return i < v.size() ? v[i] : 0u; };
        static const std::vector<uint32_t> none;
        for (size_t c = 1; c < chips.programs.size(); ++c) {
            const auto& pc = chips.programs[c].descriptor;
            if (pc.kernels.size() != p0.kernels.size()) {
                note(fmt::format("chip {}: {} kernels, chip 0 {}", c, pc.kernels.size(), p0.kernels.size()));
                continue;
            }
            for (size_t k = 0; k < p0.kernels.size(); ++k) {
                const auto& a = p0.kernels[k];
                const auto& b = pc.kernels[k];
                if (a.kernel_source != b.kernel_source || a.core_ranges != b.core_ranges ||
                    a.compile_time_args != b.compile_time_args ||
                    a.named_compile_time_args != b.named_compile_time_args || a.defines != b.defines) {
                    note(fmt::format(
                        "kernel {} ({}): chip {} differs in its source, cores, compile-time args (positional or "
                        "named) or defines",
                        k,
                        a.kernel_source,
                        c));
                }
                const auto &ca = a.common_runtime_args, &cb = b.common_runtime_args;
                for (size_t i = 0; i < std::max(ca.size(), cb.size()); ++i) {
                    if (at(ca, i) != at(cb, i)) {
                        note(fmt::format(
                            "kernel {} ({}) common arg {} (of {} / {}): chip 0 = {}, chip {} = {}",
                            k,
                            a.kernel_source,
                            i,
                            ca.size(),
                            cb.size(),
                            at(ca, i),
                            c,
                            at(cb, i)));
                    }
                }
                // (per core: the union of both chips' cores with args)
                using ArgsPtr = const std::vector<uint32_t>*;
                std::map<std::pair<size_t, size_t>, std::pair<ArgsPtr, ArgsPtr>> by_core;
                for (const auto& [core, v] : a.runtime_args) {
                    by_core[{core.x, core.y}].first = &v;
                }
                for (const auto& [core, v] : b.runtime_args) {
                    by_core[{core.x, core.y}].second = &v;
                }
                for (const auto& [xy, ab] : by_core) {
                    const auto& va = ab.first != nullptr ? *ab.first : none;
                    const auto& vb = ab.second != nullptr ? *ab.second : none;
                    for (size_t i = 0; i < std::max(va.size(), vb.size()); ++i) {
                        if (at(va, i) != at(vb, i)) {
                            note(fmt::format(
                                "kernel {} ({}) core ({}, {}) runtime arg {} (of {} / {}): chip 0 = {}, chip {} = {}",
                                k,
                                a.kernel_source,
                                xy.first,
                                xy.second,
                                i,
                                va.size(),
                                vb.size(),
                                at(va, i),
                                c,
                                at(vb, i)));
                        }
                    }
                }
            }
        }
        if (total == 0) {
            return "no runtime / common arg, source, core set, compile-time arg or define differs (semaphores or CBs?)";
        }
        std::string out = fmt::format("{} difference(s)", total);
        if (total > lines.size()) {
            out += fmt::format(", the first {}", lines.size());
        }
        out += ":";
        for (const auto& l : lines) {
            out += "\n  " + l;
        }
        return out;
    };
    const std::string diff = alike ? differences() : std::string{};
    ChipWorkload one;
    if (alike) {
        // (what differs between chips lives in the per-chip DRAM tables but for the fabric connections' args, which
        // become per-device runtime args: the programs are alike unless the chips place a kernel on different cores -
        // then one program per chip, still nothing patched)
        std::string why;
        one = merge_chip_workload(mesh_device, std::move(chips), "routed_experts_combine", &why);
        if (one.workload.programs.size() > 1) {
            log_debug(tt::LogOp, "routed_experts_combine: one program per chip: {}. {}", why, diff);
        }
    } else {
        log_debug(tt::LogOp, "routed_experts_combine: a program a chip (the chips' address args differ)");
        one.per_device_runtime_args.resize(chips.programs.size());
        one.workload = std::move(chips);
    }
    // (one program for the mesh: chip 0's; or a program a chip, in chip order - merge_chip_workload keeps the order)
    TT_FATAL(
        one.workload.programs.size() == 1 || one.workload.programs.size() == builds.size(),
        "routed_experts_combine: {} programs for {} chips",
        one.workload.programs.size(),
        builds.size());
    for (size_t i = 0; i < one.workload.programs.size(); ++i) {
        const auto& [r, desc] = one.workload.programs[i];
        // (the descriptor holds every address whole: the flat expert's were added before the args moved)
        shared_variables_t sv{
            .addresses = program_addresses(converted, builds[i], i, p, desc, bufs[CombineBuf::Arena])};
        add_chip_program(workload, one, i, tt::tt_metal::Program(desc));
        shared.emplace(r, std::move(sv));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void RoutedExpertsCombineFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const RoutedExpertsCombineParams& /*p*/,
    const RoutedExpertsCombineInputs& t,
    ttnn::Tensor& output) {
    ZoneScopedN("routed_experts_combine: override (hit)");
    using namespace routed_experts_combine_factory_detail;
    const auto bufs = buffers_of(t, output);
    const auto sources = arg_sources(t, bufs);
    for (auto& [range, sv] : cached.shared_variables) {
        // (the report is per-core args in the DRAM tables, its addresses fixups: the sources cover it)
        write_program_addresses(
            cached.workload.get_programs().at(range), sv.addresses, sources, bufs[CombineBuf::Arena]);
    }
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
