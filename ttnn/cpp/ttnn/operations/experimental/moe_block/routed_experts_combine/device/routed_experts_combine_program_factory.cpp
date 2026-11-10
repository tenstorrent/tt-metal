// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine_program_factory.hpp"
#include <tracy/Tracy.hpp>

#include <algorithm>
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

std::array<uint32_t, 3> report_addrs(const CombineBuffers& b) {
    return {
        static_cast<uint32_t>(b[CombineBuf::Arena]->address()),
        static_cast<uint32_t>(b[CombineBuf::Table]->address()),
        static_cast<uint32_t>(b[CombineBuf::Bar]->address())};
}

// The address sources of the kernels' args in DRAM: the flat expert's FlatAddr buffers, then the combine's CombineBufs.
constexpr uint32_t kFlatSources = static_cast<uint32_t>(flat_routed_expert::FlatAddr::Count);
std::vector<uint32_t> arg_sources(const RoutedExpertsCombineInputs& t, const CombineBuffers& b) {
    const auto fa = flat_routed_expert::flat_source_addresses(t.flat);
    std::vector<uint32_t> s(fa.begin(), fa.end());
    for (size_t i = 0; i < static_cast<size_t>(CombineBuf::Count); ++i) {
        s.push_back(b.buf[i] != nullptr ? static_cast<uint32_t>(b.buf[i]->address()) : 0u);
    }
    return s;
}

// Where the kernels' args blocks land: the arena's args region (combine_host.hpp kArgsRows) on every core.
uint32_t args_base(const RoutedExpertsCombineParams& p, const CombineBuffers& b) {
    return static_cast<uint32_t>(b[CombineBuf::Arena]->address()) + args_offset(p.plan->arena_bytes);
}

// Moves every data-movement kernel's per-core args of the chips' programs into per-chip DRAM tables (common/
// args_table.hpp runtime_args_to_dram_fixup): each address arg the flat expert or the combine recorded becomes a fixup
// (its source's address added on the device), the senders' fabric connection args stay runtime args too. The compute
// kernels keep their one or two per-core args (the same on every chip).
void args_to_dram(
    tt::tt_metal::WorkloadDescriptor& chips,
    std::vector<RoutedExpertsCombineFactory::shared_variables_t>& svs,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const RoutedExpertsCombineParams& p,
    const RoutedExpertsCombineInputs& t,
    const CombineBuffers& bufs) {
    ZoneScopedN("routed_experts_combine: args to DRAM");
    const auto sources = arg_sources(t, bufs);
    const size_t N = chips.programs.size();
    // (the flat expert's descriptors hold its address args' offsets only - its handles write the addresses later: the
    // converter takes every address arg whole, so add them now)
    for (size_t c = 0; c < N; ++c) {
        auto& kernels = chips.programs[c].descriptor.kernels;
        for (const auto& pa : svs[c].flat.patches) {
            TT_FATAL(pa.kernel < kernels.size(), "routed_experts_combine: patch for kernel {} of {}", pa.kernel, kernels.size());
            for (auto& [core, a] : kernels[pa.kernel].runtime_args) {
                if (core == pa.core) {
                    a.at(pa.index) += sources[static_cast<size_t>(pa.src)];
                }
            }
        }
    }
    std::vector<ArgsFixupKernel> ak;
    const auto& k0 = chips.programs.at(0).descriptor.kernels;
    for (size_t i = 0; i < k0.size(); ++i) {
        const auto* dmc = std::get_if<tt::tt_metal::DataMovementConfigDescriptor>(&k0[i].config);
        if (dmc == nullptr) {
            continue;  // (a compute kernel)
        }
        ArgsFixupKernel kk;
        kk.kernel = i;
        kk.l1_offset = dmc->processor == tt::tt_metal::DataMovementProcessor::RISCV_0 ? 0u : kArgsHalfBytes;
        kk.bytes = kArgsHalfBytes;
        kk.fixups.resize(N);
        kk.kept.resize(N);
        for (size_t c = 0; c < N; ++c) {
            for (const auto& pa : svs[c].flat.patches) {
                if (pa.kernel == i) {
                    kk.fixups[c][{pa.core.x, pa.core.y}].push_back({pa.index, static_cast<uint32_t>(pa.src)});
                }
            }
            for (const auto& pa : svs[c].combine.patches) {
                if (pa.kernel == i) {
                    kk.fixups[c][{pa.core.x, pa.core.y}].push_back(
                        {pa.index, kFlatSources + static_cast<uint32_t>(pa.buf)});
                }
            }
            // the y writers' report blocks (per-core args - their destinations differ by chip): the counts address
            // and the L1 hand-off's buffer addresses (set_report wrote them whole)
            const auto& fl = svs[c].flat;
            for (const auto& [k, core] : fl.y_writers) {
                if (k != i) {
                    continue;
                }
                auto& fx = kk.fixups[c][{core.x, core.y}];
                fx.push_back({fl.y_report_arg + 2, kFlatSources + static_cast<uint32_t>(CombineBuf::Arena)});
                const auto& la = svs[c].combine.l1_args;
                for (size_t j = 0; j < la.size(); ++j) {
                    if (la[j].buf) {
                        fx.push_back(
                            {fl.y_report_arg + flat_routed_expert::kFlatReportArgs + static_cast<uint32_t>(j),
                             kFlatSources + static_cast<uint32_t>(*la[j].buf)});
                    }
                }
            }
            // (the senders' fabric connection args stay in the block only: the kernel opens the connection with
            // build_from_eth_channel - no runtime-arg copy, so the chips' programs stay alike)
        }
        ak.push_back(std::move(kk));
    }
    std::vector<std::pair<tt::tt_metal::distributed::MeshCoordinate, tt::tt_metal::ProgramDescriptor>> cv;
    for (auto& pr : chips.programs) {
        cv.emplace_back(pr.range.start_coord(), std::move(pr.descriptor));
    }
    std::string why;
    auto res = runtime_args_to_dram_fixup(mesh_device, cv, ak, sources, args_base(p, bufs), why);
    for (size_t c = 0; c < N; ++c) {
        chips.programs[c].descriptor = std::move(cv[c].second);
    }
    TT_FATAL(res.tables != nullptr, "routed_experts_combine: the kernels' args to DRAM: {}", why);
    // (the converted kernels' address args are now the sources in their common args: no per-core patch left)
    for (size_t c = 0; c < svs.size(); ++c) {
        auto& sv = svs[c];
        auto converted = [&](uint32_t k) {
            return std::any_of(ak.begin(), ak.end(), [&](const ArgsFixupKernel& x) { return x.kernel == k; });
        };
        std::erase_if(sv.flat.patches, [&](const auto& pa) { return converted(pa.kernel); });
        std::erase_if(sv.combine.patches, [&](const auto& pa) { return converted(pa.kernel); });
        sv.flat.args_in_dram = true;
        sv.args_tables = res.tables;
        sv.args_kernels.clear();
        for (size_t j = 0; j < ak.size(); ++j) {
            sv.args_kernels.push_back({static_cast<uint32_t>(ak[j].kernel), res.ca[j][c], ak[j].l1_offset});
        }
    }
}

// A cache hit with the args in DRAM: each converted kernel's sources and its block's L1 address (common args).
template <typename P>
void override_arg_sources(
    P& program,
    const RoutedExpertsCombineFactory::shared_variables_t& sv,
    const std::vector<uint32_t>& sources,
    uint32_t base) {
    for (const auto& [k, ca, off] : sv.args_kernels) {
        auto& common = tt::tt_metal::GetCommonRuntimeArgs(program, k);
        common[ca + kArgsFixupL1] = base + off;
        for (size_t s = 0; s < sources.size(); ++s) {
            common[ca + kArgsFixupSources + s] = sources[s];
        }
    }
}

template <typename P>
void set_report(
    P& program,
    RoutedExpertsCombineFactory::shared_variables_t& sv,
    const RoutedExpertsCombineParams& p,
    const CombineBuffers& b) {
    flat_routed_expert::set_flat_combine_report(
        program,
        sv.flat,
        p.flat_config,
        flat_routed_expert::FlatCombineReport{
            .collector = sv.combine.collector,
            .counts_addr = static_cast<uint32_t>(b[CombineBuf::Arena]->address()) + sv.combine.counts_offset,
            .l1_args = resolve_l1_args(sv.combine, b)});
    sv.report = report_addrs(b);
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
    std::vector<shared_variables_t> svs;
    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& coord : range) {
            tt::tt_metal::ProgramDescriptor program;
            shared_variables_t sv;
            // the flat routed expert on its cells (the same fragment on every chip), then the combine on the rest
            sv.flat = flat_routed_expert::add_flat_routed_expert_program(
                program,
                mesh_device,
                p.flat_config,
                *flat_routed_expert::flat_routed_expert_plan(mesh_device, p.flat_config),
                t.flat);
            TT_FATAL(
                sv.flat.y_writers.size() == p.flat.writers.size(),
                "routed_experts_combine: {} y writers in the program, {} planned",
                sv.flat.y_writers.size(),
                p.flat.writers.size());
            sv.combine =
                add_combine_program(program, *p.plan, chip_of(coord, p.plan->topo), mesh_device, bufs, p.flat);
            set_report(program, sv, p, bufs);
            chips.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(program)});
            svs.push_back(std::move(sv));
        }
    }
    // every data-movement kernel's per-core args into the per-chip DRAM tables: the chips' programs then differ in their
    // tables' contents only (and the senders' fabric args, kept as runtime args)
    args_to_dram(chips, svs, mesh_device, p, t, bufs);
    // (a merged program keeps chip 0's handles: every chip's args that the host rewrites from an address must be the
    // same args from the same source - the chips differ in offsets only)
    bool alike = true;
    for (size_t c = 1; c < svs.size() && alike; ++c) {
        const auto &a = svs[0], &b = svs[c];
        alike = a.flat.patches.size() == b.flat.patches.size() &&
                a.combine.patches.size() == b.combine.patches.size() &&
                (a.args_tables != nullptr || a.combine.l1_args.size() == b.combine.l1_args.size()) &&
                a.flat.y_writers == b.flat.y_writers &&
                a.combine.arena_cbs.size() == b.combine.arena_cbs.size() && a.flat.arena_cbs == b.flat.arena_cbs;
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
        const bool f_ok = alike && flat_set(a.flat) == flat_set(b.flat);
        const bool c_ok = alike && comb_set(a.combine) == comb_set(b.combine);
        // (the L1 hand-off: rewritten from chip 0's handles on a hit unless the args are in DRAM, where each chip's
        // block holds its own and the hit writes only the sources)
        bool l_ok = alike;
        for (size_t i = 0; l_ok && a.args_tables == nullptr && i < a.combine.l1_args.size(); ++i) {
            l_ok = a.combine.l1_args[i].buf == b.combine.l1_args[i].buf;
        }
        bool cb_ok = alike;
        for (size_t i = 0; cb_ok && i < a.combine.arena_cbs.size(); ++i) {
            cb_ok = a.combine.arena_cbs[i].first == b.combine.arena_cbs[i].first;
        }
        alike = f_ok && c_ok && l_ok && cb_ok;
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
                    a.compile_time_args != b.compile_time_args || a.defines != b.defines) {
                    note(fmt::format(
                        "kernel {} ({}): chip {} differs in its source, cores, compile-time args or defines",
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
    if (alike) {
        // (what differs between chips lives in the per-chip DRAM tables: the programs are alike unless the chips
        // place a kernel on different cores - then one program per chip, still nothing patched)
        std::string why;
        chips = merge_chip_workload(std::move(chips), "routed_experts_combine", &why);
        if (chips.programs.size() > 1) {
            log_debug(tt::LogOp, "routed_experts_combine: one program per chip: {}. {}", why, diff);
        }
    } else {
        log_debug(tt::LogOp, "routed_experts_combine: a program a chip (the chips' address args differ)");
    }
    for (size_t i = 0; i < chips.programs.size(); ++i) {
        auto& [r, desc] = chips.programs[i];
        auto sv = std::move(svs[i]);
        tt::tt_metal::Program program(desc);
        flat_routed_expert::finalize_flat_handles(program, sv.flat, t.flat);
        finalize_combine_handles(program, sv.combine, bufs);
        workload.add_program(r, std::move(program));
        shared.emplace(r, std::move(sv));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void RoutedExpertsCombineFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const RoutedExpertsCombineParams& p,
    const RoutedExpertsCombineInputs& t,
    ttnn::Tensor& output) {
    ZoneScopedN("routed_experts_combine: override (hit)");
    using namespace routed_experts_combine_factory_detail;
    const auto bufs = buffers_of(t, output);
    const auto sources = arg_sources(t, bufs);
    const uint32_t base = args_base(p, bufs);
    for (auto& [range, sv] : cached.shared_variables) {
        auto& program = cached.workload.get_programs().at(range);
        flat_routed_expert::override_flat_routed_expert_addresses(program, sv.flat, t.flat);
        override_combine_addresses(program, sv.combine, bufs);
        if (sv.args_tables != nullptr) {
            // (the report is per-core args in the DRAM tables, its addresses fixups: the sources cover it - nothing
            // per chip or per core is written on a hit)
            override_arg_sources(program, sv, sources, base);
        } else if (sv.report != report_addrs(bufs)) {
            set_report(program, sv, p, bufs);
        }
    }
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
