// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine_program_factory.hpp"

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
                add_combine_program(program, *p.plan, ChipCoord{coord[0], coord[1]}, mesh_device, bufs, p.flat);
            set_report(program, sv, p, bufs);
            chips.programs.push_back({ttnn::MeshCoordinateRange(coord, coord), std::move(program)});
            svs.push_back(std::move(sv));
        }
    }
    // (a merged program keeps chip 0's handles: every chip's args that the host rewrites from an address must be the
    // same args from the same source - the chips differ in offsets only)
    bool alike = true;
    for (size_t c = 1; c < svs.size() && alike; ++c) {
        const auto &a = svs[0], &b = svs[c];
        alike = a.flat.patches.size() == b.flat.patches.size() &&
                a.combine.patches.size() == b.combine.patches.size() &&
                a.combine.l1_args.size() == b.combine.l1_args.size() && a.flat.y_writers == b.flat.y_writers &&
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
        bool l_ok = alike;
        for (size_t i = 0; l_ok && i < a.combine.l1_args.size(); ++i) {
            l_ok = a.combine.l1_args[i].buf == b.combine.l1_args[i].buf;
        }
        bool cb_ok = alike;
        for (size_t i = 0; cb_ok && i < a.combine.arena_cbs.size(); ++i) {
            cb_ok = a.combine.arena_cbs[i].first == b.combine.arena_cbs[i].first;
        }
        alike = f_ok && c_ok && l_ok && cb_ok;
    }
    std::shared_ptr<const void> table;
    if (alike) {
        std::tie(chips, table) = merge_chip_workload(std::move(chips), mesh_device, "routed_experts_combine");
    } else {
        log_debug(tt::LogOp, "routed_experts_combine: a program a chip (the chips' address args differ)");
    }
    for (size_t i = 0; i < chips.programs.size(); ++i) {
        auto& [r, desc] = chips.programs[i];
        auto sv = std::move(svs[i]);
        tt::tt_metal::Program program(desc);
        flat_routed_expert::finalize_flat_handles(program, sv.flat, t.flat);
        finalize_combine_handles(program, sv.combine, bufs);
        sv.chip_table = table;
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
    using namespace routed_experts_combine_factory_detail;
    const auto bufs = buffers_of(t, output);
    for (auto& [range, sv] : cached.shared_variables) {
        auto& program = cached.workload.get_programs().at(range);
        flat_routed_expert::override_flat_routed_expert_addresses(program, sv.flat, t.flat);
        override_combine_addresses(program, sv.combine, bufs);
        if (sv.report != report_addrs(bufs)) {
            set_report(program, sv, p, bufs);
        }
    }
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
