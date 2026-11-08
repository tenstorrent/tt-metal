// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine_program_factory.hpp"

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

void set_report(
    tt::tt_metal::Program& program,
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
    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& coord : range) {
            tt::tt_metal::Program program;
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
            const ttnn::MeshCoordinateRange r(coord, coord);
            workload.add_program(r, std::move(program));
            shared.emplace(r, std::move(sv));
        }
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
