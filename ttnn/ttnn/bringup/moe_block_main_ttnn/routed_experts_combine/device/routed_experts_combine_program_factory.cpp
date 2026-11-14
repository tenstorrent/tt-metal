// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine_program_factory.hpp"

#include <atomic>
#include <unordered_map>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::bringup::moe_block_main::routed_experts_combine {

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

void set_fill(
    tt::tt_metal::Program& program,
    RoutedExpertsCombineFactory::shared_variables_t& sv,
    const RoutedExpertsCombineParams& p,
    const RoutedExpertsCombineInputs& t) {
    if (!p.fill) {
        return;
    }
    TT_FATAL(t.fill_done.has_value(), "routed_experts_combine: MOE_FILL without its done words");
    const CombineFillCall* call = p.fill_call.get();
    const bool ring = p.fill_exact && call != nullptr && call->exact.has_value() && call->exact_ring > 0;
    // P.60: no reader waits - no token (the waiter then writes no landed word)
    const bool wait = !(p.fill_exact && call != nullptr && call->exact.has_value() && !call->exact_wait);
    // P.55c: a token a call (never 0): the chip's waiter port writes it into the landed word once the copy has landed,
    // the flat expert's weight readers wait for it before the replica's weights (fx_wait.hpp)
    static std::atomic<uint32_t> next_token{1};
    uint32_t token = 0;
    while (p.fill_exact && wait && token == 0) {
        token = next_token.fetch_add(1);
    }
    const uint32_t done = static_cast<uint32_t>(t.fill_done->buffer()->address());
    if (ring) {  // (P.60: every fill arg is a fixed address or constant - written only when one of them changed)
        const auto addr = [](const Tensor& x) { return static_cast<uint32_t>(x.buffer()->address()); };
        std::vector<uint32_t> sig = {
            addr(call->gu),
            addr(call->d),
            addr(call->runs),
            addr(*call->exact),
            addr(*call->exact_ordinal),
            call->exact_ring,
            call->exact_limit,
            done,
            token,
            wait ? 1u : 0u,
            call->prev_counts ? addr(*call->prev_counts) : 0u,
            call->prev_landing ? addr(*call->prev_landing) : 0u,
            call->prev_sem,
            call->prev_spc,
            static_cast<uint32_t>(call->prev_core.x) | static_cast<uint32_t>(call->prev_core.y) << 16};
        if (sig == sv.fill_sig) {
            return;
        }
        sv.fill_sig = std::move(sig);
    } else {
        sv.fill_sig.clear();
    }
    set_combine_fill_args(
        program, sv.combine, *p.plan, sv.chip, *t.out.device(), p.fill_call.get(), done, p.fill_exact, token);
    if (p.fill_exact) {
        TT_FATAL(!sv.flat.fx_readers.empty(), "routed_experts_combine: the exact fill without the readers' waits");
        const auto rx = t.out.device()->worker_core_from_logical_core(p.plan->collector);
        const bool any = wait && p.fill_call && p.fill_call->exact.has_value();
        for (const auto& [k, core, idx] : sv.flat.fx_readers) {
            auto& rt = tt::tt_metal::GetRuntimeArgs(program, k, core);
            rt[idx] = any ? 1u : 0u;
            rt[idx + 1] = static_cast<uint32_t>(rx.x) | static_cast<uint32_t>(rx.y) << 16;
            rt[idx + 2] = done + 4 * kFxLanded;
            rt[idx + 3] = token;
            rt[idx + 4] = any ? p.fill_call->exact_limit : 0u;
            rt[idx + 5] = 0;
        }
    }
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
            // (P.55c: with the exact copy the flat expert schedules the spare slot last and waits for its copy)
            auto fcfg = p.flat_config;
            if (p.fill_exact) {
                fcfg.fx_spare = fcfg.experts_per_chip;
            }
            sv.flat = flat_routed_expert::add_flat_routed_expert_program(
                program, mesh_device, fcfg, *flat_routed_expert::flat_routed_expert_plan(mesh_device, fcfg), t.flat);
            TT_FATAL(
                sv.flat.y_writers.size() == p.flat.writers.size(),
                "routed_experts_combine: {} y writers in the program, {} planned",
                sv.flat.y_writers.size(),
                p.flat.writers.size());
            sv.chip = ChipCoord{coord[0], coord[1]};
            sv.combine =
                add_combine_program(program, *p.plan, sv.chip, mesh_device, bufs, p.flat, p.fill, p.fill_exact);
            set_report(program, sv, p, bufs);
            set_fill(program, sv, p, t);
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
        if (!p.cached_slots || sv.program == nullptr) {
            sv.program = &cached.workload.get_programs().at(range);
        }
        auto& program = *sv.program;
        flat_routed_expert::override_flat_routed_expert_addresses(program, sv.flat, t.flat, p.cached_slots);
        override_combine_addresses(program, sv.combine, bufs, p.cached_slots);
        if (sv.report != report_addrs(bufs)) {
            set_report(program, sv, p, bufs);
        }
        set_fill(program, sv, p, t);
    }
}

}  // namespace ttnn::operations::bringup::moe_block_main::routed_experts_combine
