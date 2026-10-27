// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_with_program_device_operation.hpp"

#include <unordered_map>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/impl/program/program_impl.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace ttnn::operations::bringup::flat_routed_expert {

void FlatWithProgramDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& p, const tensor_args_t& t) {
    if (std::getenv("FLAT_WP_KEYLOG")) {  // (debug: the cache key of every miss)
        log_info(
            tt::LogOp,
            "flat_with_program miss: extra_key {} writers {} counts_addr {} addrs {} flat_key {}",
            p.extra_key,
            p.writers,
            p.counts_addr,
            p.addrs,
            p.flat_key);
    }
    TT_FATAL(p.flat.cmb_rt > 0, "flat_with_program: the report args are not set");
    TT_FATAL(
        p.extra && !p.extra->mesh_programs.empty(),
        "flat_with_program: no programs to append (a program-cache miss: the caller passed no extra program, expecting "
        "a hit)");
    TT_FATAL(p.writers > 0, "flat_with_program: no y writers");
    (void)t;
}

namespace {

// The y writers' report args (collector, counts_addr) and, with the L1 hand-off, this appended program's hand-off args
// (the writer index filled in): set on a miss, re-applied on a patch_addresses hit.
void apply_writer_args(
    tt::tt_metal::Program& program,
    const FlatRoutedExpertSharedVariables& fsv,
    const FlatWithProgramParams& p,
    const tt::tt_metal::CoreCoord& cv,
    uint32_t prog_i) {
    using namespace tt::tt_metal;
    for (const auto& [k, core] : fsv.cmb_writers) {
        auto& a = GetRuntimeArgs(program, k, core);
        a[p.flat.cmb_rt + 0] = static_cast<uint32_t>(cv.x);
        a[p.flat.cmb_rt + 1] = static_cast<uint32_t>(cv.y);
        a[p.flat.cmb_rt + 2] = p.counts_addr;
        a[p.flat.cmb_rt + 3] = 0;  // (no go handshake: SE_CMB_NO_GO)
    }
    if (p.l1_args.empty()) {
        return;
    }
    TT_FATAL(prog_i < p.l1_args.size(), "flat_with_program: l1_args per appended program");
    const auto& l1 = p.l1_args[prog_i];
    TT_FATAL(l1.size() <= FLAT_Y_L1_ARGS, "flat_with_program: {} L1 hand-off args > {}", l1.size(), FLAT_Y_L1_ARGS);
    uint32_t widx = 0;
    for (const auto& [k, core] : fsv.cmb_writers) {
        auto& a = GetRuntimeArgs(program, k, core);
        TT_FATAL(
            a.size() >= p.flat.cmb_rt + 4 + FLAT_Y_L1_ARGS, "flat_with_program: writer args too short (MIMO_FL_Y_L1?)");
        for (uint32_t j = 0; j < l1.size(); ++j) {
            a[p.flat.cmb_rt + 4 + j] = l1[j];
        }
        a[p.flat.cmb_rt + 4 + 4] = widx++;
    }
}

// patch_addresses: the appended program's runtime args (per core and common) from its descriptor, at the appended
// kernels' handles (Program::append creates them in the descriptor's order after the flat program's kernels).
void apply_extra_runtime_args(
    tt::tt_metal::Program& program, const tt::tt_metal::ProgramDescriptor& desc, uint32_t k0) {
    using namespace tt::tt_metal;
    for (uint32_t k = 0; k < desc.kernels.size(); ++k) {
        const auto& kd = desc.kernels[k];
        for (const auto& [core, args] : kd.runtime_args) {
            auto& a = GetRuntimeArgs(program, k0 + k, core);
            TT_FATAL(a.size() == args.size(), "flat_with_program: appended kernel {} args changed size", k);
            for (uint32_t i = 0; i < args.size(); ++i) {
                a[i] = args[i];
            }
        }
        if (!kd.common_runtime_args.empty()) {
            auto& a = GetCommonRuntimeArgs(program, k0 + k);
            for (uint32_t i = 0; i < kd.common_runtime_args.size(); ++i) {
                a[i] = kd.common_runtime_args[i];
            }
        }
    }
}

using AddrArray = std::array<uint32_t, static_cast<size_t>(AddrSrc::Count)>;

// every buffer address of the call, by AddrSrc (as FlatRoutedExpertProgramFactory::apply_addresses reads them)
AddrArray call_addresses(const FlatRoutedExpertInputs& t, const Tensor& output) {
    AddrArray a{};
    a[size_t(AddrSrc::X)] = t.x.buffer()->address();
    a[size_t(AddrSrc::Y)] = output.buffer()->address();
    a[size_t(AddrSrc::Counts)] = t.counts.buffer()->address();
    a[size_t(AddrSrc::Regions)] = t.regions.buffer()->address();
    a[size_t(AddrSrc::Ids)] = t.global_expert_ids.buffer()->address();
    a[size_t(AddrSrc::GateUp)] = t.gate_up_weights.buffer()->address();
    a[size_t(AddrSrc::Down)] = t.down_weights.buffer()->address();
    a[size_t(AddrSrc::ReaderDown)] = t.reader_down_weights ? t.reader_down_weights->buffer()->address() : 0;
    a[size_t(AddrSrc::Done)] = t.done_words.buffer()->address();
    a[size_t(AddrSrc::Arena)] = t.arena.buffer()->address();
    a[size_t(AddrSrc::Words)] = t.words.buffer()->address();
    a[size_t(AddrSrc::TokenIndex)] = t.token_index ? t.token_index->buffer()->address() : 0;
    return a;
}

}  // namespace

FlatWithProgramFactory::cached_mesh_workload_t FlatWithProgramFactory::create_mesh_workload(
    const FlatWithProgramParams& p,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const FlatRoutedExpertInputs& t,
    Tensor& output) {
    using namespace tt::tt_metal;
    (void)tensor_coords;
    const auto cv = t.x.device()->worker_core_from_logical_core(p.collector);
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    uint32_t prog_i = 0;
    for (const auto& [range, desc_in] : p.extra->mesh_programs) {
        TT_FATAL(range.shape().mesh_size() == 1, "flat_with_program: one appended program per chip");
        auto flat = FlatRoutedExpertProgramFactory::create(p.flat, t, output);
        shared_variables_t sv;
        sv.prog_i = prog_i;
        sv.extra_k0 = static_cast<uint32_t>(flat.program.impl().num_kernels());
        const size_t cb0 = flat.program.circular_buffers().size();
        if (p.patch_addresses) {
            // (the appended program's buffers live in the arena, on its own cells: its globally allocated CBs are
            // bound to THIS call's arena at their offsets - the caller's descriptor may still name an earlier,
            // freed arena tensor of the same address - and re-pointed at the call's arena on every hit)
            ProgramDescriptor desc = desc_in;
            for (auto& cb : desc.cbs) {
                TT_FATAL(
                    cb.global_circular_buffer == nullptr,
                    "flat_with_program(patch_addresses): an appended global circular buffer cannot be re-pointed");
                if (cb.buffer != nullptr || cb.tensor != nullptr) {
                    cb.buffer = t.arena.buffer();
                    cb.tensor = nullptr;
                }
            }
            flat.program.append(desc);
            const auto cbs = flat.program.circular_buffers();
            for (size_t i = 0; i < desc.cbs.size(); ++i) {
                if (desc.cbs[i].buffer != nullptr) {
                    sv.extra_cbs.emplace_back(cbs[cb0 + i]->id(), desc.cbs[i].address_offset);
                }
            }
            sv.extra_addrs = p.extra_addrs;
            sv.flat = flat.shared_variables;
            for (const auto& pa : flat.shared_variables.patches) {
                sv.by_src[size_t(pa.src)].push_back(
                    {&GetRuntimeArgs(flat.program, pa.kernel, pa.core), pa.index, pa.offset});
            }
            sv.addr = call_addresses(t, output);  // (create applied these)
            sv.counts_addr = p.counts_addr;
            if (!p.l1_args.empty()) {
                sv.l1 = p.l1_args.at(prog_i);
            }
        } else {
            flat.program.append(desc_in);
        }
        apply_writer_args(flat.program, flat.shared_variables, p, cv, prog_i);
        ++prog_i;
        if (std::getenv("FLAT_WP_LOG") && range.start_coord() == p.extra->mesh_programs.front().first.start_coord()) {
            const uint32_t ti =
                MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
            for (const auto& kg : flat.program.impl().get_kernel_groups(ti)) {
                auto kc = kg->launch_msg.view().kernel_config();
                log_info(
                    tt::LogOp,
                    "flat_with_program: kernel group {} kernels {} local_cb_mask {:#x} min_remote_cb {} rta {}",
                    kg->core_ranges.str(),
                    kg->kernel_ids.size(),
                    static_cast<uint64_t>(kc.local_cb_mask()),
                    static_cast<uint32_t>(kc.min_remote_cb_start_index()),
                    kg->total_rta_size);
            }
        }
        TT_FATAL(
            flat.shared_variables.cmb_writers.size() == p.writers,
            "flat_with_program: {} y writers in the program, {} declared",
            flat.shared_variables.cmb_writers.size(),
            p.writers);
        workload.add_program(range, std::move(flat.program));
        shared[range] = std::move(sv);
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void FlatWithProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached, const FlatWithProgramParams& p, const FlatRoutedExpertInputs& t, Tensor& output) {
    using namespace tt::tt_metal;
    if (!p.patch_addresses) {
        return;  // (the key holds every address: a hit is the same buffers)
    }
    const auto cv = t.x.device()->worker_core_from_logical_core(p.collector);
    const AddrArray now = call_addresses(t, output);
    const uint32_t arena = now[size_t(AddrSrc::Arena)];
    for (auto& [range, program] : cached.workload.get_programs()) {
        auto& sv = cached.shared_variables.at(range);
        // (a hit writes only what moved: the args of every buffer at a new address, the arena CBs when the arena
        // moved, the writers' args when they changed; the rest still holds what the miss or an earlier hit wrote)
        for (size_t s = 0; s < now.size(); ++s) {
            if (now[s] == sv.addr[s]) {
                continue;
            }
            for (const auto& pa : sv.by_src[s]) {
                (*pa.args)[pa.index] = now[s] + pa.offset;
            }
        }
        if (arena != sv.addr[size_t(AddrSrc::Arena)]) {
            for (const auto& [cb, off] : sv.flat.arena_cbs) {
                UpdateDynamicCircularBufferAddress(program, cb, *t.arena.buffer(), off);
            }
            for (const auto& [cb, off] : sv.extra_cbs) {
                UpdateDynamicCircularBufferAddress(program, cb, *t.arena.buffer(), off);
            }
        }
        sv.addr = now;
        const std::vector<uint32_t> no_l1;
        const auto& l1 = p.l1_args.empty() ? no_l1 : p.l1_args.at(sv.prog_i);
        if (p.counts_addr != sv.counts_addr || l1 != sv.l1) {
            apply_writer_args(program, sv.flat, p, cv, sv.prog_i);
            sv.counts_addr = p.counts_addr;
            sv.l1 = l1;
        }
        if (sv.extra_addrs != p.extra_addrs) {
            TT_FATAL(
                p.extra,
                "flat_with_program(patch_addresses): the appended programs' addresses changed (extra_addrs {} -> {}): "
                "pass their descriptor",
                sv.extra_addrs,
                p.extra_addrs);
            const ProgramDescriptor* desc = nullptr;
            for (const auto& [r, d] : p.extra->mesh_programs) {
                if (r == range) {
                    desc = &d;
                    break;
                }
            }
            TT_FATAL(desc != nullptr, "flat_with_program(patch_addresses): no appended program for range {}", range);
            apply_extra_runtime_args(program, *desc, sv.extra_k0);
            sv.extra_addrs = p.extra_addrs;
        }
    }
}

}  // namespace ttnn::operations::bringup::flat_routed_expert

namespace ttnn::prim::bringup {
ttnn::Tensor flat_with_program(
    const ttnn::operations::bringup::flat_routed_expert::FlatWithProgramParams& params,
    const ttnn::operations::bringup::flat_routed_expert::FlatRoutedExpertInputs& inputs) {
    using Op = ttnn::operations::bringup::flat_routed_expert::FlatWithProgramDeviceOperation;
    return ttnn::device_operation::launch<Op>(params, inputs);
}
}  // namespace ttnn::prim::bringup
