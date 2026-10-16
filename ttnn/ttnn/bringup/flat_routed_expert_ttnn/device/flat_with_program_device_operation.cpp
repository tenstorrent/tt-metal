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
    TT_FATAL(p.flat.cmb_rt > 0, "flat_with_program: the report args are not set");
    TT_FATAL(!p.extra.mesh_programs.empty(), "flat_with_program: no programs to append");
    TT_FATAL(p.writers > 0, "flat_with_program: no y writers");
    (void)t;
}

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
    for (const auto& [range, desc] : p.extra.mesh_programs) {
        TT_FATAL(range.shape().mesh_size() == 1, "flat_with_program: one appended program per chip");
        auto flat = FlatRoutedExpertProgramFactory::create(p.flat, t, output);
        flat.program.append(desc);
        for (const auto& [k, core] : flat.shared_variables.cmb_writers) {
            auto& a = GetRuntimeArgs(flat.program, k, core);
            a[p.flat.cmb_rt + 0] = static_cast<uint32_t>(cv.x);
            a[p.flat.cmb_rt + 1] = static_cast<uint32_t>(cv.y);
            a[p.flat.cmb_rt + 2] = p.counts_addr;
            a[p.flat.cmb_rt + 3] = 0;  // (no go handshake: SE_CMB_NO_GO)
        }
        if (!p.l1_args.empty()) {
            TT_FATAL(prog_i < p.l1_args.size(), "flat_with_program: l1_args per appended program");
            const auto& l1 = p.l1_args[prog_i];
            TT_FATAL(
                l1.size() <= FLAT_Y_L1_ARGS, "flat_with_program: {} L1 hand-off args > {}", l1.size(), FLAT_Y_L1_ARGS);
            uint32_t widx = 0;
            for (const auto& [k, core] : flat.shared_variables.cmb_writers) {
                auto& a = GetRuntimeArgs(flat.program, k, core);
                TT_FATAL(
                    a.size() >= p.flat.cmb_rt + 4 + FLAT_Y_L1_ARGS,
                    "flat_with_program: writer args too short (MIMO_FL_Y_L1?)");
                for (uint32_t j = 0; j < l1.size(); ++j) {
                    a[p.flat.cmb_rt + 4 + j] = l1[j];
                }
                a[p.flat.cmb_rt + 4 + 4] = widx++;
            }
        }
        ++prog_i;
        if (std::getenv("FLAT_WP_LOG") && range.start_coord() == p.extra.mesh_programs.front().first.start_coord()) {
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
        shared[range] = shared_variables_t{};
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
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
