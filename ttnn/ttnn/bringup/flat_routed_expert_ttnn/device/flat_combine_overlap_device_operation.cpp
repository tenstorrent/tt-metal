// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_combine_overlap_device_operation.hpp"

#include <map>
#include <unordered_map>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_workload.hpp>

namespace ttnn::operations::bringup::flat_routed_expert {

namespace cmb = ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine;

void FlatCombineOverlapDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& p, const tensor_args_t& t) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;
    TT_FATAL(p.flat.cmb_rt > 0, "flat_combine_overlap: the report args are not set");
    // bfp8 tiles (combine's untilizers read them) or row-major bf16 rows (combine's readers read them directly)
    TT_FATAL(
        p.flat.y_row_major ? t.y.layout() == Layout::ROW_MAJOR && t.y.dtype() == DataType::BFLOAT16
                           : t.y.layout() == Layout::TILE && t.y.dtype() == DataType::BFLOAT8_B,
        "flat_combine_overlap: y must be bfp8 TILE, or bf16 ROW_MAJOR with y_row_major");
    TT_FATAL(
        p.fwd_arrived && p.final_arrived && p.expert_go,
        "flat_combine_overlap: fwd_arrived, final_arrived and expert_go global semaphores are required");
    TT_FATAL(p.writers > 0, "flat_combine_overlap: no y writers");
}

FlatCombineOverlapFactory::cached_mesh_workload_t FlatCombineOverlapFactory::create_mesh_workload(
    const FlatCombineOverlapParams& p,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const FlatCombineOverlapInputs& t,
    Tensor& output) {
    using namespace tt::tt_metal;
    // Row-major y: combine walks every chip's schedule entries (an expert, or one chunk of a pinned one) in the order
    // the flat expert computes them; its readers rebuild the schedules from the counts with the same defines
    // (MIMO_FL_CMB_SLOT_ORDER=1, probe: whole experts in slot order)
    std::vector<std::string> schedule_defines;
    if (p.flat.y_row_major && !std::getenv("MIMO_FL_CMB_SLOT_ORDER")) {
        for (const auto& [k, v] : flat_schedule_defines(p.flat, make_flat_routed_expert_plan(t.x.device(), p.flat))) {
            schedule_defines.push_back(k + "=" + v);
        }
        // steps = the schedule's entries (a pinned expert chunk by chunk), padded to a fixed count on every chip
        schedule_defines.push_back(
            "CMBF2D_WALK_STEPS=" + std::to_string(flat_combine_overlap_walk_steps(p.flat.experts_per_chip)));
    }
    // combine as the hybrid's overlap builds it, one pass (threshold 0), its L1 in the arena
    const cmb::CombineFabric2dParams cp{
        .device = p.device,
        .experts_per_chip = p.experts_per_chip,
        .num_experts_per_tok = p.num_experts_per_tok,
        .seq_len_per_chip = p.seq_len_per_chip,
        .axis = p.axis,
        .num_links = p.num_links,
        .hybrid_token_threshold = 0,
        .routed_expert_writers = p.writers,
        .routed_expert_cores = p.flat_cores,
        .routed_expert_go_addr = static_cast<uint32_t>(p.expert_go->address()),
        .routed_expert_extra_cores = p.extra_cores,
        .routed_expert_schedule_defines = schedule_defines,
    };
    const cmb::CombineFabric2dInputs ci{
        .dispatched_buffer = t.y,
        .dispatched_metadata = t.dispatched_metadata,
        .expert_token_counts = t.counts,
        .expert_region_offsets = t.regions,
        .expert_offsets = t.expert_offsets,
        .global_expert_idx_table = t.global_expert_idx_table,
    };
    std::map<ttnn::MeshCoordinate, cmb::CollectorTarget> collectors;
    Tensor combine_out = output;
    auto combine = std::make_shared<WorkloadDescriptor>(cmb::create_combine_workload(
        cp,
        ci,
        combine_out,
        tensor_coords,
        cmb::CombineL1{.fwd_arrived = &*p.fwd_arrived, .final_arrived = &*p.final_arrived, .arena = t.arena.buffer()},
        &collectors));

    const FlatRoutedExpertInputs fi{
        .x = t.x,
        .counts = t.counts,
        .regions = t.regions,
        .global_expert_ids = t.global_expert_ids,
        .gate_up_weights = t.gate_up_weights,
        .down_weights = t.down_weights,
        .reader_down_weights = t.reader_down_weights,
        .done_words = t.done_words,
        .arena = t.arena,
        .words = t.words,
        .output = t.y,
        .token_index = std::nullopt};
    Tensor y = t.y;
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    for (auto& prog : combine->programs) {
        TT_FATAL(prog.range.shape().mesh_size() == 1, "flat_combine_overlap: combine builds one program per chip");
        const auto coord = prog.range.start_coord();
        if (std::getenv("FLAT_CMB_LOG") && coord == combine->programs.front().range.start_coord()) {
            for (const auto& k : prog.descriptor.kernels) {
                log_info(tt::LogOp, "flat_combine_overlap: combine kernel {} on {}", k.kernel_source, k.core_ranges.str());
            }
        }
        auto flat = FlatRoutedExpertProgramFactory::create(p.flat, fi, y);
        flat.program.append(prog.descriptor);
        const auto& c = collectors.at(coord);
        for (const auto& [k, core] : flat.shared_variables.cmb_writers) {
            auto& a = GetRuntimeArgs(flat.program, k, core);
            a[p.flat.cmb_rt + 0] = static_cast<uint32_t>(c.worker_virtual.x);
            a[p.flat.cmb_rt + 1] = static_cast<uint32_t>(c.worker_virtual.y);
            a[p.flat.cmb_rt + 2] = c.counts_addr;
            a[p.flat.cmb_rt + 3] = static_cast<uint32_t>(p.expert_go->address());
        }
        TT_FATAL(
            flat.shared_variables.cmb_writers.size() == p.writers,
            "flat_combine_overlap: {} y writers in the program, {} declared to combine",
            flat.shared_variables.cmb_writers.size(),
            p.writers);
        workload.add_program(prog.range, std::move(flat.program));
        shared[prog.range] = shared_variables_t{combine};
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

}  // namespace ttnn::operations::bringup::flat_routed_expert

namespace ttnn::prim::bringup {
ttnn::Tensor flat_combine_overlap(
    const ttnn::operations::bringup::flat_routed_expert::FlatCombineOverlapParams& params,
    const ttnn::operations::bringup::flat_routed_expert::FlatCombineOverlapInputs& inputs) {
    using Op = ttnn::operations::bringup::flat_routed_expert::FlatCombineOverlapDeviceOperation;
    return ttnn::device_operation::launch<Op>(params, inputs);
}
}  // namespace ttnn::prim::bringup
