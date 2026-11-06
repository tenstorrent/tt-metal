// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_combine_program_factory.hpp"

#include <limits>
#include <map>

#include <tt-metalium/mesh_coord.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/combine_fabric2d_kernel_interface.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/routed_expert_combine_overlap/device/routed_expert_handoff.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine {

namespace ure = ::ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn;
namespace overlap = ::ttnn::operations::experimental::deepseek_prefill::routed_expert_combine_overlap;

ure::UnifiedRoutedExpertFfnParams routed_expert_attributes(const UnifiedRoutedExpertCombineParams& op) {
    // Every expert runs here, so the band is wide open, as unified_routed_expert_moe alone builds it.
    return ure::UnifiedRoutedExpertFfnParams{
        .m_tiles = op.m_tiles,
        .experts_per_chip = op.experts_per_chip,
        .x_is_row_major = true,
        .activation = op.activation,
        .fuse_bias = op.fuse_bias,
        .compute_kernel_config = op.compute_kernel_config,
        .min_active_tokens = 0,
        .max_active_tokens = std::numeric_limits<uint32_t>::max(),
    };
}

ure::UnifiedRoutedExpertFfnInputs routed_expert_inputs(const UnifiedRoutedExpertCombineInputs& t) {
    return ure::UnifiedRoutedExpertFfnInputs{
        .x = t.x,
        .gate_projs = t.gate_projs,
        .up_projs = t.up_projs,
        .down_projs = t.down_projs,
        .counts = t.counts,
        .global_expert_idx_table = t.global_expert_idx_table,
        .output = t.output,
        .expert_region_offsets = t.expert_region_offsets,
        .gate_biases = t.gate_biases,
        .up_biases = t.up_biases,
        .down_biases = t.down_biases,
    };
}

overlap::CombineFabric2dParams combine_attributes(
    const UnifiedRoutedExpertCombineParams& op, const UnifiedRoutedExpertCombineInputs& t) {
    return overlap::CombineFabric2dParams{
        .device = t.x.device(),
        .experts_per_chip = op.experts_per_chip,
        .num_experts_per_tok = op.num_experts_per_tok,
        .seq_len_per_chip = op.seq_len_per_chip,
        .axis = op.combine_axis,
        .num_links = op.combine_num_links,
        .l1_floor = op.combine_l1_floor,
    };
}

overlap::CombineFabric2dInputs combine_inputs(const UnifiedRoutedExpertCombineInputs& t) {
    return overlap::CombineFabric2dInputs{
        .dispatched_buffer = t.output,
        .dispatched_metadata = t.dispatched_metadata,
        .expert_token_counts = t.counts,
        .expert_region_offsets = t.expert_region_offsets,
        .expert_offsets = t.expert_offsets,
        .global_expert_idx_table = t.replicated_global_expert_idx_table,
    };
}

tt::tt_metal::WorkloadDescriptor UnifiedRoutedExpertCombineProgramFactory::create_workload_descriptor(
    const UnifiedRoutedExpertCombineParams& op,
    const UnifiedRoutedExpertCombineInputs& t,
    ttnn::Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    namespace cf = ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d;
    const auto go_addr = static_cast<uint32_t>(op.expert_go->address());
    const auto re_args = routed_expert_attributes(op);
    const auto re_inputs = routed_expert_inputs(t);

    // The routed expert sizes its circular buffers from the free L1, and combine predicts its finish order from the
    // plan that gives; both were fixed when the op was called.
    const auto plan = ure::UnifiedRoutedExpertFfnProgramFactory::select_l1_plan(re_args, re_inputs);
    TT_FATAL(
        plan.first == op.wave_per_core_M && plan.second == op.wave_in0_block_w,
        "unified_routed_expert_moe_combine was called with L1 plan ({}, {}) but would now build ({}, {}); the free L1 "
        "changed between the call and the program build",
        op.wave_per_core_M,
        op.wave_in0_block_w,
        plan.first,
        plan.second);
    ttnn::Tensor re_output = t.output;
    const auto re_descriptor = ure::UnifiedRoutedExpertFfnProgramFactory::create_embedded_descriptor(
        re_args,
        re_inputs,
        re_output,
        ure::UnifiedRoutedExpertFfnProgramFactory::EmbeddedOptions{
            .origin_y = overlap::kRoutedExpertOriginY,
            .big_first_threshold = op.big_first_threshold,
            .report_predicted_order = true,
            .far_chunk_table = true,
        });

    auto combine_args = combine_attributes(op, t);
    // The writers report each expert at its place in the predicted finish order; combine predicts the same order for
    // every ring chip, from the same planner configuration and threshold.
    const auto planner = ure::UnifiedRoutedExpertFfnProgramFactory::wave_planner_config(re_args, re_inputs);
    combine_args.predicted_walk = true;
    combine_args.walk_per_core_M_max = planner.per_core_M_max;
    combine_args.walk_partial_eighth = planner.partial_eighth;
    combine_args.walk_taper_tail = planner.taper_tail;
    combine_args.walk_big_first_threshold = op.big_first_threshold;
    combine_args.routed_expert_writers = overlap::expert_done_writer_count(re_descriptor);
    combine_args.routed_expert_cores = tt::tt_metal::CoreRange(
        tt::tt_metal::CoreCoord{0, overlap::kRoutedExpertOriginY},
        tt::tt_metal::CoreCoord{ure::kCoreGridX - 1, overlap::kRoutedExpertOriginY + ure::kCoreGridY - 1});
    TT_FATAL(
        combine_args.routed_expert_writers == combine_args.routed_expert_cores.size(),
        "unified_routed_expert_moe_combine: {} writer cores report to combine, but `go` is multicast to the {}-core "
        "rectangle",
        combine_args.routed_expert_writers,
        combine_args.routed_expert_cores.size());
    combine_args.routed_expert_go_addr = go_addr;
    TT_FATAL(
        op.experts_per_chip <= 32 && cf::ring_extent(combine_args) <= 16,
        "unified_routed_expert_moe_combine predicts at most 32 experts per chip on a ring of at most 16 chips (got {} "
        "experts, ring of {})",
        op.experts_per_chip,
        cf::ring_extent(combine_args));

    // No L1 arena: each op keeps static circular buffers on its own rows, and combine's hand-placed L1 stays under
    // combine_l1_floor.
    std::map<ttnn::MeshCoordinate, overlap::CollectorTarget> collectors;
    auto workload = overlap::create_combine_workload(
        combine_args,
        combine_inputs(t),
        output,
        tensor_coords,
        overlap::CombineL1{
            .fwd_arrived = &*op.fwd_arrived,
            .final_arrived = &*op.final_arrived,
            .channels = &op.channel_semaphores,
            .arena = nullptr,
            .forwarding = t.forwarding_buffer.has_value() ? t.forwarding_buffer->buffer() : nullptr},
        &collectors);
    overlap::merge_routed_expert(workload, collectors, re_descriptor, combine_args, t.expert_offsets.buffer(), go_addr);

    // The caller owns all three semaphores; holding them here as well only keeps them alive with the program.
    // combine already holds fwd_arrived and final_arrived through its semaphore list.
    workload.semaphores.push_back(*op.expert_go);
    return workload;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine
