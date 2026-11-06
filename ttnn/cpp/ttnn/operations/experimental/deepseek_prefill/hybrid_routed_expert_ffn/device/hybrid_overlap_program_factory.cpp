// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_overlap_program_factory.hpp"

#include <map>

#include <tt-metalium/mesh_coord.hpp>
#include <ttnn/global_semaphore.hpp>

#include "hybrid_program_factory.hpp"
#include "hybrid_routed_expert_ffn_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/routed_expert_combine_overlap/device/routed_expert_handoff.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

tt::tt_metal::ProgramDescriptor HybridSoloProgramFactory::create_descriptor(
    const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t, ttnn::Tensor& output) {
    return create_hybrid_program_descriptor(op, t, output);
}

combine::CombineFabric2dParams combine_attributes(
    const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t) {
    return combine::CombineFabric2dParams{
        .device = t.x.device(),
        .experts_per_chip = op.experts_per_chip,
        .num_experts_per_tok = op.num_experts_per_tok,
        .seq_len_per_chip = op.seq_len_per_chip,
        .axis = op.combine_axis,
        .num_links = op.combine_num_links,
        .hybrid_token_threshold = op.hybrid_token_threshold,
        .l1_floor = op.combine_l1_floor,
    };
}

combine::CombineFabric2dInputs combine_inputs(const HybridRoutedExpertFfnInputs& t) {
    return combine::CombineFabric2dInputs{
        .dispatched_buffer = t.output,
        .dispatched_metadata = t.dispatched_metadata.value(),
        .expert_token_counts = t.counts,
        .expert_region_offsets = t.expert_region_offsets.value(),
        .expert_offsets = t.expert_offsets.value(),
        .global_expert_idx_table = t.replicated_global_expert_idx_table,
    };
}

tt::tt_metal::WorkloadDescriptor HybridOverlapProgramFactory::create_workload_descriptor(
    const HybridRoutedExpertFfnParams& op,
    const HybridRoutedExpertFfnInputs& t,
    ttnn::Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    // The arena is the caller's, allocated per call in hybrid_routed_expert_moe and freed after it, so every op that
    // follows gets that L1 back; its address is hashed, so a cached program is only reused over the same layout.
    // Without the fused pass there is none, and neither op needs one: each keeps static circular buffers on its own
    // rows.
    const bool run_fused_pass = op.hybrid_token_threshold > 0;
    tt::tt_metal::Buffer* arena = run_fused_pass ? t.l1_arena->buffer() : nullptr;
    const auto go_addr = static_cast<uint32_t>(op.expert_go->address());

    ttnn::Tensor re_output = t.output;
    const auto re_descriptor = create_hybrid_program_descriptor(op, t, re_output, arena);

    auto combine_args = combine_attributes(op, t);
    combine_args.routed_expert_writers = combine::expert_done_writer_count(re_descriptor);
    combine_args.routed_expert_cores = tt::tt_metal::CoreRange(
        tt::tt_metal::CoreCoord{0, kOriginY}, tt::tt_metal::CoreCoord{kGridX - 1, kOriginY + kGridY - 1});
    TT_FATAL(
        combine_args.routed_expert_writers == combine_args.routed_expert_cores.size(),
        "hybrid routed expert: {} writer cores report to combine, but `go` is multicast to the {}-core rectangle",
        combine_args.routed_expert_writers,
        combine_args.routed_expert_cores.size());
    combine_args.routed_expert_go_addr = go_addr;
    std::map<ttnn::MeshCoordinate, combine::CollectorTarget> collectors;
    auto workload = combine::create_combine_workload(
        combine_args,
        combine_inputs(t),
        output,
        tensor_coords,
        combine::CombineL1{
            .fwd_arrived = &*op.fwd_arrived,
            .final_arrived = &*op.final_arrived,
            .channels = &op.channel_semaphores,
            .arena = arena,
            .forwarding = t.forwarding_buffer.has_value() ? t.forwarding_buffer->buffer() : nullptr},
        &collectors);

    combine::merge_routed_expert(
        workload, collectors, re_descriptor, combine_args, t.expert_offsets->buffer(), go_addr);

    // The caller owns all three semaphores; holding them here as well only keeps them alive with the program.
    // combine_fabric2d already holds fwd_arrived and final_arrived through its semaphore list.
    workload.semaphores.push_back(*op.expert_go);
    return workload;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
