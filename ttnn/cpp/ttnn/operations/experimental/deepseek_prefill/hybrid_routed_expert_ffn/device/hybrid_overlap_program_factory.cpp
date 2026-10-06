// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_overlap_program_factory.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>

#include <tt-metalium/mesh_coord.hpp>
#include <ttnn/global_semaphore.hpp>

#include "hybrid_program_factory.hpp"
#include "hybrid_routed_expert_ffn_device_operation.hpp"
#include "combine/combine_fabric2d_program_factory.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/combine_fabric2d_kernel_interface.hpp"
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
    };
}

combine::CombineFabric2dInputs combine_inputs(const HybridRoutedExpertFfnInputs& t) {
    return combine::CombineFabric2dInputs{
        .dispatched_buffer = t.output,
        .dispatched_metadata = t.dispatched_metadata.value(),
        .expert_token_counts = t.counts,
        .expert_region_offsets = t.expert_region_offsets.value(),
        .expert_offsets = t.expert_offsets.value(),
        .global_expert_idx_table = t.replicated_global_expert_idx_table.value(),
    };
}

namespace {

// The unified reader's view of where combine's walks open: expert_offsets' row for the ring chip diametrically
// opposite this one, which both of combine's walk directions take first. Appended past every other argument.
// The reader is the merged one when the fused pass runs and the unified half's own when it does not (threshold 0),
// as for the writer in expert_done_writer_count; both are the unified reader that consumes these arguments.
void append_far_run(tt::tt_metal::ProgramDescriptor& desc, tt::tt_metal::Buffer* expert_offsets, uint32_t dg_far) {
    auto reader = std::find_if(desc.kernels.begin(), desc.kernels.end(), [](const auto& k) {
        return k.kernel_source.ends_with("hybrid_reader.cpp") ||
               k.kernel_source.ends_with("unified_routed_expert_ffn_reader.cpp");
    });
    TT_FATAL(reader != desc.kernels.end(), "hybrid routed expert: the program carries no reader kernel");
    uint32_t base = 0;
    for (const auto& [core, args] : reader->runtime_args) {
        base = std::max(base, static_cast<uint32_t>(args.size()));
    }
    for (auto& [core, args] : reader->runtime_args) {
        args.resize(base, 0);
        args.push_back(static_cast<uint32_t>(expert_offsets->address()));
        args.push_back(static_cast<uint32_t>(expert_offsets->aligned_page_size()));
        args.push_back(dg_far);
        // The address is re-patched on every program-cache hit: the tensor may live elsewhere next call.
        reader->buffer_bindings.push_back(
            tt::tt_metal::BufferBinding{.core = core, .arg_idx = base, .buffer = expert_offsets});
    }
    reader->defines.emplace_back("URE_FAR_RT_BASE", std::to_string(base));
}

}  // namespace

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
    combine_args.routed_expert_writers = expert_done_writer_count(re_descriptor);
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
        combine::CombineL1{.fwd_arrived = &*op.fwd_arrived, .final_arrived = &*op.final_arrived, .arena = arena},
        &collectors);

    for (auto& program : workload.programs) {
        TT_FATAL(
            program.range.shape().mesh_size() == 1,
            "hybrid routed expert: combine built one program for {} chips; each chip needs its own collector",
            program.range.shape().mesh_size());
        const auto coord = program.range.start_coord();
        const auto& collector = collectors.at(coord);

        auto re = re_descriptor;
        append_expert_done_signal(
            re,
            ExpertDoneSignal{
                .collector_noc_x = static_cast<uint32_t>(collector.worker_virtual.x),
                .collector_noc_y = static_cast<uint32_t>(collector.worker_virtual.y),
                .collector_counts_addr = collector.counts_addr,
                .go_addr = go_addr});
        {
            namespace cf = ttnn::operations::experimental::deepseek_prefill::combine_fabric2d;
            const uint32_t extent = cf::ring_extent(combine_args);
            append_far_run(
                re, t.expert_offsets->buffer(), (cf::my_dg_index(combine_args, coord) + extent / 2) % extent);
        }
        program.descriptor = tt::tt_metal::merge_program_descriptors({program.descriptor, re});
    }

    // The caller owns all three semaphores; holding them here as well only keeps them alive with the program.
    // combine_fabric2d already holds fwd_arrived and final_arrived through its semaphore list.
    workload.semaphores.push_back(*op.expert_go);
    return workload;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
