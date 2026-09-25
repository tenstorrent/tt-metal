// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_overlap_program_factory.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <ttnn/global_semaphore.hpp>

#include "hybrid_program_factory.hpp"
#include "hybrid_routed_expert_ffn_device_operation.hpp"
#include "combine/combine_fabric2d_program_factory.hpp"
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
        .wait_for_routed_expert = true,
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

struct OverlapL1 {
    std::optional<tt::tt_metal::GlobalSemaphore> fwd_arrived;
    // Released by combine's collector to the routed expert's writers; see kernels/hybrid_expert_done.hpp.
    std::optional<tt::tt_metal::GlobalSemaphore> go;
    std::shared_ptr<ttnn::Tensor> arena;
};

// The two global semaphores first, then the arena under them. The other order cannot work: an arena over all of L1
// leaves the semaphore nowhere to go, and it cannot be allocated per call like the solo arena because its address is a
// compile-time argument of combine's kernels.
OverlapL1 allocate_overlap_l1(ttnn::MeshDevice* mesh, bool with_arena) {
    const auto grid = mesh->compute_with_storage_grid_size();
    const tt::tt_metal::CoreRangeSet all_workers(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{grid.x - 1, grid.y - 1}));

    OverlapL1 l1;
    l1.fwd_arrived =
        ttnn::global_semaphore::create_global_semaphore(mesh, all_workers, 0, tt::tt_metal::BufferType::L1);
    const tt::tt_metal::CoreRangeSet re_cores(tt::tt_metal::CoreRange(
        tt::tt_metal::CoreCoord{0, kOriginY}, tt::tt_metal::CoreCoord{kGridX - 1, kOriginY + kGridY - 1}));
    l1.go = ttnn::global_semaphore::create_global_semaphore(mesh, re_cores, 0, tt::tt_metal::BufferType::L1);
    if (!with_arena) {
        return l1;
    }

    // Whatever L1 is left, not a fixed share: other cached programs (combine alone among them) may hold
    // their own allocations, and the fused half budgets its blocking against this arena's size.
    const auto free_l1 = mesh->allocator()->get_statistics(tt::tt_metal::BufferType::L1);
    const uint32_t arena_bytes =
        std::min<uint32_t>(hybrid_l1_arena_bytes(mesh), static_cast<uint32_t>(free_l1.largest_free_block_bytes)) &
        ~static_cast<uint32_t>(63);
    const uint32_t cols = arena_bytes / sizeof(uint16_t);
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({static_cast<uint32_t>(grid.x * grid.y), cols}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig{
                tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                tt::tt_metal::BufferType::L1,
                tt::tt_metal::ShardSpec{all_workers, {1, cols}, tt::tt_metal::ShardOrientation::ROW_MAJOR}}));
    l1.arena = std::make_shared<ttnn::Tensor>(create_device_tensor(spec, mesh));
    const auto* arena = l1.arena->buffer();
    TT_FATAL(
        arena->address() + arena->aligned_size_per_bank() <= std::min(l1.fwd_arrived->address(), l1.go->address()),
        "hybrid routed expert: the L1 arena [0x{:x}, +{} B) runs into the global semaphores at 0x{:x} / 0x{:x}",
        arena->address(),
        arena->aligned_size_per_bank(),
        l1.fwd_arrived->address(),
        l1.go->address());
    return l1;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor HybridOverlapProgramFactory::create_workload_descriptor(
    const HybridRoutedExpertFfnParams& op,
    const HybridRoutedExpertFfnInputs& t,
    ttnn::Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    auto* mesh = t.x.device();
    // Without pass A the routed expert has no arena to lay its buffers over, and neither op needs one: each
    // keeps static circular buffers on its own rows, both below fwd_arrived.
    const bool run_fused_pass = op.hybrid_token_threshold > 0;
    const auto l1 = allocate_overlap_l1(mesh, run_fused_pass);
    tt::tt_metal::Buffer* arena = run_fused_pass ? l1.arena->buffer() : nullptr;

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
    combine_args.routed_expert_go_addr = static_cast<uint32_t>(l1.go->address());
    std::map<ttnn::MeshCoordinate, combine::CollectorTarget> collectors;
    auto workload = combine::create_combine_workload(
        combine_args,
        combine_inputs(t),
        output,
        tensor_coords,
        combine::CombineL1{.fwd_arrived = &*l1.fwd_arrived, .arena = arena},
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
                .go_addr = static_cast<uint32_t>(l1.go->address())});
        program.descriptor = tt::tt_metal::merge_program_descriptors({program.descriptor, re});
    }

    // combine_fabric2d already holds fwd_arrived through its semaphore list.
    workload.semaphores.push_back(*l1.go);
    if (l1.arena) {
        workload.buffers.push_back({l1.arena, l1.arena->buffer()});
    }
    return workload;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
