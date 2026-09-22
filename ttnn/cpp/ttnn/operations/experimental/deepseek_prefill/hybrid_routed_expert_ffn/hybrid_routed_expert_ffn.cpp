// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_routed_expert_ffn.hpp"

#include "device/hybrid_routed_expert_ffn_device_operation.hpp"
#include "device/hybrid_program_factory.hpp"
#include "device/combine_fabric2d_placement.hpp"
#include <tt-metalium/hal.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/creation/creation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

ttnn::Tensor hybrid_routed_expert_moe(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    uint32_t hybrid_token_threshold,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    RoutedExpertActivation activation,
    const std::optional<std::vector<ttnn::Tensor>>& gate_biases,
    const std::optional<std::vector<ttnn::Tensor>>& up_biases,
    const std::optional<std::vector<ttnn::Tensor>>& down_biases,
    bool overlap_combine,
    const std::optional<ttnn::Tensor>& dispatched_metadata,
    const std::optional<ttnn::Tensor>& expert_offsets,
    const std::optional<ttnn::Tensor>& combine_output,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    uint32_t cluster_axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology) {
    // The combine half is off by default, so its tensors are optional here and unconditional in the
    // attributes: everything past this point reads them as plain tensors, and only this check stands
    // between that and a default-constructed one.
    TT_FATAL(
        !overlap_combine ||
            (dispatched_metadata.has_value() && expert_offsets.has_value() && combine_output.has_value()),
        "overlap_combine needs combine's tensors: dispatched_metadata, expert_offsets and combine_output");

    TT_FATAL(
        gate_projs.size() == up_projs.size() && gate_projs.size() == down_projs.size(),
        "gate/up/down projection lists must have the same length (got {}, {}, {})",
        gate_projs.size(),
        up_projs.size(),
        down_projs.size());
    const uint32_t experts_per_chip = static_cast<uint32_t>(gate_projs.size());
    TT_FATAL(experts_per_chip > 0, "Need at least one expert per chip");

    const int bias_lists = static_cast<int>(gate_biases.has_value()) + static_cast<int>(up_biases.has_value()) +
                           static_cast<int>(down_biases.has_value());
    TT_FATAL(
        bias_lists == 0 || bias_lists == 3,
        "gate/up/down bias lists must all be provided together or all omitted (got {} of 3)",
        bias_lists);
    const bool has_bias = bias_lists == 3;
    if (has_bias) {
        TT_FATAL(
            gate_biases->size() == experts_per_chip && up_biases->size() == experts_per_chip &&
                down_biases->size() == experts_per_chip,
            "bias lists must have one entry per local expert ({}), got ({}, {}, {})",
            experts_per_chip,
            gate_biases->size(),
            up_biases->size(),
            down_biases->size());
    }

    // A count can never exceed the expert's own region, so a threshold at or above the allocated
    // rows would leave the unified half an empty band -- the merged op would read the counts and
    // skip every expert. Reject it rather than silently compute nothing.
    TT_FATAL(
        hybrid_token_threshold < max_dispatched_tokens_per_expert,
        "hybrid_token_threshold ({}) must be below max_dispatched_tokens_per_expert ({}), otherwise no expert "
        "reaches the unified half",
        hybrid_token_threshold,
        max_dispatched_tokens_per_expert);

    const uint32_t m_tiles = (max_dispatched_tokens_per_expert + 31) / 32;

    // Output strategy, and the one place the two halves disagree about aliasing.
    //
    // The fused half refuses to write into its own activations -- its readers can overlap the
    // output writeback -- so whenever it runs, output must be a buffer distinct from x and both
    // halves read the pristine x. Aliasing output onto x, which is what the unified half's own
    // entry point does on the TILE path, is that half's convention and not a kernel requirement:
    // x and out are separate accessors. Keeping them separate is what makes this ONE dispatch on
    // every path -- seeding a distinct output with a copy of x would take a second program.
    const bool x_is_row_major = dispatched_buffer.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    const bool fused_half_runs = hybrid_token_threshold > 0;

    // The combine half moves whole tokens from this output into its own, and sizes its ring slot,
    // fabric payload and output pages off one token size, so the two have to agree on bytes per
    // element. Its output is bf16, so the overlap costs this op a bf16 output where it would
    // otherwise emit bf8 -- twice the write bytes here and twice the read bytes in combine's
    // untilizer. That is the price of the merge, not an oversight.
    //
    // It also pins x to ROW_MAJOR. A tiled x is bf8, and the unified half only tolerates an
    // output dtype different from x's on the row-major path, so a bf16 output over a tiled x
    // would be rejected there instead -- with a message about dtypes rather than about overlap.
    // Both halves have to be carried for the overlap: the shared arena is only laid under the
    // routed expert's circular buffers by the merge, and with no threshold the unified half keeps
    // statically placed ones that collide with the arena combine binds into. Rejected here rather
    // than left to surface as a circular-buffer clash naming addresses and no cause.
    TT_FATAL(
        !overlap_combine || hybrid_token_threshold > 0,
        "overlap_combine needs hybrid_token_threshold > 0: the routed expert's circular buffers only move "
        "into the shared L1 arena when both halves are merged, and combine's untilizer needs that arena");

    TT_FATAL(
        !overlap_combine || x_is_row_major,
        "overlap_combine needs a ROW_MAJOR dispatched_buffer: the combine half requires a bf16 output, which "
        "this op only produces on the row-major path");

    ttnn::Tensor output = dispatched_buffer;
    if (x_is_row_major) {
        output = ttnn::empty(
            dispatched_buffer.logical_shape(),
            overlap_combine ? tt::tt_metal::DataType::BFLOAT16 : tt::tt_metal::DataType::BFLOAT8_B,
            tt::tt_metal::Layout::TILE,
            dispatched_buffer.device(),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
    } else if (fused_half_runs) {
        output = ttnn::empty(
            dispatched_buffer.logical_shape(),
            dispatched_buffer.dtype(),
            dispatched_buffer.layout(),
            dispatched_buffer.device(),
            dispatched_buffer.memory_config());
    }

    using OperationType = HybridRoutedExpertFfnDeviceOperation;
    OperationType::operation_attributes_t attributes{
        .m_tiles = m_tiles,
        .experts_per_chip = experts_per_chip,
        .x_is_row_major = x_is_row_major,
        .activation = activation,
        .fuse_bias = has_bias,
        .compute_kernel_config = compute_kernel_config.has_value()
                                     ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                     : std::nullopt,
        .hybrid_token_threshold = hybrid_token_threshold,
        .overlap_combine = overlap_combine,
        .num_experts_per_tok = num_experts_per_tok,
        .seq_len_per_chip = seq_len_per_chip,
        .cluster_axis = cluster_axis,
        .num_links = num_links,
        .topology = topology};

    OperationType::tensor_args_t tensors{
        .x = dispatched_buffer,
        .gate_projs = gate_projs,
        .up_projs = up_projs,
        .down_projs = down_projs,
        .counts = expert_token_counts,
        .global_expert_idx_table = global_expert_idx_table,
        .output = output,
        .expert_region_offsets = expert_region_offsets,
        .gate_biases = has_bias ? *gate_biases : std::vector<ttnn::Tensor>{},
        .up_biases = has_bias ? *up_biases : std::vector<ttnn::Tensor>{},
        .down_biases = has_bias ? *down_biases : std::vector<ttnn::Tensor>{},
        .l1_arena = std::nullopt,
        .dispatched_metadata = dispatched_metadata,
        .expert_offsets = expert_offsets,
        .combine_output = combine_output};

    // The L1 arena both halves' circular buffers are laid over, allocated here rather than inside
    // the op: the program keeps a raw pointer to this buffer and re-reads its address on every
    // program-cache hit, so it must be owned by something that outlives the program. One shard per
    // core, which gives every core an arena at one common L1 address.
    //
    // Overlapping widens it to combine's rows as well, because combine's untilizer binds its
    // circular buffers into the same arena. That is what keeps the merged program to ONE L1
    // allocation: the allocator hands out addresses device-wide, so two halves each claiming the
    // base on their own rows collide in its bookkeeping even though they never touch.
    if (fused_half_runs || overlap_combine) {
        auto* device = dispatched_buffer.device();

        // Only combine's ring semaphores stay outside the arena -- everything else it places is
        // bound into it. One 64-byte block each, the coarsest alignment the L1 allocator applies,
        // at the MAX untilizer count so the reservation does not move with
        // CMBF2D_UNTILIZERS_PER_GROUP.
        constexpr uint32_t kSemaphoreBlock = 64;
        const uint32_t floor =
            overlap_combine ? (3 + combine::MAX_UNTILIZERS_PER_GROUP + num_links) * kSemaphoreBlock : 0;
        const uint32_t arena_bytes = hybrid_l1_arena_bytes(device) - floor;
        const uint32_t cols = arena_bytes / 2;  // bfloat16 elements

        // Overlapping spans the WHOLE compute grid, not just the routed expert's rectangle:
        // combine's cores are chosen by its own placement, from wherever sits closest to each
        // ethernet core, so nothing here can assume they fall inside kGridX by kGridY.
        const auto compute_grid = device->compute_with_storage_grid_size();
        const uint32_t grid_w = overlap_combine ? compute_grid.x : kGridX;
        const uint32_t origin_y = overlap_combine ? 0 : kOriginY;
        const uint32_t last_y = overlap_combine ? compute_grid.y - 1 : kOriginY + kGridY - 1;
        const uint32_t rows = last_y - origin_y + 1;
        const tt::tt_metal::CoreRangeSet grid(
            tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord{0, origin_y}, tt::tt_metal::CoreCoord{grid_w - 1, last_y}));

        tensors.l1_arena = ttnn::empty(
            ttnn::Shape({grid_w * rows, cols}),
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::Layout::ROW_MAJOR,
            device,
            tt::tt_metal::MemoryConfig{
                tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                tt::tt_metal::BufferType::L1,
                tt::tt_metal::ShardSpec{grid, {1, cols}, tt::tt_metal::ShardOrientation::ROW_MAJOR}});
    }

    return ttnn::device_operation::launch<OperationType>(std::move(attributes), std::move(tensors));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
