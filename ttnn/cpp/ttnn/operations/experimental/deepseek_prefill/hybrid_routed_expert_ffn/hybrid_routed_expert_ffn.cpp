// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_routed_expert_ffn.hpp"

#include "device/hybrid_routed_expert_ffn_device_operation.hpp"
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
    const std::optional<ttnn::Tensor>& dispatched_metadata,
    const std::optional<ttnn::Tensor>& expert_offsets,
    const std::optional<ttnn::Tensor>& replicated_global_expert_idx_table,
    uint32_t combine_axis,
    uint32_t combine_num_links,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    std::optional<tt::tt_metal::DataType> output_dtype) {
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
    const bool overlap_combine = dispatched_metadata.has_value();
    ttnn::Tensor output = dispatched_buffer;
    TT_FATAL(
        !output_dtype.has_value() || x_is_row_major,
        "output_dtype applies only to a ROW_MAJOR dispatched_buffer; a TILE one is written back in its own dtype");
    if (overlap_combine) {
        // Combine reads bfloat16 tiles, and a TILE x is bfloat8_b, so the tilized row-major path is the one
        // that can produce them.
        TT_FATAL(
            x_is_row_major &&
                output_dtype.value_or(tt::tt_metal::DataType::BFLOAT16) == tt::tt_metal::DataType::BFLOAT16,
            "overlapped with combine, dispatched_buffer must be ROW_MAJOR and the output bfloat16 (got {} {})",
            dispatched_buffer.dtype(),
            dispatched_buffer.layout());
        output_dtype = tt::tt_metal::DataType::BFLOAT16;
    }
    if (x_is_row_major) {
        output = ttnn::empty(
            dispatched_buffer.logical_shape(),
            output_dtype.value_or(tt::tt_metal::DataType::BFLOAT8_B),
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

    // The L1 arena both halves' circular buffers are laid over, allocated here rather than
    // inside the op: the program keeps a raw pointer to this buffer and re-reads its address on
    // every program-cache hit, so it must be owned by something that outlives the program. One
    // shard per worker core, which gives every core an arena at one common L1 address.
    // Overlapped with combine, the op allocates its own arena; see HybridOverlapProgramFactory.
    std::optional<ttnn::Tensor> l1_arena;
    if (fused_half_runs && !overlap_combine) {
        auto* device = dispatched_buffer.device();
        const uint32_t arena_bytes = hybrid_l1_arena_bytes(device);
        // Whole bfloat16 elements. hybrid_l1_arena_bytes rounds down to 64B units, so the halving
        // is exact; check it here rather than trust a rule that lives in another file, because an
        // odd byte count would silently drop the tail of every core's shard instead of failing.
        TT_FATAL(
            arena_bytes % 2 == 0, "arena is {} bytes, which is not a whole number of bfloat16 elements", arena_bytes);
        const uint32_t cols = arena_bytes / 2;
        const tt::tt_metal::CoreRangeSet grid(tt::tt_metal::CoreRange(
            tt::tt_metal::CoreCoord{0, kOriginY}, tt::tt_metal::CoreCoord{kGridX - 1, kOriginY + kGridY - 1}));
        l1_arena = ttnn::empty(
            ttnn::Shape({kGridX * kGridY, cols}),
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::Layout::ROW_MAJOR,
            device,
            tt::tt_metal::MemoryConfig{
                tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                tt::tt_metal::BufferType::L1,
                tt::tt_metal::ShardSpec{grid, {1, cols}, tt::tt_metal::ShardOrientation::ROW_MAJOR}});
    }

    using OperationType = HybridRoutedExpertFfnDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
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
            .combine_axis = combine_axis,
            .combine_num_links = combine_num_links,
            .num_experts_per_tok = num_experts_per_tok,
            .seq_len_per_chip = seq_len_per_chip},
        OperationType::tensor_args_t{
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
            .l1_arena = l1_arena,
            .dispatched_metadata = dispatched_metadata,
            .expert_offsets = expert_offsets,
            .replicated_global_expert_idx_table = replicated_global_expert_idx_table});
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
