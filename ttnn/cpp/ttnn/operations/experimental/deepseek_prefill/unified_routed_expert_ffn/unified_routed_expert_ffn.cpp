// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn.hpp"

#include "device/unified_routed_expert_ffn_device_operation.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/creation/creation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

ttnn::Tensor unified_routed_expert_moe(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    RoutedExpertActivation activation,
    const std::optional<std::vector<ttnn::Tensor>>& gate_biases,
    const std::optional<std::vector<ttnn::Tensor>>& up_biases,
    const std::optional<std::vector<ttnn::Tensor>>& down_biases) {
    // Single fused device op across ALL local experts. This builds ONE program
    // whose reader/compute/writer kernels iterate over every local expert.
    TT_FATAL(
        gate_projs.size() == up_projs.size() && gate_projs.size() == down_projs.size(),
        "gate/up/down projection lists must have the same length (got {}, {}, {})",
        gate_projs.size(),
        up_projs.size(),
        down_projs.size());
    const uint32_t experts_per_chip = static_cast<uint32_t>(gate_projs.size());
    TT_FATAL(experts_per_chip > 0, "Need at least one expert per chip");

    // Optional per-expert biases (gpt-oss): all three lists together or none,
    // each the same length as the weight lists (one bias per local expert).
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
    const std::vector<ttnn::Tensor> gate_biases_v = has_bias ? *gate_biases : std::vector<ttnn::Tensor>{};
    const std::vector<ttnn::Tensor> up_biases_v = has_bias ? *up_biases : std::vector<ttnn::Tensor>{};
    const std::vector<ttnn::Tensor> down_biases_v = has_bias ? *down_biases : std::vector<ttnn::Tensor>{};

    // Per-expert M in tiles: every local expert is sized to the same
    // max_dispatched_tokens_per_expert. The program config (including the M-axis
    // chunk size) is built once from this M by the program factory and reused for
    // every expert.
    const uint32_t m_tiles = (max_dispatched_tokens_per_expert + 31) / 32;

    // The input layout selects the output strategy (unchanged from the retired
    // per-expert composite):
    //   * TILE bf8 buffer -> write IN PLACE (output == dispatched_buffer). The
    //     reader reads x before the writer drains cb_out for the same rows, so a
    //     row's write is ordered after its read via the CB chain; chunks cover
    //     disjoint rows and experts touch disjoint regions, so no expert can
    //     disturb another. No allocation, no up-front fill.
    //   * ROW_MAJOR bf16 buffer -> the op tilizes x and packs bf8 internally, so
    //     input and output differ in layout and dtype and cannot alias. One
    //     shared TILE bf8 output is allocated for all experts; each writes its
    //     own region. Left uninitialized (downstream combine reads only written
    //     rows, bounded per expert).
    const bool x_is_row_major = dispatched_buffer.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    const ttnn::Tensor output =
        x_is_row_major ? ttnn::empty(
                             dispatched_buffer.logical_shape(),
                             tt::tt_metal::DataType::BFLOAT8_B,
                             tt::tt_metal::Layout::TILE,
                             dispatched_buffer.device(),
                             tt::tt_metal::MemoryConfig{
                                 tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM})
                       : dispatched_buffer;

    return ttnn::prim::unified_routed_expert_moe(
        dispatched_buffer,
        gate_projs,
        up_projs,
        down_projs,
        expert_token_counts,
        global_expert_idx_table,
        expert_region_offsets,
        output,
        m_tiles,
        experts_per_chip,
        x_is_row_major,
        compute_kernel_config.has_value() ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                          : std::nullopt,
        activation,
        gate_biases_v,
        up_biases_v,
        down_biases_v);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
