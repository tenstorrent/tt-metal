// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_combine.hpp"

#include <tuple>
#include <utility>

#include "device/unified_routed_expert_combine_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine {

ttnn::Tensor unified_routed_expert_moe_combine(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_offsets,
    const ttnn::Tensor& replicated_global_expert_idx_table,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    const GlobalSemaphore& fwd_arrived_semaphore,
    const GlobalSemaphore& final_arrived_semaphore,
    const GlobalSemaphore& expert_go_semaphore,
    const std::vector<GlobalSemaphore>& channel_semaphores,
    uint32_t combine_axis,
    uint32_t combine_num_links,
    uint32_t big_first_threshold,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    RoutedExpertActivation activation,
    const std::optional<std::vector<ttnn::Tensor>>& gate_biases,
    const std::optional<std::vector<ttnn::Tensor>>& up_biases,
    const std::optional<std::vector<ttnn::Tensor>>& down_biases,
    const std::optional<ttnn::Tensor>& forwarding_buffer) {
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

    // The routed expert's output: the bfloat8_b tiles the row-major path writes, which combine's untilizers read --
    // the same handoff as running the two ops back to back.
    const ttnn::Tensor output = ttnn::empty(
        dispatched_buffer.logical_shape(),
        tt::tt_metal::DataType::BFLOAT8_B,
        tt::tt_metal::Layout::TILE,
        dispatched_buffer.device(),
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});

    std::vector<uint32_t> channel_addrs;
    channel_addrs.reserve(channel_semaphores.size());
    for (const auto& sem : channel_semaphores) {
        channel_addrs.push_back(static_cast<uint32_t>(sem.address()));
    }

    using OperationType = UnifiedRoutedExpertCombineDeviceOperation;
    OperationType::operation_attributes_t attributes{
        .m_tiles = (max_dispatched_tokens_per_expert + 31) / 32,
        .experts_per_chip = experts_per_chip,
        .activation = activation,
        .fuse_bias = has_bias,
        .compute_kernel_config = compute_kernel_config.has_value()
                                     ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                     : std::nullopt,
        .big_first_threshold = big_first_threshold,
        // Combine's hand-placed L1 stays under every buffer already allocated: see combine_l1_floor.
        .combine_l1_floor =
            static_cast<uint32_t>(dispatched_buffer.device()->lowest_occupied_compute_l1_address().value_or(0)),
        .combine_axis = combine_axis,
        .combine_num_links = combine_num_links,
        .num_experts_per_tok = num_experts_per_tok,
        .seq_len_per_chip = seq_len_per_chip,
        .fwd_arrived = fwd_arrived_semaphore,
        .final_arrived = final_arrived_semaphore,
        .expert_go = expert_go_semaphore,
        .channel_semaphores = channel_semaphores,
        .fwd_arrived_addr = static_cast<uint32_t>(fwd_arrived_semaphore.address()),
        .final_arrived_addr = static_cast<uint32_t>(final_arrived_semaphore.address()),
        .expert_go_addr = static_cast<uint32_t>(expert_go_semaphore.address()),
        .channel_semaphore_addrs = std::move(channel_addrs)};
    OperationType::tensor_args_t tensor_args{
        .x = dispatched_buffer,
        .gate_projs = gate_projs,
        .up_projs = up_projs,
        .down_projs = down_projs,
        .counts = expert_token_counts,
        .global_expert_idx_table = global_expert_idx_table,
        .expert_region_offsets = expert_region_offsets,
        .gate_biases = has_bias ? *gate_biases : std::vector<ttnn::Tensor>{},
        .up_biases = has_bias ? *up_biases : std::vector<ttnn::Tensor>{},
        .down_biases = has_bias ? *down_biases : std::vector<ttnn::Tensor>{},
        .output = output,
        .dispatched_metadata = dispatched_metadata,
        .expert_offsets = expert_offsets,
        .replicated_global_expert_idx_table = replicated_global_expert_idx_table,
        .forwarding_buffer = forwarding_buffer};
    // The routed expert sizes its circular buffers from the free L1, so the plan it picks now is part of the
    // program-cache key.
    std::tie(attributes.wave_per_core_M, attributes.wave_in0_block_w) =
        unified_routed_expert_ffn::UnifiedRoutedExpertFfnProgramFactory::select_l1_plan(
            routed_expert_attributes(attributes), routed_expert_inputs(tensor_args));
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine
