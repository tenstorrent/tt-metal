// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "device/unified_routed_expert_combine_types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine {

// unified_routed_expert_moe and combine_fabric2d in ONE program: every routed expert runs on the wave-planned
// unified program on rows 2-9, and combine on rows 0-1 ships each expert's tokens as soon as the routed expert has
// written them, in the order the experts actually finish. Returns combine's output, as combine_fabric2d does.
//
// The routed expert's arguments are unified_routed_expert_moe's; dispatched_buffer must be bfloat16 ROW_MAJOR, and
// the routed expert hands combine bfloat8_b tiles, as the two ops do back to back. `big_first_threshold` (0, or at
// least 96 tokens) runs the experts above that many tokens first, so combine can start on the big ones early.
//
// Combine's arguments are combine_fabric2d's, with the index table the full (groups, extent, experts_per_chip) one,
// replicated on every device. The overlap also needs global semaphores the caller creates once and passes on every
// call: combine's fwd_arrived and final_arrived receive counts, the routed expert's expert_go start signal, and
// combine's `channel_semaphores` (hybrid_routed_expert_moe_channel_semaphore_count of them, the same as the hybrid
// overlap). They outlive the program because neighbouring chips bump them across launches. `forwarding_buffer` is
// combine's DRAM staging buffer (hybrid_routed_expert_moe_forwarding_buffer); omitted, combine allocates its own at
// the first call and keeps it for the program's lifetime.
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
    uint32_t combine_axis = 0,
    uint32_t combine_num_links = 2,
    uint32_t big_first_threshold = 0,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    RoutedExpertActivation activation = RoutedExpertActivation::Silu,
    const std::optional<std::vector<ttnn::Tensor>>& gate_biases = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& up_biases = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& down_biases = std::nullopt,
    const std::optional<ttnn::Tensor>& forwarding_buffer = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine

namespace ttnn {
using operations::experimental::deepseek_prefill::unified_routed_expert_combine::unified_routed_expert_moe_combine;
}  // namespace ttnn
