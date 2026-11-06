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

// The routed expert on unified_routed_expert_moe's wave-planned program: small experts run at once on slices of the
// grid, each with its own weight stream. dispatched_buffer must be bfloat16 ROW_MAJOR; the routed expert writes
// bfloat8_b tiles, which is what the op returns alone. `big_first_threshold` (0, or at least 96 tokens) runs the
// experts above that many tokens first.
//
// Passing `dispatched_metadata` overlaps combine_fabric2d in the same program: the routed expert runs on rows 2-9,
// combine on rows 0-1 ships each expert's tokens as soon as the routed expert has written them, in the order the
// experts actually finish, and the op returns combine's output. The remaining combine arguments are
// combine_fabric2d's, required in that mode and ignored otherwise; the index table there is the full (groups,
// extent, experts_per_chip) one, replicated on every device.
//
// The overlap also needs global semaphores the caller creates once and passes on every call: combine's fwd_arrived
// and final_arrived receive counts, the routed expert's expert_go start signal, and combine's `channel_semaphores`
// (hybrid_routed_expert_moe_channel_semaphore_count of them). They outlive the program because neighbouring chips
// bump them across launches. `forwarding_buffer` is combine's DRAM staging buffer
// (hybrid_routed_expert_moe_forwarding_buffer); omitted, combine allocates its own at the first call and keeps it for
// the program's lifetime.
ttnn::Tensor unified_routed_expert_moe_combine(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    uint32_t big_first_threshold = 0,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    RoutedExpertActivation activation = RoutedExpertActivation::Silu,
    const std::optional<std::vector<ttnn::Tensor>>& gate_biases = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& up_biases = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& down_biases = std::nullopt,
    const std::optional<ttnn::Tensor>& dispatched_metadata = std::nullopt,
    const std::optional<ttnn::Tensor>& expert_offsets = std::nullopt,
    const std::optional<ttnn::Tensor>& replicated_global_expert_idx_table = std::nullopt,
    uint32_t combine_axis = 0,
    uint32_t combine_num_links = 2,
    uint32_t num_experts_per_tok = 0,
    uint32_t seq_len_per_chip = 0,
    const std::optional<GlobalSemaphore>& fwd_arrived_semaphore = std::nullopt,
    const std::optional<GlobalSemaphore>& final_arrived_semaphore = std::nullopt,
    const std::optional<GlobalSemaphore>& expert_go_semaphore = std::nullopt,
    const std::vector<GlobalSemaphore>& channel_semaphores = {},
    const std::optional<ttnn::Tensor>& forwarding_buffer = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine

namespace ttnn {
using operations::experimental::deepseek_prefill::unified_routed_expert_combine::unified_routed_expert_moe_combine;
}  // namespace ttnn
