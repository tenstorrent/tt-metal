// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

#include "device/flat_routed_expert_plan.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::operations::bringup::flat_routed_expert {

// Flat spatially pipelined routed experts (Blackhole): every local expert's SwiGLU FFN in one program that streams
// each expert's weights once through a spatial pipeline (DRAM bank readers -> gate/up cores, x relays that read /
// tilize / multicast the row-major dispatch buffer, down cores writing bfp8 y at each expert's region). Token counts
// and regions are read on device, so one cached program serves every routing. Weights come in the plan's
// per-core bank layout (flat_routed_expert_plan + the Python weight preparation); done_words is a small persistent
// zeroed L1 tensor on the plan's coordinator cores. Returns y [rows, H] bfp8 TILE (only active experts' rows).
ttnn::Tensor flat_routed_expert(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& global_expert_ids,
    const ttnn::Tensor& gate_up_weights,
    const ttnn::Tensor& down_weights,
    const std::optional<ttnn::Tensor>& reader_down_weights,
    const ttnn::Tensor& done_words,
    uint32_t intermediate,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    uint32_t pin,
    const std::optional<ttnn::Tensor>& token_index = std::nullopt,
    uint32_t x_pages_per_row = 1,
    bool y_row_major = false,
    bool down_fp32 = false,
    bool pack_stochastic_rounding = false,
    bool x_bf16 = false,
    bool h_bf16 = false);

// The flat routed expert overlapped with combine_fabric2d in one program per chip (initial version; see
// device/flat_combine_overlap_device_operation.hpp): the flat expert's arguments as for flat_routed_expert, combine's
// as for hybrid_routed_expert_moe's overlap (the full replicated expert index table, the three global semaphores the
// caller keeps across calls). The flat expert must be planned below combine's rows: row 0 with row-major y (the
// default, MIMO_FL_ROWS=1,9), rows 0-1 with bfp8 y tiles (MIMO_FL_ROWS=2,9). Returns combine's output
// [1, 1, seq_len_per_chip, num_experts_per_tok, H] bf16 ROW_MAJOR; `y_out` (optional) receives the flat expert's y.
ttnn::Tensor flat_combine_overlap(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& global_expert_ids,
    const ttnn::Tensor& gate_up_weights,
    const ttnn::Tensor& down_weights,
    const std::optional<ttnn::Tensor>& reader_down_weights,
    const ttnn::Tensor& done_words,
    uint32_t intermediate,
    uint32_t max_tokens_per_expert,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& expert_offsets,
    const ttnn::Tensor& replicated_global_expert_idx_table,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    uint32_t combine_axis,
    uint32_t combine_num_links,
    const tt::tt_metal::GlobalSemaphore& fwd_arrived_semaphore,
    const tt::tt_metal::GlobalSemaphore& final_arrived_semaphore,
    const tt::tt_metal::GlobalSemaphore& expert_go_semaphore,
    uint32_t activation,
    uint32_t pin,
    bool down_fp32 = false,
    bool x_bf16 = false,
    bool h_bf16 = false,
    bool y_row_major = true,
    const std::optional<ttnn::Tensor>& y_out = std::nullopt);

// The plan for a device / config (cached): what the weight layout and the done words need.
std::shared_ptr<const FlatRoutedExpertPlan> flat_routed_expert_plan(
    tt::tt_metal::IDevice* device, const FlatRoutedExpertConfig& config);

}  // namespace ttnn::operations::bringup::flat_routed_expert

namespace ttnn::experimental::deepseek_prefill::bringup {
using ttnn::operations::bringup::flat_routed_expert::flat_routed_expert;
}  // namespace ttnn::experimental::deepseek_prefill::bringup
