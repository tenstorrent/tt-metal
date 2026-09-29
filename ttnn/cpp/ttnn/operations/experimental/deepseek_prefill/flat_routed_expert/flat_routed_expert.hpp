// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

#include "device/flat_routed_expert_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert {

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
    bool y_row_major = false);

// The plan for a device / config (cached): what the weight layout and the done words need.
std::shared_ptr<const FlatRoutedExpertPlan> flat_routed_expert_plan(
    tt::tt_metal::IDevice* device, const FlatRoutedExpertConfig& config);

}  // namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert

namespace ttnn::experimental::deepseek_prefill {
using ttnn::operations::experimental::deepseek_prefill::flat_routed_expert::flat_routed_expert;
}  // namespace ttnn::experimental::deepseek_prefill
