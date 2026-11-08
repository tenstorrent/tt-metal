// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.flat_routed_expert: every local expert's GLU FFN of a chip in one program that streams
// each expert's weights once through a spatial pipeline (Blackhole). The composable form for another op's program is
// device/flat_program.hpp.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "device/flat_routed_expert_device_operation_types.hpp"
#include "device/flat_routed_expert_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe_block {
class MoEBlockResources;
}

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

// The flat expert's config from its tensors (what the plan and the program are built for): hidden = x width x
// x_pages_per_row, E = the expert-id table's length, NG = the counts' length, the capacity rounded up to 32 rows,
// bfp8 weights when gate_up_weights is bfloat8_b (else bfp4).
FlatRoutedExpertConfig make_flat_routed_expert_config(
    uint32_t hidden,
    uint32_t intermediate,
    uint32_t experts_per_chip,
    uint32_t num_global_experts,
    uint32_t max_tokens_per_expert,
    bool weights_bf8,
    uint32_t pin,
    const FlatLayoutOptions& layout);

// The down coordinators' done words for a plan: a zeroed persistent L1 (or L1_SMALL) tensor on the plan's `coords`
// cores, a word a down core (the op leaves them zero).
ttnn::Tensor create_flat_routed_expert_done_words(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const FlatRoutedExpertPlan& plan,
    tt::tt_metal::BufferType buffer_type = tt::tt_metal::BufferType::L1);

// Flat spatially pipelined routed experts. Token counts and regions are read on device, so one cached program serves
// every routing. Weights come in the plan's per-core bank layout (flat_routed_expert_weights.py); done_words from
// create_flat_routed_expert_done_words. Returns y [rows, H] (bfp8 TILE; bf16 TILE with y_bf16; bf16 ROW_MAJOR with y_row_major), written at the
// active experts' rows only.
ttnn::Tensor flat_routed_expert(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& global_expert_ids,
    const ttnn::Tensor& gate_up_weights,
    const ttnn::Tensor& down_weights,
    const ttnn::Tensor& done_words,
    uint32_t intermediate,
    uint32_t max_tokens_per_expert,
    uint32_t activation = 0,
    uint32_t pin = 1,
    const std::optional<ttnn::Tensor>& token_index = std::nullopt,
    uint32_t x_pages_per_row = 1,
    bool y_row_major = false,
    bool down_fp32 = false,
    bool pack_stochastic_rounding = false,
    bool y_bf16 = false,
    uint32_t math_fidelity = 0,
    const FlatLayoutOptions& layout = {},
    const std::optional<ttnn::Tensor>& output = std::nullopt);

// What a caller composing the flat expert into its own program (flat_program.hpp) takes from the MoE block's
// resources: the plan and the persistent done words and relay words (holding the x-ready word).
struct FlatRoutedExpertResources {
    std::shared_ptr<const FlatRoutedExpertPlan> plan;
    ttnn::Tensor done_words;
    ttnn::Tensor relay_words;
};
// Get-or-create the flat expert's slots in `resources` ("flat_routed_expert/done", "flat_routed_expert/relay_words")
// for `config`.
FlatRoutedExpertResources ensure_flat_routed_expert_resources(
    MoEBlockResources& resources, const FlatRoutedExpertConfig& config);

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
