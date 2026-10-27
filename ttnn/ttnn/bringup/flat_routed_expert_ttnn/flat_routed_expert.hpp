// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

#include "device/flat_routed_expert_plan.hpp"
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include "ttnn/tensor/tensor.hpp"

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
    bool h_bf16 = false,
    bool y_bf16 = false,
    uint32_t math_fidelity = 0,
    bool weights_nk = false,
    uint32_t hidden = 0);

// The flat routed expert with another op's programs appended, one program per chip (device/
// flat_with_program_device_operation.hpp): the flat expert's arguments as for flat_routed_expert, plus the caller's
// arena (persistent, over the flat expert's cores and the other op's cells, >= the plan's arena_tiles * 32 rows per
// core) and y ([rows, H] bf16 ROW_MAJOR, DRAM), the other op's MeshProgramDescriptor (`extra_key` identifies it for
// the program cache), and where the y writers report each local expert: counts[e] (cumulative, +1 per writer and
// call) at `counts_addr` on the logical `collector` core; `writers` is the number of y writers the caller expects.
// Returns y.
ttnn::Tensor flat_routed_expert_with_program(
    const ttnn::Tensor& x,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& regions,
    const ttnn::Tensor& global_expert_ids,
    const ttnn::Tensor& gate_up_weights,
    const ttnn::Tensor& down_weights,
    const std::optional<ttnn::Tensor>& reader_down_weights,
    const ttnn::Tensor& done_words,
    uint32_t intermediate,
    uint32_t max_tokens_per_expert,
    const ttnn::Tensor& arena,
    const ttnn::Tensor& y,
    const tt::tt_metal::experimental::MeshProgramDescriptor& extra,
    uint64_t extra_key,
    const tt::tt_metal::CoreCoord& collector,
    uint32_t counts_addr,
    uint32_t writers,
    uint32_t activation,
    uint32_t pin,
    const std::optional<ttnn::Tensor>& token_index = std::nullopt,
    bool down_fp32 = false,
    bool x_bf16 = false,
    bool h_bf16 = false,
    const std::vector<std::vector<uint32_t>>& l1_args = {},
    const std::optional<ttnn::Tensor>& relay_words = std::nullopt,
    uint32_t math_fidelity = 0,
    bool weights_nk = false);

// The plan for a device / config (cached): what the weight layout and the done words need.
std::shared_ptr<const FlatRoutedExpertPlan> flat_routed_expert_plan(
    tt::tt_metal::IDevice* device, const FlatRoutedExpertConfig& config);

}  // namespace ttnn::operations::bringup::flat_routed_expert

namespace ttnn::experimental::deepseek_prefill::bringup {
using ttnn::operations::bringup::flat_routed_expert::flat_routed_expert;
}  // namespace ttnn::experimental::deepseek_prefill::bringup
