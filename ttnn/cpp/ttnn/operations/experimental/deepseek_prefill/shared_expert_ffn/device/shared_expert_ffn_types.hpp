// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn {

using unified_routed_expert_ffn::RoutedExpertActivation;

struct SharedExpertFfnParams {
    RoutedExpertActivation activation = RoutedExpertActivation::Silu;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;
    // Top-left logical core of the fixed unified_routed_expert_ffn worker rectangle.
    CoreCoord core_grid_origin{0, 0};

    static constexpr auto attribute_names = std::forward_as_tuple("activation", "core_grid_origin");
    auto attribute_values() const { return std::forward_as_tuple(activation, core_grid_origin); }
};

// x: (1, .., 1, M, emb), every token one row. gate_proj / up_proj: (emb, hidden). down_proj:
// (hidden, N), N emb or one mesh column's emb slice -- the output is then that slice.
//
// The unified routed-expert program reads its expert layout from three UINT32 vectors; the shared
// expert is one expert owning all M rows from row 0, so the caller builds them once:
// expert_token_counts = [M], global_expert_idx_table = [0], expert_region_offsets = [0].
struct SharedExpertFfnInputs {
    Tensor x;
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
    Tensor expert_token_counts;
    Tensor global_expert_idx_table;
    Tensor expert_region_offsets;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn
