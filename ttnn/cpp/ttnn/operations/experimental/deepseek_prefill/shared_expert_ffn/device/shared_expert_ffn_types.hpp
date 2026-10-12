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
    // Mesh axis to partition the output's emb over, as mesh_partition(dim=-1, cluster_axis) would:
    // the device at index i along it computes only emb slice i. nullopt => full-width output.
    std::optional<uint32_t> cluster_axis;

    static constexpr auto attribute_names = std::forward_as_tuple("activation", "core_grid_origin", "cluster_axis");
    auto attribute_values() const { return std::forward_as_tuple(activation, core_grid_origin, cluster_axis); }
};

// x: (1, .., 1, M, emb), every token one row. gate_proj / up_proj: (emb, hidden), down_proj:
// (hidden, emb), all replicated on every device.
struct SharedExpertFfnInputs {
    Tensor x;
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn
