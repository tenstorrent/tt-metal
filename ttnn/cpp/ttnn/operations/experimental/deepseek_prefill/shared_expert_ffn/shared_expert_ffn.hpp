// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/sub_device_types.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn {

// Composite shared expert FFN for DeepSeek MoE prefill.
//
// Computes:
//     gate_out  = x @ gate_proj         (with fused SiLU activation)
//     up_out    = x @ up_proj
//     activated = gate_out * up_out     (in-place into gate_out)
//     full_out  = activated @ down_proj
//     output    = reduce_scatter(full_out, dim=-1, cluster_axis=cluster_axis)
//
// When tp_axis_size == 1 the reduce_scatter step is skipped and full_out
// is returned directly.
ttnn::Tensor shared_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    uint32_t cluster_axis,
    uint32_t tp_axis_size,
    uint32_t num_links = 1,
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn

namespace ttnn {
using operations::experimental::deepseek_prefill::shared_expert_ffn::shared_expert_ffn;
}  // namespace ttnn
