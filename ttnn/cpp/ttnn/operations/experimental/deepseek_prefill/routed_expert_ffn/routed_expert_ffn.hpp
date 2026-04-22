// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

ttnn::Tensor routed_expert_ffn(
    uint32_t expert_iter,
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<ttnn::Tensor> output = std::nullopt,
    // Optional max_iter tensor (uint32 DRAM scalar tile). When provided, each of
    // the 3 BH matmuls dispatches via the forked routed_matmul device op that
    // threads max_iter/expert_iter through to the reader/compute kernel guard.
    // When absent, falls back to ttnn::matmul — identical to pre-routed behavior.
    const std::optional<ttnn::Tensor>& max_iter = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn

namespace ttnn {
using operations::experimental::deepseek_prefill::routed_expert_ffn::routed_expert_ffn;
}  // namespace ttnn
