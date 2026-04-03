// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& gate_program_config = std::nullopt,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& up_program_config = std::nullopt,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& down_program_config = std::nullopt,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<ttnn::Tensor> output = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn

namespace ttnn {
using operations::experimental::deepseek_prefill::routed_expert_ffn::routed_expert_ffn;
}  // namespace ttnn
