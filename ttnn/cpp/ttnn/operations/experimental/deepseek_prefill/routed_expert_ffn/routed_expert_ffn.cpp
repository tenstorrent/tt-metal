// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn.hpp"

#include "device/gate_up_matmul_device_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& /*gate_program_config*/,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& /*up_program_config*/,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& down_program_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    // activated = silu(x @ gate_proj) * (x @ up_proj)  — fused in compute kernel
    auto activated = ttnn::prim::gate_up_matmul(x, gate_proj, up_proj, compute_kernel_config);

    // output = activated @ down_proj
    return ttnn::matmul(
        activated,
        down_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        down_program_config,
        /*activation=*/std::nullopt,
        compute_kernel_config,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        output);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
