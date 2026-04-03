// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn.hpp"

#include <array>

#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& gate_program_config,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& up_program_config,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& down_program_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output) {
    // gate_out = x @ gate_proj
    auto gate_out = ttnn::matmul(
        x,
        gate_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        gate_program_config,
        /*activation=*/std::nullopt,
        compute_kernel_config);

    // up_out = x @ up_proj
    auto up_out = ttnn::matmul(
        x,
        up_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        up_program_config,
        /*activation=*/std::nullopt,
        compute_kernel_config);

    // activated = silu(gate_out) * up_out
    const std::array<unary::EltwiseUnaryWithParam, 1> lhs_acts{unary::EltwiseUnaryWithParam{unary::UnaryOpType::SILU}};
    auto activated = ttnn::multiply(
        gate_out,
        up_out,
        /*output_dtype=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        /*output=*/std::nullopt,
        /*post_activations=*/{},
        /*lhs_activations=*/ttsl::Span<const unary::EltwiseUnaryWithParam>(lhs_acts.data(), lhs_acts.size()),
        /*rhs_activations=*/{});

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
