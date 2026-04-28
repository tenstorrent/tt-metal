// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "shared_expert_ffn.hpp"

#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn {

ttnn::Tensor shared_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    uint32_t cluster_axis,
    uint32_t tp_axis_size,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto gate_out = ttnn::matmul(
        /*input_tensor_a=*/x,
        /*input_tensor_b=*/gate_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::string("silu"),
        /*compute_kernel_config=*/compute_kernel_config);

    auto up_out = ttnn::matmul(
        /*input_tensor_a=*/x,
        /*input_tensor_b=*/up_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config);

    ttnn::multiply_(/*lhs=*/gate_out, /*rhs=*/up_out);
    up_out.deallocate();

    auto full_out = ttnn::matmul(
        /*input_tensor_a=*/gate_out,
        /*input_tensor_b=*/down_proj,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config);

    if (tp_axis_size <= 1) {
        return full_out;
    }

    return ttnn::reduce_scatter(
        /*input_tensor=*/full_out,
        /*dim=*/-1,
        /*cluster_axis=*/cluster_axis,
        /*subdevice_id=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        /*intermediate_memory_config=*/std::nullopt,
        /*optional_output_tensor=*/std::nullopt,
        /*num_links=*/num_links,
        /*topology=*/topology);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn
