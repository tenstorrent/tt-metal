// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_matmul.hpp"
#include "device/routed_matmul_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

ttnn::Tensor routed_matmul(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& max_expert_iter,
    uint32_t curr_expert_iter,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<tt::tt_metal::DataType>& output_dtype) {
    return ttnn::prim::routed_matmul(
        a,
        b,
        max_expert_iter,
        curr_expert_iter,
        program_config,
        compute_kernel_config,
        output_memory_config,
        optional_output_tensor,
        output_dtype);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
