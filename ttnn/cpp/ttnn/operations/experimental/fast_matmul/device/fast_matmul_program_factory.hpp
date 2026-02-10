// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fast_matmul_device_operation_types.hpp"
#include "tt-metalium/circular_buffer.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::experimental::prim {

struct FastMatmulProgramFactory {
    struct shared_variables_t {
        std::array<tt::tt_metal::CBHandle, 3> cbs{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FastMatmulParams& operation_attributes, const FastMatmulInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FastMatmulParams& operation_attributes,
        const FastMatmulInputs& tensor_args,
        Tensor& tensor_return_value);
};

FastMatmulProgramFactory::shared_variables_t fast_matmul_factory_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const FastMatmulConfig>& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config);

// Shared implementation for variable number of output tensors (used by both fast_matmul and fast_matmul_split)
// Unlike fast_matmul_factory_helper, this function takes a number of output tensors as an argument (N_chunks) and
// a vector of output tensors.
FastMatmulProgramFactory::shared_variables_t fast_matmul_factory_helper_common(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const FastMatmulConfig>& config,
    const std::vector<Tensor>& output_tensors,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t N_chunks);

// Common helper for override_runtime_arguments - used by both fast_matmul and fast_matmul_split
void override_runtime_arguments_common(
    FastMatmulProgramFactory::cached_program_t& cached_program,
    const Buffer& in0_buffer,
    const Buffer& in1_buffer,
    const Buffer& output_buffer);

}  // namespace ttnn::experimental::prim
