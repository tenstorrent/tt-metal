// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "minimal_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::experimental::prim {

struct MinimalMatmulProgramFactory {
    struct shared_variables_t {
        uint32_t num_cores{};
        std::vector<CoreCoord> cores;
        tt::tt_metal::KernelHandle in0_sender_kernels_id{};
        tt::tt_metal::KernelHandle in0_receiver_kernels_id{};
        tt::tt_metal::KernelHandle in1_sender_kernels_id{};
        tt::tt_metal::KernelHandle in1_receiver_kernels_id{};
        bool transpose_core_grid{};
        bool read_local_slice_from_input{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const MinimalMatmulParams& operation_attributes,
        const MinimalMatmulInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MinimalMatmulParams& operation_attributes,
        const MinimalMatmulInputs& tensor_args,
        Tensor& tensor_return_value);
};

MinimalMatmulProgramFactory::shared_variables_t minimal_matmul_factory_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const MinimalMatmulConfig>& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler>& fused_op_signaler);

}  // namespace ttnn::experimental::prim
