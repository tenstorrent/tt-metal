// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "parallel_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "parallel.hpp"
#include "device/parallel_device_operation_types.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::parallel::detail {

using namespace ttnn::operations::normalization;
using RMSNormOp = layer_norm::LayerNormDeviceOperation;

//=============================================================================
// Branch factory functions
//=============================================================================

std::shared_ptr<BranchDescriptor> make_rms_norm_branch(
    const CoreRangeSet& cores,
    const Tensor& input,
    float epsilon,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& residual_input,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    RMSNormOp::operation_attributes_t attrs{
        .norm_type = LayerNormType::RMSNORM,
        .distributed_norm_stage = DistributedLayerNormStage::NOT_DISTRIBUTED,
        .eps = epsilon,
        .output_mem_config = memory_config.value_or(MemoryConfig{}),
        .program_config = LayerNormDefaultProgramConfig{},
        .compute_kernel_config = compute_kernel_config.value_or(DeviceComputeKernelConfig{})};

    RMSNormOp::tensor_args_t tensor_args{
        .input = input, .residual_input_tensor = residual_input, .weight = weight, .bias = bias};

    return std::make_shared<TypedBranchDescriptor<RMSNormOp>>(cores, attrs, tensor_args);
}

std::shared_ptr<BranchDescriptor> make_layer_norm_branch(
    const CoreRangeSet& cores,
    const Tensor& input,
    float epsilon,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& residual_input,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    RMSNormOp::operation_attributes_t attrs{
        .norm_type = LayerNormType::LAYERNORM,
        .distributed_norm_stage = DistributedLayerNormStage::NOT_DISTRIBUTED,
        .eps = epsilon,
        .output_mem_config = memory_config.value_or(MemoryConfig{}),
        .program_config = LayerNormDefaultProgramConfig{},
        .compute_kernel_config = compute_kernel_config.value_or(DeviceComputeKernelConfig{})};

    RMSNormOp::tensor_args_t tensor_args{
        .input = input, .residual_input_tensor = residual_input, .weight = weight, .bias = bias};

    return std::make_shared<TypedBranchDescriptor<RMSNormOp>>(cores, attrs, tensor_args);
}

//=============================================================================
// Binding implementation
//=============================================================================

void bind_parallel_operation(nb::module_& mod) {
    // Bind BranchDescriptor as an opaque type
    // Note: nanobind handles std::shared_ptr automatically via stl caster
    nb::class_<BranchDescriptor>(
        mod,
        "BranchDescriptor",
        "A descriptor for a parallel branch operation. Created via factory functions like rms_norm_branch().");

    // Factory function for RMS norm branch
    mod.def(
        "rms_norm_branch",
        &make_rms_norm_branch,
        nb::arg("cores"),
        nb::arg("input"),
        nb::arg("epsilon") = 1e-5f,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("residual_input") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        R"doc(
            Create a parallel branch for RMS normalization.

            Args:
                cores (CoreRangeSet): The cores this branch should execute on (must be disjoint from other branches).
                input (Tensor): Input tensor to normalize.
                epsilon (float): Small constant for numerical stability. Defaults to 1e-5.
                weight (Tensor, optional): Gamma scale tensor. Defaults to None.
                bias (Tensor, optional): Beta bias tensor. Defaults to None.
                residual_input (Tensor, optional): Residual tensor to add before normalization. Defaults to None.
                memory_config (MemoryConfig, optional): Output memory configuration. Defaults to None.
                compute_kernel_config (DeviceComputeKernelConfig, optional): Compute kernel config. Defaults to None.

            Returns:
                BranchDescriptor: A branch descriptor for use with ttnn.experimental.parallel().
        )doc");

    // Factory function for Layer norm branch
    mod.def(
        "layer_norm_branch",
        &make_layer_norm_branch,
        nb::arg("cores"),
        nb::arg("input"),
        nb::arg("epsilon") = 1e-5f,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("residual_input") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        R"doc(
            Create a parallel branch for Layer normalization.

            Args:
                cores (CoreRangeSet): The cores this branch should execute on (must be disjoint from other branches).
                input (Tensor): Input tensor to normalize.
                epsilon (float): Small constant for numerical stability. Defaults to 1e-5.
                weight (Tensor, optional): Gamma scale tensor. Defaults to None.
                bias (Tensor, optional): Beta bias tensor. Defaults to None.
                residual_input (Tensor, optional): Residual tensor to add before normalization. Defaults to None.
                memory_config (MemoryConfig, optional): Output memory configuration. Defaults to None.
                compute_kernel_config (DeviceComputeKernelConfig, optional): Compute kernel config. Defaults to None.

            Returns:
                BranchDescriptor: A branch descriptor for use with ttnn.experimental.parallel().
        )doc");

    // Bind the parallel execution function
    mod.def(
        "parallel",
        [](const std::vector<std::shared_ptr<BranchDescriptor>>& branches) {
            return ExecuteParallel::invoke(std::vector<std::shared_ptr<BranchDescriptor>>(branches));
        },
        nb::arg("branches"),
        R"doc(
            Execute multiple operations in parallel as a single fused program.

            Each branch runs on disjoint core ranges within a single program dispatch.
            This enables operation fusion without kernel boundaries, maximizing hardware utilization.

            Args:
                branches (list[BranchDescriptor]): List of branch descriptors created via factory
                    functions (e.g., rms_norm_branch, layer_norm_branch).

            Returns:
                list[list[Tensor]]: Nested list where results[i] contains the output tensors
                from the i-th branch.

            Example:
                >>> cores_a = ttnn.CoreRangeSet(ttnn.CoreRange((0, 0), (3, 7)))
                >>> cores_b = ttnn.CoreRangeSet(ttnn.CoreRange((4, 0), (7, 7)))
                >>>
                >>> branch_a = ttnn.experimental.rms_norm_branch(cores_a, input_a, epsilon=1e-5)
                >>> branch_b = ttnn.experimental.rms_norm_branch(cores_b, input_b, epsilon=1e-5)
                >>>
                >>> results = ttnn.experimental.parallel([branch_a, branch_b])
                >>> output_a = results[0][0]
                >>> output_b = results[1][0]

            Note:
                - Core ranges must be disjoint between branches
                - All branches execute within a single program dispatch
                - Each branch can be a different operation type
        )doc");
}

}  // namespace ttnn::operations::experimental::parallel::detail
