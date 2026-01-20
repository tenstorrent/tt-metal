// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include "ttnn-nanobind/decorators.hpp"
#include "rmsnorm.hpp"
#include "ttnn/operations/experimental/parallel/device/parallel_device_operation_types.hpp"
#include "ttnn/operations/experimental/sequential/device/sequential_device_operation_types.hpp"

namespace ttnn::operations::normalization::detail {

void bind_normalization_rms_norm(nb::module_& mod) {
    auto py_operation = ttnn::bind_registered_operation(
        mod,
        ttnn::rms_norm,
        R"doc(
            Computes RMS norm over :attr:`input_tensor`.
            See `Root Mean Square Layer Normalization <https://arxiv.org/pdf/1910.07467>`_ for more details.

            .. math::
              \text{RMS_norm}(x, \gamma, \beta, \epsilon) = \frac{x}{\sqrt{\epsilon+\frac{1}{N}\sum_{i=1}^{N}x^{2}}} \cdot \gamma + \beta

            Where:
                - :math:`\gamma` and :math:`\beta` are optional scale and shift parameters
                - :math:`\epsilon` is a small constant

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            epsilon (float): 1e-12.
            weight (ttnn.Tensor, optional): Defaults to `None`.
            bias (ttnn.Tensor, optional): Defaults to `None`.
            residual_input_tensor (ttnn.Tensor, optional): Defaults to `None`.
            program_config (ttnn.ProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig): Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported data types and layouts by tensor:

            .. list-table:: input_tensor
               :header-rows: 1

               * - dtype
                 - layout
               * - BFLOAT16, FLOAT32, BFLOAT8_B
                 - TILE

            .. list-table:: residual_input_tensor
               :header-rows: 1

               * - dtype
                 - layout
               * - BFLOAT16, FLOAT32, BFLOAT8_B
                 - TILE

            .. list-table:: weight (gamma) and bias (beta)
               :header-rows: 1

               * - dtype
                 - layout
               * - BFLOAT16, FLOAT32
                 - TILE, ROW_MAJOR

            .. list-table:: output_tensor
               :header-rows: 1

               * - dtype
                 - layout
               * - BFLOAT16, FLOAT32, BFLOAT8_B (matching input)
                 - TILE

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded (L1): Width and Block sharded

        Limitations:
            - All input tensors must be on-device and have a rank >= 1.
            - Unsharded tensors must be interleaved, sharded inputs cannot be height-sharded.
            - If `residual_input_tensor` is provided, it must match the :attr:`input_tensor`'s padded shape.
            - If the `weight`/`bias` tensors are TILE layout: last padded dim must match :attr:`input_tensor`'s last padded dim.
            - If the `weight`/`bias` tensors are ROW_MAJOR layout: last padded dim must be TILE_WIDTH.
            - If the :attr:`input_tensor` is sharded, the :attr:`output` must also be sharded. In that case, the
              :attr:`output` memory layout and buffer type must match the :attr:`input_tensor`'s memory configuration.
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("epsilon") = 1e-12,
            nb::arg("weight") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("residual_input_tensor") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});

    // Add branch() method for parallel execution support
    // Using a lambda to wrap the static function since py_operation.def() expects a method with 'self'
    py_operation.def(
        "branch",
        [](const std::decay_t<decltype(ttnn::rms_norm)>& /*self*/,
           const ttnn::Tensor& input_tensor,
           const tt::tt_metal::CoreRangeSet& cores,
           float epsilon,
           const std::optional<const ttnn::Tensor>& weight,
           const std::optional<const ttnn::Tensor>& bias,
           const std::optional<const ttnn::Tensor>& residual_input_tensor,
           const std::optional<MemoryConfig>& memory_config,
           const std::optional<const LayerNormProgramConfig>& program_config,
           std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
            return ExecuteRMSNorm::branch(
                input_tensor,
                cores,
                epsilon,
                weight,
                bias,
                residual_input_tensor,
                memory_config,
                program_config,
                compute_kernel_config);
        },
        nb::arg("input_tensor"),
        nb::arg("cores"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-12f,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("residual_input_tensor") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        R"doc(
            Create a branch descriptor for parallel execution with ttnn.parallel.

            This allows running multiple RMS norm operations in parallel on disjoint
            core ranges within a single fused program.

            Args:
                input_tensor (ttnn.Tensor): Input tensor to normalize.
                cores (ttnn.CoreRangeSet): Core range for this branch (must be disjoint from other branches).

            Keyword Args:
                epsilon (float): Small constant for numerical stability. Defaults to 1e-12.
                weight (ttnn.Tensor, optional): Gamma scale tensor. Defaults to None.
                bias (ttnn.Tensor, optional): Beta bias tensor. Defaults to None.
                residual_input_tensor (ttnn.Tensor, optional): Residual tensor. Defaults to None.
                memory_config (ttnn.MemoryConfig, optional): Output memory config. Defaults to None.
                program_config (ttnn.ProgramConfig, optional): Program config. Defaults to None.
                compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute config. Defaults to None.

            Returns:
                BranchDescriptor: A branch descriptor for use with ttnn.parallel().

            Example:
                >>> cores_a = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
                >>> cores_b = ttnn.CoreRangeSet([ttnn.CoreRange((4, 0), (7, 3))])
                >>> branch_a = ttnn.rms_norm.branch(input_a, cores_a, epsilon=1e-5, weight=w_a)
                >>> branch_b = ttnn.rms_norm.branch(input_b, cores_b, epsilon=1e-5, weight=w_b)
                >>> results = ttnn.parallel([branch_a, branch_b])
        )doc");

    // Add step() method for sequential execution support
    py_operation.def(
        "step",
        [](const std::decay_t<decltype(ttnn::rms_norm)>& /*self*/,
           const ttnn::Tensor& input_tensor,
           const tt::tt_metal::CoreRangeSet& cores,
           float epsilon,
           const std::optional<const ttnn::Tensor>& weight,
           const std::optional<const ttnn::Tensor>& bias,
           const std::optional<const ttnn::Tensor>& residual_input_tensor,
           const std::optional<MemoryConfig>& memory_config,
           const std::optional<const LayerNormProgramConfig>& program_config,
           std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
            return ExecuteRMSNorm::step(
                input_tensor,
                cores,
                epsilon,
                weight,
                bias,
                residual_input_tensor,
                memory_config,
                program_config,
                compute_kernel_config);
        },
        nb::arg("input_tensor"),
        nb::arg("cores"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-12f,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("residual_input_tensor") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        R"doc(
            Create a step descriptor for sequential execution with ttnn.sequential.

            This allows chaining RMS norm as a step in a sequential pipeline. Each step
            specifies the cores it should execute on, allowing sequential operations
            to be composed with parallel operations.

            Args:
                input_tensor (ttnn.Tensor): Input tensor to normalize.
                cores (ttnn.CoreRangeSet): Core range for this step to execute on.

            Keyword Args:
                epsilon (float): Small constant for numerical stability. Defaults to 1e-12.
                weight (ttnn.Tensor, optional): Gamma scale tensor. Defaults to None.
                bias (ttnn.Tensor, optional): Beta bias tensor. Defaults to None.
                residual_input_tensor (ttnn.Tensor, optional): Residual tensor. Defaults to None.
                memory_config (ttnn.MemoryConfig, optional): Output memory config. Defaults to None.
                program_config (ttnn.ProgramConfig, optional): Program config. Defaults to None.
                compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute config. Defaults to None.

            Returns:
                StepDescriptor: A step descriptor for use with ttnn.sequential().

            Example:
                >>> cores = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (3, 3))])
                >>> step1 = ttnn.rms_norm.step(input1, cores, epsilon=1e-5, weight=w1)
                >>> step2 = ttnn.layer_norm.step(input2, cores, epsilon=1e-6, weight=w2)
                >>> results = ttnn.sequential([step1, step2])
        )doc");
}

}  // namespace ttnn::operations::normalization::detail
