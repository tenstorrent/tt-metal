// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "rmsnorm.hpp"

namespace ttnn::operations::normalization::detail {

void bind_normalization_rms_norm_descriptor(nb::module_& mod) {
    mod.def(
        "rms_norm_descriptor",
        [](const ttnn::Tensor& input_tensor,
           float epsilon,
           const std::optional<const ttnn::Tensor>& weight,
           const std::optional<const ttnn::Tensor>& bias,
           const std::optional<const ttnn::Tensor>& residual_input_tensor,
           const std::optional<ttnn::MemoryConfig>& memory_config,
           const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config,
           std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
            auto result = ttnn::rms_norm_descriptor(
                input_tensor,
                epsilon,
                weight,
                bias,
                residual_input_tensor,
                memory_config,
                program_config,
                compute_kernel_config);
            return nb::make_tuple(std::move(result.descriptor), std::move(result.output_tensors));
        },
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-12,
        nb::arg("weight") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::arg("residual_input_tensor") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        R"doc(
        Creates a ProgramDescriptor for an RMS norm operation without enqueuing it.

        Runs the same pipeline as rms_norm() (output allocation, validation, factory selection)
        but returns a (ProgramDescriptor, output_tensor) tuple instead of executing.

        Args:
            input_tensor (ttnn.Tensor): The input tensor.

        Keyword args:
            epsilon (float): Small constant for numerical stability. Default: 1e-12.
            weight (ttnn.Tensor, optional): Weight (gamma) tensor. Default: None.
            bias (ttnn.Tensor, optional): Bias (beta) tensor. Default: None.
            residual_input_tensor (ttnn.Tensor, optional): Residual tensor. Default: None.
            memory_config (ttnn.MemoryConfig, optional): Output memory config. Default: None.
            program_config (LayerNormProgramConfig, optional): Program config. Default: None.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute config. Default: None.

        Returns:
            tuple: (ProgramDescriptor, output_tensor)
        )doc");
}

void bind_normalization_rms_norm(nb::module_& mod) {
    bind_normalization_rms_norm_descriptor(mod);

    // Bind rmsnorm_default_compute_config function
    mod.def(
        "rmsnorm_default_compute_config",
        &ttnn::rmsnorm_default_compute_config,
        nb::arg("arch"),
        R"doc(
        Returns the default compute kernel config for rmsnorm.

        Args:
            arch (tt.ARCH): The device architecture.

        Returns:
            ttnn.DeviceComputeKernelConfig: The default compute config for RMS norm (HiFi4, approx_mode=True, fp32_dest_acc_en=False).
        )doc");

    const auto* doc = R"doc(
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
        )doc";

    ttnn::bind_function<"rms_norm">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::rms_norm,
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("epsilon") = 1e-12,
            nb::arg("weight") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("residual_input_tensor") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}

}  // namespace ttnn::operations::normalization::detail
