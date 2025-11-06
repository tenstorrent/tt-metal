// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "rmsnorm.hpp"

namespace ttnn::operations::normalization::detail {

void bind_normalization_rms_norm(nb::module_& mod) {
    ttnn::bind_registered_operation(
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

        Limitations:
            - All input tensors must be on-device and have a rank >= 1.
            - Unsharded tensors must be interleaved, sharded inputs cannot be height-sharded.
            - If `residual_input_tensor` is provided, it must match the :attr:`input_tensor`'s padded shape.
            - If the `weight`/`bias` tensors are TILE layout: last padded dim must match :attr:`input_tensor`'s last padded dim.
            - If the `weight`/`bias` tensors are ROW_MAJOR layout: last padded dim must be TILE_WIDTH.
            - If the :attr:`input_tensor` is sharded, the :attr:`output` must also be sharded. In that case, the
              :attr:`output` memory layout and buffer type must match the :attr:`input_tensor`'s memory configuration.

        Example:
            .. code-block:: python

              h, w = 32, 64
              batch_size = 1

              input_tensor = ttnn.rand([batch_size, h, w], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
              weight = ttnn.rand([w], dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
              output_tensor = ttnn.rms_norm(input_tensor, weight=weight)

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
}

}  // namespace ttnn::operations::normalization::detail
