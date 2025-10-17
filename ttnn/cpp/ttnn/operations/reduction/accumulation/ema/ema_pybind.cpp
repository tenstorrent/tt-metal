// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ema_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/reduction/accumulation/ema/ema.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::reduction::accumulation::detail {
void bind_reduction_ema_operation(py::module& module) {
    auto docstring =
        R"doc(
        ``ttnn.ema(input: ttnn.Tensor, alpha: float, out: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor``

        Returns the exponential moving average of `input` along the last dimension.

        For a given `input` of size N along the last dimension, the `output` will also contain N elements and be such that:

        .. math::
            \mathrm{{output}}_i = \alpha \times \mathrm{{input}}_i + (1 - \alpha) \times \mathrm{{output}}_{i-1}

        with \mathrm{{output}}_0 = \mathrm{{input}}_0

        Args:
            input (ttnn.Tensor): input tensor. Must be on the device.
            alpha (float): the smoothing factor, typically between 0 and 1.

        Keyword Args:
            out (ttnn.Tensor, optional): preallocated output. If specified, `out` must have same shape as `input`, and must be on the same device.
            core_grid (ttnn.CoreGrid, optional): core grid for the operation. If not provided, an optimal core grid will be selected.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.ComputeKernelConfig, optional): compute kernel configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:

            Supported dtypes, layouts, ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - BFLOAT16, FLOAT32
                 - TILE
                 - 4

        Memory Support:
            - Interleaved: DRAM and L1

        Limitations:
            - Preallocated output must have the same shape as the input

        Example:
            .. code-block:: python

                # Create tensor
                tensor_input = ttnn.rand((2,3,4), device=device)

                # Apply ttnn.ema() with alpha=0.99
                tensor_output = ttnn.ema(tensor_input, 0.99)

                # With preallocated output
                preallocated_output = ttnn.rand([2, 3, 4], dtype=ttnn.bfloat16, device=device)

                tensor_output = ttnn.ema(tensor_input, 0.99, out=preallocated_output)

        )doc";

    using OperationType = decltype(ttnn::ema);
    bind_registered_operation(
        module,
        ttnn::ema,
        docstring,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const float& alpha,
               std::optional<Tensor> optional_out,
               const std::optional<CoreGrid>& core_grid,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) -> Tensor {
                return self(input_tensor, alpha, optional_out, core_grid, memory_config, compute_kernel_config);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("alpha"),
            py::kw_only(),
            py::arg("out") = std::nullopt,
            py::arg("core_grid") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::reduction::accumulation::detail
