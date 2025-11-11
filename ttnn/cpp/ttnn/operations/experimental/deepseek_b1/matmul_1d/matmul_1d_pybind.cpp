// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_1d_pybind.hpp"
#include "matmul_1d.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

namespace py = pybind11;

namespace ttnn::operations::experimental::deepseek_b1::matmul_1d::detail {

void bind_matmul_1d(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Performs matrix multiplication with 1D multicast configuration optimized for DeepSeek models.

        This operation performs matmul with hardcoded optimizations:
        - in0_block_w = K (entire K dimension)
        - out_subblock_h = 1, out_subblock_w = 1
        - per_core_M = 1, per_core_N = 1

        Args:
            input_tensor_a (ttnn.Tensor): First input tensor (typically width-sharded).
            input_tensor_b (ttnn.Tensor): Second input tensor (typically interleaved).
            core_grid (ttnn.CoreGrid): Grid of cores to use (e.g., CoreGrid(x=8, y=7)).

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for output.
            dtype (Optional[ttnn.DataType]): Data type for output.
            compute_kernel_config (Optional[DeviceComputeKernelConfig]): Compute kernel configuration.

        Returns:
            ttnn.Tensor: Result of the matrix multiplication.

        Example:
            >>> # Width-sharded matmul across 56 cores
            >>> output = ttnn.experimental.deepseek_b1.matmul_1d(
            ...     input_a, input_b,
            ...     core_grid=ttnn.CoreGrid(x=8, y=7)
            ... )
        )doc");

    using OperationType = decltype(ttnn::experimental::deepseek_b1::matmul_1d);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::deepseek_b1::matmul_1d,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::CoreGrid& core_grid,
               const std::optional<const ttnn::MemoryConfig>& memory_config,
               const std::optional<const ttnn::DataType>& dtype,
               const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
                return self(input_tensor_a, input_tensor_b, core_grid, memory_config, dtype, compute_kernel_config);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("core_grid"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
        });
}

}  // namespace ttnn::operations::experimental::deepseek_b1::matmul_1d::detail
