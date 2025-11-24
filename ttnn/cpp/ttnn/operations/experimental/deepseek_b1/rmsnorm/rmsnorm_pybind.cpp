// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_pybind.hpp"
#include "rmsnorm.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

namespace py = pybind11;

namespace ttnn::operations::experimental::deepseek_b1::rmsnorm::detail {

void bind_rmsnorm(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Performs RMSNorm (Root Mean Square Layer Normalization) on the input tensor.

        This operation normalizes the input tensor using RMS normalization and scales by gamma (weight).
        All tensors must be sharded on the same cores in L1 memory.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to normalize (sharded in L1).
            gamma_tensor (ttnn.Tensor): Scale/weight tensor (sharded in L1, same sharding as input).
            output_tensor (ttnn.Tensor): Output tensor (sharded in L1, same sharding as input).

        Keyword args:
            epsilon (float): A small constant for numerical stability. Defaults to 1e-6.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Configuration for the compute kernel. Defaults to `None`.

        Returns:
            ttnn.Tensor: The output tensor with RMSNorm applied.

        Example:
            >>> # RMSNorm with gamma weight
            >>> output = ttnn.experimental.deepseek_b1.rmsnorm(
            ...     input_tensor,
            ...     gamma_tensor,
            ...     output_tensor,
            ...     epsilon=1e-6
            ... )
        )doc");

    using OperationType = decltype(ttnn::experimental::deepseek_b1::rmsnorm);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::deepseek_b1::rmsnorm,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& gamma_tensor,
               const ttnn::Tensor& output_tensor,
               float epsilon,
               uint32_t numel,
               const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
                return self(input_tensor, gamma_tensor, output_tensor, epsilon, numel, compute_kernel_config);
            },
            py::arg("input_tensor"),
            py::arg("gamma_tensor"),
            py::arg("output_tensor"),
            py::kw_only(),
            py::arg("epsilon") = 1e-6f,
            py::arg("numel"),
            py::arg("compute_kernel_config") = std::nullopt,
        });
}

}  // namespace ttnn::operations::experimental::deepseek_b1::rmsnorm::detail
