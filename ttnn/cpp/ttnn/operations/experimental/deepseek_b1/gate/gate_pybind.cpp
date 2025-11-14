// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_pybind.hpp"
#include "gate.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

namespace py = pybind11;

namespace ttnn::operations::experimental::deepseek_b1::gate::detail {

void bind_gate(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Performs gate operation optimized for DeepSeek models.

        This operation applies a gating mechanism with router matmul, bias addition,
        sigmoid, top-k selection, and normalization.

        Args:
            a (ttnn.Tensor): First input tensor (activations) [1, 7168].
            b (ttnn.Tensor): Second input tensor (router weights) [7168, 256].
            expert_bias (ttnn.Tensor): Expert bias tensor [1, 256] or [256].

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for output.
            dtype (Optional[ttnn.DataType]): Data type for output.
            compute_kernel_config (Optional[DeviceComputeKernelConfig]): Compute kernel configuration.

        Returns:
            ttnn.Tensor: Result of the gate operation [1, 256].

        Example:
            >>> output = ttnn.experimental.deepseek_b1.gate(
            ...     a,
            ...     b,
            ...     expert_bias,
            ...     memory_config=ttnn.DRAM_MEMORY_CONFIG
            ... )
        )doc");

    using OperationType = decltype(ttnn::experimental::deepseek_b1::gate);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::deepseek_b1::gate,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& a,
               const ttnn::Tensor& b,
               const ttnn::Tensor& expert_bias,
               const std::optional<const ttnn::MemoryConfig>& memory_config,
               const std::optional<const ttnn::DataType>& dtype,
               const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
                return self(a, b, expert_bias, memory_config, dtype, compute_kernel_config);
            },
            py::arg("a"),
            py::arg("b"),
            py::arg("expert_bias"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
        });
}

}  // namespace ttnn::operations::experimental::deepseek_b1::gate::detail
