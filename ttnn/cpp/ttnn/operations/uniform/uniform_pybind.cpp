// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "uniform_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "uniform.hpp"

namespace ttnn::operations::uniform {
void bind_uniform_operation(py::module &module) {
    auto doc =
        R"doc(uniform(input: Tensor, from: float = 0, to: float = 1, memory_config: Optional[MemoryConfig] = None, compute_kernel_config: Optional[ComputeKernelConfig] = None) -> Tensor
    Generates a tensor with values drawn from a uniform distribution [`from`, `to`). The input tensor provides the shape for the output tensor, while the data type remains unchanged.
    This operation allows configuration of memory allocation using `memory_config` and computation settings via `compute_kernel_config`.

    Args:
        * :attr:`input`: The tensor that provides the shape for the generated uniform tensor.
        * :attr:`from`: The lower bound of the uniform distribution. Defaults to 0.
        * :attr:`to`: The upper bound of the uniform distribution. Defaults to 1.
        * :attr:`memory_config`: The memory configuration for the generated tensor.
        * :attr:`compute_kernel_config`: Optional configuration for the compute kernel used during generation.

    Returns:
        Tensor: A new tensor with the same shape as `input` and values drawn from the specified uniform distribution.
    )doc";

    bind_registered_operation(
        module,
        ttnn::uniform,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("from") = 0,
            py::arg("to") = 1,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::uniform
