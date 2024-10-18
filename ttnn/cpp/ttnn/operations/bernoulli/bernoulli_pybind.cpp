// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bernoulli_pybind.hpp"

#include <optional>

#include "bernoulli.hpp"
#include "pybind11/decorators.hpp"

namespace ttnn::operations::bernoulli {
void bind_bernoulli_operation(py::module &module) {
    std::string doc =
        R"doc(Bernoulli(input: Tensor, output: Optional[Tensor] = None, dtype: Optional[DataType] = None, memory_config: Optional[MemoryConfig] = None, compute_kernel_config: Optional[ComputeKernelConfig] = None) -> Tensor
    Generates a tensor to draw binary random numbers (0 or 1) from a Bernoulli distribution.
    This operation allows configuration of memory allocation using `memory_config` and computation settings via `compute_kernel_config`.

    Args:
        * :attr:`input`: The input tensor of probability values for the Bernoulli distribution.
        * :attr:`output`: The output tensor.
        * :attr:`dtype`: The output tensor dtype (default float32).
        * :attr:`memory_config`: The memory configuration for the generated tensor.
        * :attr:`compute_kernel_config`: Optional configuration for the compute kernel used during generation.

    Returns:
        Tensor: A new tensor with the same shape as `input` contains only 1 or 0.
    )doc";

    bind_registered_operation(
        module,
        ttnn::bernoulli,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::bernoulli
