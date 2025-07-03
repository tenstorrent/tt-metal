// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bernoulli_pybind.hpp"

#include "bernoulli.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::bernoulli {
void bind_bernoulli_operation(py::module& module) {
    std::string doc =
        R"doc(
        Generates a tensor to draw binary random numbers (0 or 1) from a Bernoulli distribution.

        Args:
            input (ttnn.Tensor): The input tensor of probability values for the Bernoulli distribution.

        Keyword args:
            output (ttnn.Tensor, optional): The output tensor.
            dtype (ttnn.DataType, optional): Output tensor dtype, default float32.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Configuration for the compute kernel. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> input = ttnn.to_device(ttnn.from_torch(torch.empty(3, 3).uniform_(0, 1), dtype=torch.bfloat16)), device=device)
            >>> output = ttnn.bernoulli(input)

        )doc";

    bind_registered_operation(
        module,
        ttnn::bernoulli,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("seed") = 0,
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::bernoulli
