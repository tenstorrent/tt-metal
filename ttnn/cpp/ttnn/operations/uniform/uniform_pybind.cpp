// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "uniform_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "uniform.hpp"

namespace ttnn::operations::uniform {
void bind_uniform_operation(py::module& module) {
    std::string doc =
        R"doc(
        Update in-place the input tensor with values drawn from the continuous uniform distribution 1 / (`to` - `from`).

        Args:
            input (ttnn.Tensor): The tensor that provides the shape for the generated uniform tensor.
            from (float32): The lower bound of the uniform distribution. Defaults to 0.
            to (float32): The upper bound of the uniform distribution. Defaults to 1.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Configuration for the compute kernel. Defaults to `None`.

        Returns:
            ttnn.Tensor: The `input` tensor with updated values drawn from the specified uniform distribution.

        Example:
            >>> input = ttnn.to_device(ttnn.from_torch(torch.ones(3, 3), dtype=torch.bfloat16)), device=device)
            >>> ttnn.uniform(input)

        )doc";

    bind_registered_operation(
        module,
        ttnn::uniform,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("from") = 0,
            py::arg("to") = 1,
            py::arg("seed") = 0,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}
}  // namespace ttnn::operations::uniform
