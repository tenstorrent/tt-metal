// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bernoulli_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "bernoulli.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::bernoulli {

void bind_bernoulli_operation(nb::module_& mod) {
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
        mod,
        ttnn::bernoulli,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::arg("seed") = 0,
            nb::kw_only(),
            nb::arg("output") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}
}  // namespace ttnn::operations::bernoulli
