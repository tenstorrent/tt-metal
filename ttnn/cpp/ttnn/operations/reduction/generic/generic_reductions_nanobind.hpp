// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "cpp/ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::reduction::detail {

namespace nb = nanobind;

template <typename reduction_operation_t>
void bind_reduction_operation(nb::module_& mod, const reduction_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(

            Args:
                input_a (ttnn.Tensor): the input tensor.
                dim (number): dimension value .

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor.

            Example:

                >>> input_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
                >>> output = ttnn.{0}(input_a, dim, memory_config)
        )doc",
        operation.base_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("dim") = std::nullopt,
            nb::arg("keepdim") = true,
            nb::kw_only(),
            nb::arg("memory_config") = std::nullopt,
            nb::arg("compute_kernel_config") = std::nullopt,
            nb::arg("scalar") = 1.0f});
}

}  // namespace ttnn::operations::reduction::detail
