// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moreh_norm.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::moreh::moreh_norm {

void bind_moreh_norm_operation(nb::module_& mod) {
    const auto* doc = R"doc(
        Moreh Norm Operation

        Computes the norm of the input tensor.

        Args:
            input (ttnn.Tensor): the input tensor.
            p (float): the order of the norm.

        Keyword args:
            dim (int or List[int], optional): the dimension(s) to reduce. Defaults to `None`.
            keepdim (bool, optional): whether the output tensor has dim retained or not. Defaults to `False`.
            output (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): compute kernel configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
        )doc";

    ttnn::bind_function<"moreh_norm">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::moreh_norm,
            nb::arg("input"),
            nb::arg("p"),
            nb::kw_only(),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::arg("output") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));
}

}  // namespace ttnn::operations::moreh::moreh_norm
