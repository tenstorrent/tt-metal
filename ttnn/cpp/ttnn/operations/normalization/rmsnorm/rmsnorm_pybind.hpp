// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "rmsnorm.hpp"

namespace py = pybind11;

namespace ttnn::operations::normalization::detail {

void bind_normalization_rms_norm_operation(py::module& module) {

    ttnn::bind_registered_operation(
        module,
        ttnn::rms_norm,
        R"doc(rms_norm(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, epsilon: float = 1e-12, Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
            Compute rms_norm over :attr:`input_tensor`.
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("weight"),
            py::kw_only(),
            py::arg("epsilon") = 1e-12,
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::normalization::detail
