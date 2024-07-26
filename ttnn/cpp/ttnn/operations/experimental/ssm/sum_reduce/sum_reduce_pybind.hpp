// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sum_reduce.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::experimental::ssm::detail {
namespace py = pybind11;

void bind_ssm_sum_reduce(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::sum_reduce,
        R"doc(Performs a custom reduction along dim 3 which is used in the SSM block of the Mamba architecture. Performs the following PyTorch equivalent (where latent_size = 32):
            x = torch.sum(x.reshape(1, 1, shape[2], shape[3] // latent_size, latent_size), dim=-1).reshape(1, 1, shape[2], shape[3] // latent_size)
        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("math_fidelity") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::ssm::detail
