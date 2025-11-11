// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_b1_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/experimental/deepseek_b1/matmul_1d/matmul_1d_pybind.hpp"
#include "ttnn/operations/experimental/deepseek_b1/layernorm/layernorm_pybind.hpp"

namespace py = pybind11;

namespace ttnn::operations::experimental::deepseek_b1 {

void py_module(py::module& module) {
    matmul_1d::detail::bind_matmul_1d(module);
    layernorm::detail::bind_normalization_layernorm(module);
}

}  // namespace ttnn::operations::experimental::deepseek_b1
