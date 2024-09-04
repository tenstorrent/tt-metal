// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "moreh_dot_pybind.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/moreh/moreh_dot_op/device/moreh_dot_device_operation.hpp"
#include "moreh_dot.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_dot {

void bind_moreh_dot_operation(py::module& module) {

    bind_registered_operation(
        module,
        ttnn::moreh_dot,
        "Moreh moreh_dot Operation",
        ttnn::pybind_arguments_t{

            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("output_dtype") = ttnn::bfloat16,
            py::arg("output_mem_config") = std::nullopt});
}

}
