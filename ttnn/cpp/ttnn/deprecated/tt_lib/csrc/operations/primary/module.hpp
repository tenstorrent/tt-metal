// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_op.hpp"

namespace py = pybind11;

namespace tt {
namespace operations {
namespace primary {

void py_module(py::module& m_primary) {
    m_primary.def("moreh_clip_grad_norm_",
                  &moreh_clip_grad_norm,
                  py::arg("inputs").noconvert(),
                  py::arg("max_norm").noconvert(),
                  py::arg("norm_type").noconvert() = 2.0f,
                  py::arg("error_if_nonfinite").noconvert() = false,
                  py::kw_only(),
                  py::arg("total_norm").noconvert() = std::nullopt,
                  py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
                  R"doc(
        "Performs a moreh_clip_grad_norm operation.
    )doc");
}

}  // namespace
   // primary
}  // namespace
   // operations
}  // namespace
   // tt
