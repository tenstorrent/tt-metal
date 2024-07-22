// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/embedding/embedding/embedding_pybind.hpp"

namespace py = pybind11;

namespace ttnn::operations::embedding {
void py_module(py::module& module) {
    detail::bind_embedding(module);
}
}  // namespace ttnn::operations::embedding
