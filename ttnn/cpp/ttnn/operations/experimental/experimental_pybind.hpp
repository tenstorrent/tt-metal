// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/experimental/transformer/transformer_pybind.hpp"

namespace ttnn::operations::experimental {

void py_module(py::module& module) {
    // Transformer ops
    transformer::detail::bind_experimental_transformer_operations(module);
}

}  // namespace ttnn::operations::experimental
