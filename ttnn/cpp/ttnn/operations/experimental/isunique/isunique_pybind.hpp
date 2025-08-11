// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::isunique::detail {

namespace py = pybind11;
void bind_scatter_operation(py::module& module);

}  // namespace ttnn::operations::experimental::isunique::detail
