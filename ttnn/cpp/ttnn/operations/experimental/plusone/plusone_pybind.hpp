// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::experimental::plusone::detail {
namespace py = pybind11;
void bind_experimental_plusone_operation(py::module& module);

}  // namespace ttnn::operations::experimental::plusone::detail
