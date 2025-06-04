// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::transformer::detail {

namespace py = pybind11;
void bind_concatenate_heads(py::module& module);

}  // namespace ttnn::operations::experimental::transformer::detail
