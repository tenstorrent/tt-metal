// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::full_like {
namespace py = pybind11;
void bind_full_like_operation(py::module& module);
}  // namespace ttnn::operations::full_like
