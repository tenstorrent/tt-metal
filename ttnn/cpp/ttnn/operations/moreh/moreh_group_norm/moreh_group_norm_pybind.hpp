// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_group_norm {
void bind_moreh_group_norm_operation(py::module& module);
}  // namespace ttnn::operations::moreh::moreh_group_norm
