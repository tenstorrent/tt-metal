// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_softmax {
void bind_moreh_softmax_operation(py::module& module);
}
