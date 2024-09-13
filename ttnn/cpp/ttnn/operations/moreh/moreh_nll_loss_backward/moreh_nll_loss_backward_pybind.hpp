// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_nll_loss_backward {

void bind_moreh_nll_loss_backward_operation(py::module &module);

}  // namespace ttnn::operations::moreh::moreh_nll_loss_backward
