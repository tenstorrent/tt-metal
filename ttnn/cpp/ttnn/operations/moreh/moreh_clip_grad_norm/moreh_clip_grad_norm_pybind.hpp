// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_clip_grad_norm {
void bind_moreh_clip_grad_norm_operation(py::module& module);
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
