// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {
namespace py = pybind11;
void bind_moreh_clip_grad_norm_operation(py::module& module);
}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
