// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::clip_grad_norm::detail {
namespace py = pybind11;
void bind_experimental_clip_grad_norm_operation(py::module& module);

}  // namespace ttnn::operations::experimental::clip_grad_norm::detail
