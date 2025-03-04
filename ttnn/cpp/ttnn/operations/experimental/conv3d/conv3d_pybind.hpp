// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace ttnn::operations::experimental::conv3d::detail {

void py_bind_conv3d(pybind11::module& module);

}  // namespace ttnn::operations::experimental::conv3d::detail
