// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::conv::conv_transpose2d {
void py_bind_conv_transpose2d(pybind11::module& module);
}
