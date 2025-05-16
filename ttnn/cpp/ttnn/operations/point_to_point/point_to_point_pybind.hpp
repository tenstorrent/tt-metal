// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::point_to_point {
void py_bind_point_to_point(pybind11::module& module);

}  // namespace ttnn::operations::point_to_point
