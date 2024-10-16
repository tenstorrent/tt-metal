// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {

void py_bind_all_reduce(pybind11::module& module);

}  // namespace ttnn::operations::experimental::ccl
