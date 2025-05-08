// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {

void py_bind_reduce_scatter_async(pybind11::module& module);

}  // namespace ttnn::operations::ccl
