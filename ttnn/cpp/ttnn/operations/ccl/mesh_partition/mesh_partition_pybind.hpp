// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::ccl {

void py_bind_mesh_partition(pybind11::module& module);

}  // namespace ttnn::operations::ccl
