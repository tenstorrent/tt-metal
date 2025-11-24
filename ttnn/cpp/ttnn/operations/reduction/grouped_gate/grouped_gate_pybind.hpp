// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ttnn::operations::reduction {

void py_bind_grouped_gate(pybind11::module& module);

}  // namespace ttnn::operations::reduction
