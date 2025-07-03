// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::ccl {
namespace py = pybind11;
void py_bind_all_to_all_combine(pybind11::module& module);

}  // namespace ttnn::operations::ccl
