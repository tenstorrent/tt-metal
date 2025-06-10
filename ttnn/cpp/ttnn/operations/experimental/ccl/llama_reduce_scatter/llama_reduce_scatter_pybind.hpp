// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::experimental::ccl {
namespace py = pybind11;
void py_bind_llama_reduce_scatter(pybind11::module& module);

}  // namespace ttnn::operations::experimental::ccl
