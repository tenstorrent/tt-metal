// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::kv_cache {
namespace py = pybind11;
void py_bind_kv_cache(py::module& module);

}  // namespace ttnn::operations::kv_cache
