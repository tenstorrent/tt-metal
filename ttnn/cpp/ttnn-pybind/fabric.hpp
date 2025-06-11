// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::fabric {

namespace py = pybind11;
void py_bind_fabric_api(py::module& module);

}  // namespace ttnn::fabric
