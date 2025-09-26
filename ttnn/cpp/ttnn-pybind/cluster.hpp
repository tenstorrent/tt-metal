// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::cluster {
namespace py = pybind11;

void py_cluster_module_types(py::module& module);
void py_cluster_module(py::module& module);

}  // namespace ttnn::cluster
