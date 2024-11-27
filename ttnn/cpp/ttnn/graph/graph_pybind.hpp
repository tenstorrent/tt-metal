// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "pybind11/pybind_fwd.hpp"

namespace ttnn::graph {

void py_graph_module_types(pybind11::module& m);
void py_graph_module(pybind11::module& m);

}  // namespace ttnn::graph
