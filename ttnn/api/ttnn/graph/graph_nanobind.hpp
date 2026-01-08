// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::graph {

namespace nb = nanobind;

void py_graph_module_types(nb::module_& m);
void py_graph_module(nb::module_& m);

}  // namespace ttnn::graph
