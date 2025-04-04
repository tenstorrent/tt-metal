// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pybind11/pybind_fwd.hpp"
#include "ttnn/cluster.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"

namespace py = pybind11;

namespace ttnn::cluster {

void py_cluster_module(py::module& module);

}  // namespace ttnn::cluster
