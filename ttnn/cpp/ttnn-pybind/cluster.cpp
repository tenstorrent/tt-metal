// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cluster.hpp"

#include <pybind11/pybind11.h>

#include <tt-metalium/tt_metal.hpp>

#include "ttnn/cluster.hpp"

using namespace tt::tt_metal;

namespace py = pybind11;

namespace ttnn::cluster {

void py_cluster_module(py::module& module) {
    module.def(
        "serialize_cluster_descriptor",
        &ttnn::cluster::serialize_cluster_descriptor,
        R"doc(
               Serialize cluster descriptor to a file.
             )doc");
}

}  // namespace ttnn::cluster
