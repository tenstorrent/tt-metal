// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ccl2_pybind.hpp"

namespace ttnn::operations::ccl2 {

// TODO this conflicts with ccl's Topology
// void py_bind_common(py::module& module) {
//    py::enum_<ttnn::ccl2::Topology>(module, "Topology")
//        .value("Ring", ttnn::ccl2::Topology::Ring)
//        .value("Linear", ttnn::ccl2::Topology::Linear);
//}

void py_module(py::module& module) {
    // py_bind_common(module);
    py_bind_all_gather(module);
}

}  // namespace ttnn::operations::ccl2
