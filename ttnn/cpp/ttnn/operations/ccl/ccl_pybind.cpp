// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/ccl_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/ccl/mesh_partition/mesh_partition_pybind.hpp"
#include "ttnn/operations/ccl/barrier/barrier_pybind.hpp"
#include "ttnn/operations/ccl/all_to_all_combine/all_to_all_combine_pybind.hpp"
#include "ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch_pybind.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include <tt-metalium/fabric.hpp>

namespace ttnn::operations::ccl {

void py_bind_common(pybind11::module& module) {
    py::enum_<ttnn::ccl::Topology>(module, "Topology")
        .value("Ring", ttnn::ccl::Topology::Ring)
        .value("Linear", ttnn::ccl::Topology::Linear);
}

void py_module(py::module& module) {
    ccl::py_bind_common(module);
    ccl::py_bind_mesh_partition(module);
    ccl::py_bind_barrier(module);
    ccl::py_bind_all_to_all_combine(module);
    ccl::py_bind_all_to_all_dispatch(module);
}

}  // namespace ttnn::operations::ccl
