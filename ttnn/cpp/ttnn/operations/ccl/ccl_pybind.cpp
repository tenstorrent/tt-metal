// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/ccl_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/ccl/mesh_partition/mesh_partition_pybind.hpp"
#include "ttnn/operations/ccl/all_broadcast/all_broadcast_pybind.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather_pybind.hpp"
#include "ttnn/operations/ccl/all_to_all_combine/all_to_all_combine_pybind.hpp"
#include "ttnn/operations/ccl/reduce_to_root/reduce_to_root_pybind.hpp"
#include "ttnn/operations/ccl/broadcast/broadcast_pybind.hpp"
#include "ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch_pybind.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter_pybind.hpp"
#include "ttnn/operations/ccl/all_reduce/all_reduce_pybind.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::ccl {

void py_bind_common(pybind11::module& module) {
    py::enum_<ttnn::ccl::Topology>(module, "Topology")
        .value("Ring", ttnn::ccl::Topology::Ring)
        .value("Linear", ttnn::ccl::Topology::Linear);
}

void py_module(py::module& module) {
    ccl::py_bind_common(module);
    ccl::py_bind_mesh_partition(module);
    ccl::py_bind_all_broadcast(module);
    ccl::py_bind_all_gather(module);
    ccl::py_bind_all_to_all_combine(module);
    ccl::py_bind_reduce_to_root(module);
    ccl::py_bind_all_to_all_dispatch(module);
    ccl::py_bind_reduce_scatter(module);
    ccl::py_bind_all_reduce(module);
    ccl::py_bind_broadcast(module);
}

}  // namespace ttnn::operations::ccl
