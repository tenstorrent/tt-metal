// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/ccl_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/ccl/all_gather/all_gather_pybind.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter_pybind.hpp"
#include "ttnn/operations/ccl/barrier/barrier_pybind.hpp"
#include "ttnn/operations/ccl/all_to_all_combine/all_to_all_combine_pybind.hpp"
#include "ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch_pybind.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/erisc_datamover_builder_helper.hpp"

namespace ttnn::operations::ccl {

void py_bind_common(pybind11::module& module) {
    py::enum_<ttnn::ccl::Topology>(module, "Topology")
        .value("Ring", ttnn::ccl::Topology::Ring)
        .value("Linear", ttnn::ccl::Topology::Linear);

    module.def(
        "initialize_edm_fabric",
        &ttnn::ccl::initialize_edm_fabric,
        py::arg("mesh_device"),
        py::kw_only(),
        py::arg("wrap_fabric_around_mesh") = false,
        py::arg("context_switch_interval_override") = std::nullopt,
        py::arg("topology") = ttnn::ccl::Topology::Linear);

    module.def(
        "teardown_edm_fabric",
        &ttnn::ccl::teardown_edm_fabric,
        py::arg("mesh_device"),
        py::kw_only(),
        py::arg("wrap_fabric_around_mesh") = false,
        py::arg("topology") = ttnn::ccl::Topology::Linear);
}

void py_module(py::module& module) {
    ccl::py_bind_common(module);
    ccl::py_bind_all_gather(module);
    ccl::py_bind_reduce_scatter(module);
    ccl::py_bind_barrier(module);
    ccl::py_bind_all_to_all_combine(module);
    ccl::py_bind_all_to_all_dispatch(module);
}

}  // namespace ttnn::operations::ccl
