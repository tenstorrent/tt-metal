// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/ccl_pybind.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/erisc_datamover_builder.hpp"

namespace ttnn::operations::ccl {

void py_bind_common(pybind11::module& module) {
    py::enum_<ttnn::ccl::Topology>(module, "Topology")
        .value("Ring", ttnn::ccl::Topology::Ring)
        .value("Linear", ttnn::ccl::Topology::Linear);

    module.def("initialize_edm_fabric", &ttnn::ccl::initialize_edm_fabric, py::kw_only(), py::arg("mesh_device"));

    module.def("teardown_edm_fabric", &ttnn::ccl::teardown_edm_fabric, py::kw_only(), py::arg("mesh_device"));
}

void py_module(py::module& module) {
    ccl::py_bind_common(module);
    ccl::py_bind_all_gather(module);
    ccl::py_bind_reduce_scatter(module);
    ccl::py_bind_barrier(module);
}

}  // namespace ttnn::operations::ccl
