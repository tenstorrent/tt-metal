// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/operations/ccl/ccl_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn/operations/ccl/all_gather/all_gather_nanobind.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter_nanobind.hpp"
#include "ttnn/operations/ccl/barrier/barrier_nanobind.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "cpp/ttnn/operations/ccl/erisc_datamover_builder_helper.hpp"

namespace nb = nanobind;

namespace ttnn::operations::ccl {

namespace {
void bind_common(nb::module_& mod) {
    nb::enum_<ttnn::ccl::Topology>(mod, "Topology")
        .value("Ring", ttnn::ccl::Topology::Ring)
        .value("Linear", ttnn::ccl::Topology::Linear);

    mod.def(
        "initialize_edm_fabric",
        &ttnn::ccl::initialize_edm_fabric,
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("wrap_fabric_around_mesh") = false,
        nb::arg("context_switch_interval_override") = std::nullopt,
        nb::arg("topology") = ttnn::ccl::Topology::Linear);

    mod.def(
        "teardown_edm_fabric",
        &ttnn::ccl::teardown_edm_fabric,
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("wrap_fabric_around_mesh") = false,
        nb::arg("topology") = ttnn::ccl::Topology::Linear);
}
} // namespace

void py_module(nb::module_& mod) {
    ccl::bind_common(mod);
    ccl::bind_all_gather(mod);
    ccl::bind_reduce_scatter(mod);
    ccl::bind_barrier(mod);
}

}  // namespace ttnn::operations::ccl
