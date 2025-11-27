// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ccl_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn/operations/ccl/mesh_partition/mesh_partition_nanobind.hpp"
#include "ttnn/operations/ccl/all_broadcast/all_broadcast_nanobind.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather_nanobind.hpp"
#include "ttnn/operations/ccl/all_to_all_combine/all_to_all_combine_nanobind.hpp"
#include "ttnn/operations/ccl/broadcast/broadcast_nanobind.hpp"
#include "ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch_nanobind.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter_nanobind.hpp"
#include "ttnn/operations/ccl/all_reduce/all_reduce_nanobind.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include <tt-metalium/fabric.hpp>

namespace ttnn::operations::ccl {

namespace {
void bind_common(nb::module_& mod) {
    nb::enum_<ttnn::ccl::Topology>(mod, "Topology")
        .value("Ring", ttnn::ccl::Topology::Ring)
        .value("Linear", ttnn::ccl::Topology::Linear);
}
}  // namespace

void py_module(nb::module_& mod) {
    ccl::bind_common(mod);
    ccl::bind_mesh_partition(mod);
    ccl::bind_all_broadcast(mod);
    ccl::bind_all_gather(mod);
    ccl::bind_all_to_all_combine(mod);
    ccl::bind_all_to_all_dispatch(mod);
    ccl::bind_reduce_scatter(mod);
    ccl::bind_all_reduce(mod);
    ccl::bind_broadcast(mod);
}

}  // namespace ttnn::operations::ccl
