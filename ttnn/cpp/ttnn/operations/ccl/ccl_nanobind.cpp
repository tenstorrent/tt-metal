// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ccl_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/ccl/mesh_partition/mesh_partition_nanobind.hpp"
#include "ttnn/operations/ccl/all_broadcast/all_broadcast_nanobind.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather_nanobind.hpp"
#include "ttnn/operations/ccl/all_to_all_combine/all_to_all_combine_nanobind.hpp"
#include "ttnn/operations/ccl/reduce_to_root/reduce_to_root_nanobind.hpp"
#include "ttnn/operations/ccl/broadcast/broadcast_nanobind.hpp"
#include "ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch_nanobind.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter_nanobind.hpp"
#include "ttnn/operations/ccl/all_reduce/all_reduce_nanobind.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::ccl {

namespace {
void bind_common(nb::module_& mod) {
    nb::enum_<ttnn::ccl::Topology>(mod, "Topology")
        .value("Ring", ttnn::ccl::Topology::Ring)
        .value("Linear", ttnn::ccl::Topology::Linear)
        .value("Mesh", ttnn::ccl::Topology::Mesh)
        .value("Torus", ttnn::ccl::Topology::Torus);

    mod.def(
        "get_usable_topology",
        [](const Tensor& tensor,
           const std::optional<tt::tt_fabric::Topology>& topology,
           const std::optional<uint32_t>& cluster_axis) {
            TT_FATAL(
                tt::tt_metal::is_device_tensor(tensor),
                "get_usable_topology requires a device tensor; got a host tensor whose mesh placement is unknown");
            return ttnn::ccl::get_usable_topology(tensor, topology, cluster_axis);
        },
        nb::arg("tensor"),
        nb::arg("topology") = nb::none(),
        nb::arg("cluster_axis") = nb::none(),
        R"doc(
            Resolve the CCL topology that is actually usable for a tensor on the current fabric.

            When ``topology`` is ``None`` this defaults to the topology the fabric was brought up
            with (``tt::tt_fabric::get_fabric_topology()``). A ring/torus request is demoted to
            linear/mesh when the tensor's devices along ``cluster_axis`` do not form a full
            wraparound, so the returned topology is always valid for the given tensor placement.
            This is the same selection the CCL ops perform internally, exposed so model code can
            stop hand-rolling Ring-vs-Linear detection.

            Args:
                tensor (ttnn.Tensor): A device tensor whose mesh placement determines the usable topology.
                topology (ttnn.Topology, optional): Requested topology. Defaults to the fabric topology.
                cluster_axis (int, optional): Cluster axis the CCL operates along.

            Returns:
                ttnn.Topology: The topology usable for this tensor.

            Example:
                >>> import ttnn
                >>> topology = ttnn.get_usable_topology(input_tensor, cluster_axis=1)
                >>> output = ttnn.reduce_scatter(input_tensor, dim=3, cluster_axis=1, topology=topology)
        )doc");
}
}  // namespace

void py_module(nb::module_& mod) {
    ccl::bind_common(mod);
    ccl::bind_mesh_partition(mod);
    ccl::bind_all_broadcast(mod);
    ccl::bind_all_gather(mod);
    ccl::bind_all_to_all_combine(mod);
    ccl::bind_reduce_to_root(mod);
    ccl::bind_all_to_all_dispatch(mod);
    ccl::bind_reduce_scatter(mod);
    ccl::bind_all_reduce(mod);
    ccl::bind_broadcast(mod);
}

}  // namespace ttnn::operations::ccl
