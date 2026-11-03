// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ccl_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
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
                ttnn::is_device_tensor(tensor),
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

    mod.def(
        "get_num_links",
        [](const tt::tt_metal::distributed::MeshDevice& mesh_device, const std::optional<size_t>& cluster_axis) {
            TT_FATAL(
                !cluster_axis.has_value() || *cluster_axis < 2,
                "Invalid cluster axis {}. Must be 0 or 1",
                *cluster_axis);
            return ttnn::operations::ccl::common::get_num_links(mesh_device, cluster_axis);
        },
        nb::arg("mesh_device"),
        nb::arg("cluster_axis") = nb::none(),
        R"doc(
            Return the number of ethernet links CCL ops can use on the mesh device.

            Queries the fabric control plane for the usable routing planes between neighbouring
            devices on every row (``cluster_axis=1``) or column (``cluster_axis=0``) of the mesh,
            and returns the lowest count. Planes reserved for dispatch are excluded, and hops owned
            by another host are skipped. With ``cluster_axis=None`` the lowest count across both
            axes is returned.

            Falls back to 1 (with a warning) when no link could be measured, e.g. on a
            single-device mesh.

            Args:
                mesh_device (ttnn.MeshDevice): The mesh device the CCL runs on.
                cluster_axis (int, optional): Cluster axis the CCL operates along. Defaults to ``None``.

            Returns:
                int: The number of links usable along ``cluster_axis``.

            Example:
                >>> import ttnn
                >>> num_links = ttnn.get_num_links(mesh_device, cluster_axis=1)
                >>> output = ttnn.all_gather(input_tensor, dim=3, cluster_axis=1, num_links=num_links)
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
