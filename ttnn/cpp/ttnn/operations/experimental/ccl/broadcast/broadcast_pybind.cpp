// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "broadcast_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/broadcast/broadcast.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_broadcast(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    // namespace py = pybind11;

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const MeshCoordinate& sender_coord,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<uint32_t> cluster_axis,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id) -> ttnn::Tensor {
                return self(input_tensor, sender_coord, num_links, memory_config, topology, cluster_axis, subdevice_id);
            },
            py::arg("input_tensor"),
            py::arg("sender_coord"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Linear,
            py::arg("cluster_axis") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt});
}

}  // namespace

void py_bind_broadcast(pybind11::module& module) {
    bind_broadcast(
        module,
        ttnn::experimental::broadcast,
        R"doc(
        Performs an broadcast operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor)
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the operation on.
            mesh_device (MeshDevice): Device mesh to perform the operation on.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-broadcast operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor.

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
