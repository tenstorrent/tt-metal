// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_broadcast_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/all_broadcast_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_all_broadcast_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    // namespace py = pybind11;

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const GlobalSemaphore& multi_device_global_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id) -> std::vector<ttnn::Tensor> {
                return self(
                    input_tensor, multi_device_global_semaphore, num_links, memory_config, topology, subdevice_id);
            },
            py::arg("input_tensor"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Linear,
            py::arg("subdevice_id") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const ttnn::ccl::Topology topology,
               const GlobalSemaphore& multi_device_global_semaphore,
               const std::optional<ttnn::Tensor>& persistent_output_tensor,
               const std::optional<size_t> num_preferred_links,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id) -> std::vector<ttnn::Tensor> {
                return self(
                    input_tensor,
                    cluster_axis,
                    mesh_device,
                    topology,
                    multi_device_global_semaphore,
                    persistent_output_tensor,  // = std::nullopt,
                    memory_config,             // = std::nullopt,
                    num_preferred_links,       // = std::nullopt,
                    subdevice_id);             // = std::nullopt
            },
            py::arg("input_tensor"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::arg("topology"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("persistent_output_tensor") = std::nullopt,
            py::arg("num_links") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt});
}

}  // namespace

void py_bind_all_broadcast_async(pybind11::module& module) {
    bind_all_broadcast_async(
        module,
        ttnn::experimental::all_broadcast_async,
        R"doc(

        Performs an all-broadcast operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the operation on.
            mesh_device (MeshDevice): Device mesh to perform the operation on.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-broadcast operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            std::vector<ttnn.Tensor>: a vector of tensors from all the devices.

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
