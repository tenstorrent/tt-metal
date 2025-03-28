// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_concat_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/all_gather_concat.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_all_gather_concat(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    // namespace py = pybind11;

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
               const uint32_t num_heads,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               bool enable_persistent_fabric_mode,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    dim,
                    multi_device_global_semaphore,
                    num_heads,
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id,
                    enable_persistent_fabric_mode);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::arg("multi_device_global_semaphore"),
            py::arg("num_heads").noconvert(),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("enable_persistent_fabric_mode") = false,
            py::arg("queue_id") = DefaultQueueId},

        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
               const uint32_t num_heads,
               const std::optional<uint32_t> num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               bool enable_persistent_fabric_mode,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    dim,
                    cluster_axis,
                    mesh_device,
                    multi_device_global_semaphore,
                    num_heads,
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id,
                    enable_persistent_fabric_mode);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::arg("multi_device_global_semaphore"),
            py::arg("num_heads").noconvert(),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("enable_persistent_fabric_mode") = false,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace detail

void py_bind_all_gather_concat(pybind11::module& module) {
    detail::bind_all_gather_concat(
        module,
        ttnn::experimental::all_gather_concat,
        R"doc(
        Performs a fused all-gather/concat operation on multi-device (specific to llama model):attr:`input_tensor` across all devices.
        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to perform operation.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-all-gather operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-all-gather operation on.
        * cluster_axis and mesh_device parameters are applicable only for Linear Topology.
        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md
        Keyword Args:
            num_links (int, optional): Number of links to use for the all-gather operation. Defaults to `1`.
            num_heads (int): Number of heads for NLP concat heads
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.
        Returns:
            ttnn.Tensor: the output tensor.
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
