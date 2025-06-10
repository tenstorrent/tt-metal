// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"
#include "ttnn/types.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_all_reduce_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const GlobalSemaphore& from_remote_multi_device_global_semaphore,
               const GlobalSemaphore& to_remote_multi_device_global_semaphore,
               const GlobalSemaphore& gather_multi_device_global_semaphore,
               ttnn::operations::reduction::ReduceType math_op,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology,
               const std::optional<size_t> num_links,
               std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    from_remote_multi_device_global_semaphore,
                    to_remote_multi_device_global_semaphore,
                    gather_multi_device_global_semaphore,
                    math_op,
                    memory_config,
                    topology,
                    num_links,
                    worker_subdevice_id_opt);
            },
            py::arg("input_tensor"),
            py::arg("from_remote_multi_device_global_semaphore"),
            py::arg("to_remote_multi_device_global_semaphore"),
            py::arg("gather_multi_device_global_semaphore"),
            py::arg("math_op"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Linear,
            py::arg("num_links") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const GlobalSemaphore& from_remote_multi_device_global_semaphore,
               const GlobalSemaphore& to_remote_multi_device_global_semaphore,
               const GlobalSemaphore& gather_multi_device_global_semaphore,
               ttnn::operations::reduction::ReduceType math_op,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology,
               const std::optional<size_t> num_links,
               std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    cluster_axis,
                    mesh_device,
                    from_remote_multi_device_global_semaphore,
                    to_remote_multi_device_global_semaphore,
                    gather_multi_device_global_semaphore,
                    math_op,
                    memory_config,
                    topology,
                    num_links,
                    worker_subdevice_id_opt);
            },
            py::arg("input_tensor"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::arg("from_remote_multi_device_global_semaphore"),
            py::arg("to_remote_multi_device_global_semaphore"),
            py::arg("gather_multi_device_global_semaphore"),
            py::arg("math_op"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Linear,
            py::arg("num_links") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               ttnn::Tensor& buffer_tensor,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const GlobalSemaphore& multi_device_global_semaphore,
               const std::optional<const DataType> dtype,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology,
               const std::optional<size_t> num_links,
               std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    buffer_tensor,
                    cluster_axis,
                    mesh_device,
                    multi_device_global_semaphore,
                    dtype,
                    memory_config,
                    topology,
                    num_links,
                    worker_subdevice_id_opt);
            },
            py::arg("input_tensor"),
            py::arg("buffer_tensor"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Linear,
            py::arg("num_links") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt});
}

}  // namespace

void py_bind_all_reduce_async(pybind11::module& module) {
    bind_all_reduce_async(
        module,
        ttnn::experimental::all_reduce_async,
        R"doc(

        Performs an all_reduce operation on multi-device :attr:`input_tensor` across all devices.  This operation requires a persistent
        fabric to be enabled in order to function.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-all-reduce operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-all-reduce operation on.
        * cluster_axis and mesh_device parameters are applicable only for Linear Topology.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            num_links (int, optional): Number of links to use for the all_reduce_async operation. Defaults to `None`, which indicates to the operation that it should choose. Note that this value will be ignored if there are fewer links available than requested.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:

            >>> full_tensor = torch.randn([1, 1, 256, 256], dtype=torch.bfloat16)
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
            >>> input_tensor = ttnn.from_torch(
                    full_tensor,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
                )
            >>> output = ttnn.experimental.all_reduce_async(input_tensor, topology=ttnn.Topology.Linear)

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
