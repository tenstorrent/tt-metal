// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swap_tensor_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "ttnn/operations/experimental/ccl/swap_tensor_async/swap_tensor_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_swap_tensor_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    // namespace py = pybind11;

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id) -> ttnn::Tensor {
                return self(
                    input_tensor, multi_device_global_semaphore, num_links, memory_config, topology, subdevice_id);
            },
            py::arg("input_tensor"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("subdevice_id") = std::nullopt});
}

}  // namespace detail

void py_bind_swap_tensor_async(pybind11::module& module) {
    detail::bind_swap_tensor_async(
        module,
        ttnn::experimental::swap_tensor_async,
        R"doc(

        Performs a swap operation on a multi-device :attr:`input_tensor`, exchanging tensor data across devices.

        This operation creates a new tensor with the same shape and layout as the input tensor, but with values swapped between devices according to the provided topology. No computation or transformation is applied to the data itself — only the location of data across devices changes.

        Args:
            input_tensor (ttnn.Tensor): A multi-device tensor to be swapped across devices.
            multi_device_global_semaphore (MultiDeviceGlobalSemaphore): A synchronization primitive used for coordinating communication across devices.

        Keyword Args:
            num_links (int, optional): Number of inter-device communication links to use. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. If not provided, defaults to the input tensor's memory config.
            topology (ttnn.Topology, optional): Communication topology to use for the swap. Valid options include `ttnn.Topology.Ring` and `ttnn.Topology.Linear`. Defaults to `ttnn.Topology.Ring`.
            subdevice_id (tt::tt_metal::SubDeviceId, optional): Optional subdevice identifier for finer control.

        Returns:
            ttnn.Tensor: A new tensor with the same shape as the input, but with values swapped across devices.

        Example:
            >>> full_tensor = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
            >>> physical_device_ids = ttnn.get_t3k_physical_device_ids_ring()
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 1), physical_device_ids=physical_device_ids[:2])
            >>> ttnn_tensor = ttnn.from_torch(
            ...     full_tensor,
            ...     dtype=input_dtype,
            ...     device=mesh_device,
            ...     layout=layout,
            ...     memory_config=mem_config,
            ...     mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(2, 1), dims=(-2, -1))
            ... )
            >>> ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
            >>> output = ttnn.swap_tensor_async(ttnn_tensor, multi_device_global_semaphore, topology=ttnn.Topology.Ring)

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
