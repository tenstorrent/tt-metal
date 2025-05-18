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
            py::arg("subdevice_id") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& priority_tensor_a,
               const ttnn::Tensor& priority_tensor_b,
               const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    priority_tensor_a,
                    priority_tensor_b,
                    multi_device_global_semaphore,
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id);
            },
            py::arg("input_tensor"),
            py::arg("priority_tensor_a"),
            py::arg("priority_tensor_b"),
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
        swap_tensor_async(input_tensor, multi_device_global_semaphore, *, num_links=1, memory_config=None, topology=Topology.Ring, subdevice_id=None)
        swap_tensor_async(input_tensor, priority_tensor_a, priority_tensor_b, multi_device_global_semaphore, *, num_links=1, memory_config=None, topology=Topology.Ring, subdevice_id=None)

        Performs an asynchronous tensor swap across multiple devices using collective communication.

        This operation redistributes tensor data among devices in a cluster according to the specified communication topology. It supports two modes:

        1. **Basic Mode**
            Swaps tensor data equally among all participating devices. All data is preserved without conflict resolution.

        2. **Priority-Based Mode**
            Incorporates `priority_tensor_a` and `priority_tensor_b`. Each device provides:
            - `priority_tensor_a`: Its own priority
            - `priority_tensor_b`: The peer device’s priority

            The operation compares these priority tensors to decide which device's data to keep. **The data from the device with the higher priority is retained.**
            Priorities are user-defined integers (e.g., 0, 1, 2), typically encoded in a (TILE_SIZE, TILE_SIZE) tensor for each device. All devices must know each other's priorities.

        Arguments:
        ----------
        - input_tensor (Tensor): The tensor to be distributed or swapped across devices.
        - priority_tensor_a (Tensor, optional): Priority value of the current device. Required in priority-based mode.
        - priority_tensor_b (Tensor, optional): Priority value of the peer device. Required in priority-based mode.
        - multi_device_global_semaphore (MultiDeviceGlobalSemaphore): Synchronization primitive to coordinate across devices.
        - num_links (int, optional): Number of inter-device links to use. Defaults to 1.
        - memory_config (MemoryConfig, optional): Configuration for how the resulting tensor is laid out in memory.
        - topology (Topology, optional): Communication pattern (e.g., Ring, Linear). Defaults to Topology.Ring.
        - subdevice_id (SubDeviceId, optional): ID for targeting a specific subdevice.

        Returns:
        --------
        - Tensor: A new tensor with values redistributed across devices. In priority mode, the returned tensor only contains data from the higher-priority device in overlapping regions.

        Example:
        --------
        >>> output = swap_tensor_async(
        ...     input_tensor,
        ...     priority_tensor_a,
        ...     priority_tensor_b,
        ...     multi_device_global_semaphore,
        ...     num_links=1,
        ...     topology=Topology.Linear)

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
