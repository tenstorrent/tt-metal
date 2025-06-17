// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_to_all_async/all_to_all_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_all_to_all_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               ttnn::Tensor& persistent_intermediate_buffer,
               ttnn::Tensor& persistent_output_buffer,
               const int32_t in_dim,
               const int32_t out_dim,
               const GlobalSemaphore& multi_device_global_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    persistent_intermediate_buffer,
                    persistent_output_buffer,
                    in_dim,
                    out_dim,
                    multi_device_global_semaphore,
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id);
            },
            py::arg("input_tensor"),
            py::arg("persistent_intermediate_buffer"),
            py::arg("persistent_output_buffer"),
            py::arg("in_dim"),
            py::arg("out_dim"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("subdevice_id") = std::nullopt});
}

}  // namespace detail

void py_bind_all_to_all_async(pybind11::module& module) {
    detail::bind_all_to_all_async(
        module,
        ttnn::experimental::all_to_all_async,
        R"doc(
        Performs an asynchronous all-to-all collective communication operation across multiple devices.

        This operation redistributes data between devices by having each device split its input tensor
        into N chunks (where N is the number of devices) and sending the i-th chunk to device i.
        The operation uses asynchronous kernels for improved performance.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to redistribute. Must be sharded along in_dim.
            persistent_intermediate_buffer (ttnn.Tensor): Intermediate buffer used during redistribution.
                Must be large enough to hold the full output tensor.
            persistent_output_buffer (ttnn.Tensor): Buffer where final output will be written.
                Will be sharded along out_dim.
            in_dim (int): Dimension along which input tensor is currently sharded (2 or 3).
            out_dim (int): Dimension along which output tensor should be sharded (2 or 3).
            multi_device_global_semaphore (GlobalSemaphore): Semaphore for synchronizing between devices.

        Keyword Args:
            num_links (int, optional): Number of fabric links to use for communication. Defaults to 1.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for buffers.
            topology (ttnn.Topology, optional): Network topology to use. Currently only Ring topology
                is supported. Defaults to ttnn.Topology.Ring.
            subdevice_id (SubDeviceId, optional): Target specific subdevice for the operation.

        Returns:
            ttnn.Tensor: Output tensor containing redistributed data, sharded along out_dim.

        Notes:
            - The operation requires persistent intermediate and output buffers to be pre-allocated
            - Only supports resharding between dimensions 2 and 3
            - Chunks are sent in a ring pattern for optimal bandwidth utilization
            - Dynamic alternate routing is not supported

        Example:
            >>> # Redistribute data from dim 2 to dim 3 across 4 devices
            >>> intermediate_buf = ttnn.zeros_like(input_tensor)  # Intermediate buffer
            >>> output_buf = ttnn.zeros_like(input_tensor)       # Output buffer
            >>> semaphore = ttnn.GlobalSemaphore()              # For device sync
            >>> result = ttnn.experimental.all_to_all_async(
            ...     input_tensor,
            ...     intermediate_buf,
            ...     output_buf,
            ...     in_dim=2,
            ...     out_dim=3,
            ...     multi_device_global_semaphore=semaphore
            ... )
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
