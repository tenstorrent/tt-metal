// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_to_all_async_generic/all_to_all_async_generic.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_all_to_all_async_generic(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t in_dim,
               const int32_t out_dim,
               const std::optional<ttnn::Tensor>& persistent_output_buffer,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               std::optional<uint32_t> cluster_axis = std::nullopt) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    persistent_output_buffer,
                    in_dim,
                    out_dim,
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id,
                    cluster_axis);
            },
            py::arg("input_tensor"),
            py::arg("in_dim"),
            py::arg("out_dim"),
            py::kw_only(),
            py::arg("persistent_output_buffer") = std::nullopt,
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Linear,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("cluster_axis") = std::nullopt});
}

}  // namespace detail

void py_bind_all_to_all_async_generic(pybind11::module& module) {
    detail::bind_all_to_all_async_generic(
        module,
        ttnn::experimental::all_to_all_async_generic,
        R"doc(
        Performs an asynchronous all-to-all collective communication operation across multiple devices.

        This operation redistributes data between devices by having each device split its input tensor
        into N chunks (where N is the number of devices) and sending the i-th chunk to device i.
        The operation uses asynchronous kernels for improved performance.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to redistribute.
            in_dim (int): The dimension number to split.
            out_dim (int): The dimension number to concatenate.

        Keyword Args:
            persistent_output_buffer (ttnn.Tensor, optional): Buffer where final output will be written.
            num_links (int, optional): Number of fabric links to use for communication.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for buffers.
            topology (ttnn.Topology, optional): Network topology to use. Defaults to ttnn.Topology.Ring.
            subdevice_id (SubDeviceId, optional): Target specific subdevice for the operation.
            cluster_axis (int, optional): The axis along which to cluster the communication.

        Returns:
            ttnn.Tensor: Output tensor containing redistributed data.

        Example:
            >>> # Redistribute data from dim 2 to dim 3 across 4 devices
            >>> output_buf = ttnn.zeros_like(input_tensor)       # Output buffer
            >>> result = ttnn.experimental.all_to_all_async_generic(
            ...     input_tensor,
            ...     persistent_output_buffer=output_buf,  # Optional buffer
            ...     in_dim=2,
            ...     out_dim=3,
            ...     cluster_axis=1
            ... )
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
