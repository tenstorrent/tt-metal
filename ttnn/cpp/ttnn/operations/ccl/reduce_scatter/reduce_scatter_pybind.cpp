// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>

#include "ttnn-pybind/decorators.hpp"
#include "reduce_scatter.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void py_bind_reduce_scatter(py::module& module) {
    const auto* doc =
        R"doc(
        Reduce-scatter operation across devices along a selected dimension and optional cluster axis. This operation reduces the mesh tensor across the devices in the mesh, along the specified dimension. It then scatters the reduced tensor back to the devices in the mesh, along the same dimension. When cluster axis is specified, we reduce and scatter along the cluster axis. When it is not specified, then we reduce and scatter across all devices in the mesh. When the layout is row-major or the scatter breaks apart tiles, we use the composite reduce-scatter implementation that falls back to all-broadcast.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to be reduced and scattered.
            dim (int): Dimension along which to reduce.

        Keyword Args:
            cluster_axis (int, optional): The cluster axis to reduce across. Defaults to `None`.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice id for worker cores.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor.
            num_links (int, optional): The number of links to use for the reduce-scatter operation. Defaults to `None`, for which the number of links is determined automatically.
            topology (ttnn.Topology, optional): Fabric topology. Defaults to `None`.

        Returns:
            ttnn.Tensor: The reduced and scattered tensor, with output_shape = input_shape for all the unspecified dimensions, and output_shape[dim] = input_shape[dim] / num_devices, where num_devices is the number of devices along the `cluster_axis` if specified, else the total number of devices along the mesh.

        Example:
            >>> # ttnn_tensor shape is [1, 8, 32, 256]
            >>> # num_devices along cluster_axis is 8
            >>> output = ttnn.reduce_scatter(ttnn_tensor, dim=1, cluster_axis=1)
            >>> print(output.shape)
            [1, 1, 32, 256]
        )doc";

    using OperationType = decltype(ttnn::reduce_scatter);
    ttnn::bind_registered_operation(
        module,
        ttnn::reduce_scatter,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               int32_t dim,
               const std::optional<uint32_t> cluster_axis,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor>& optional_output_tensor,
               const std::optional<uint32_t> num_links,
               const std::optional<tt::tt_fabric::Topology> topology) {
                return self(
                    input_tensor,
                    dim,
                    cluster_axis,
                    subdevice_id,
                    memory_config,
                    optional_output_tensor,
                    num_links,
                    topology);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim"),
            py::kw_only(),
            py::arg("cluster_axis") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("num_links") = std::nullopt,
            py::arg("topology").noconvert() = std::nullopt,
        });
}

}  // namespace ttnn::operations::ccl
