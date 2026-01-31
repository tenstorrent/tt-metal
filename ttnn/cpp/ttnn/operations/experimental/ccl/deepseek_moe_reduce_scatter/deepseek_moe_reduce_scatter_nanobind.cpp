// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "deepseek_moe_reduce_scatter_nanobind.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/deepseek_moe_reduce_scatter.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_deepseek_moe_reduce_scatter(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const std::vector<ttnn::Tensor>& input_tensors,
               const tt::tt_metal::MemoryConfig& output_memory_config,
               int32_t dim,
               uint32_t num_links,
               tt::tt_fabric::Topology topology,
               std::optional<uint32_t> cluster_axis) -> ttnn::Tensor {
                return self(input_tensors, output_memory_config, dim, num_links, topology, cluster_axis);
            },
            nb::arg("input_tensors"),
            nb::arg("output_memory_config"),
            nb::arg("dim"),
            nb::kw_only(),
            nb::arg("num_links") = 4,
            nb::arg("topology") = nb::cast(tt::tt_fabric::Topology::Ring),
            nb::arg("cluster_axis") = nb::none()});
}

}  // namespace

void bind_deepseek_moe_reduce_scatter(nb::module_& mod) {
    bind_deepseek_moe_reduce_scatter(
        mod,
        ttnn::experimental::deepseek_moe_reduce_scatter,
        R"doc(
        Reduce-scatter operation across devices along a selected dimension and optional cluster axis. This operation reduces the mesh tensor across the devices in the mesh, along the specified dimension. It then scatters the reduced tensor back to the devices in the mesh, along the same dimension. When cluster axis is specified, we reduce and scatter along the cluster axis. When it is not specified, then we reduce and scatter across all devices in the mesh.

        Args:
            input_tensors (List of ttnn.Tensor): the input tensors, which when concatted together on the scatter dim repesent the single logical input tensor.
            output_memory_config (ttnn.MemoryConfig): output memory configuration.
            dim (int): dimension along which to scatter.

        Keyword Args:
            num_links (int, optional): the number of links to use for the reduce-scatter operation. Defaults to `None`, for which the number of links is determined automatically.
            cluster_axis (int, optional): the cluster axis to reduce across. Defaults to `None`.
            topology (ttnn.Topology, optional): fabric topology. Defaults to `None`.

        Returns:
            ttnn.Tensor: The reduced and scattered tensor, with output_shape = input_shape (the logical input shape) for all the unspecified dimensions, and output_shape[dim] = input_shape[dim] / num_devices, where num_devices is the number of devices along the `cluster_axis` if specified, else the total number of devices along the mesh.

        Example:
            >>> # ttnn_tensor shape is [1, 1, 32, 7168]
            >>> # num_devices along cluster_axis is 8
            >>> input_tensors = ttnn.split(ttnn_tensor, split_size=896, dim=3)
            >>> output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
            >>> output = ttnn.reduce_scatter(input_tensors, output_memory_config=output_memory_config, dim=3, cluster_axis=1)
            >>> print(output.shape)
            [1, 1, 32, 896]
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
